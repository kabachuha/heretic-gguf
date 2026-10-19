// heretic-eval: REPL-style evaluation harness for activation/weight-ablation
// experiments, driven from Python over stdin/stdout.
//
// Protocol (after model load):
//   stdout: "READY"
//   then one single-line command per line on stdin; for each command exactly one
//   response line on stdout: "OK" or "ERR <message>" (RESIDUALS/LOGITS print an
//   extra "DIMS ..." line before OK). Human-readable progress goes to stderr only.
//
// Commands:
//   ADAPTER <path.gguf>   load + apply LoRA adapter (scale 1.0), replacing the current one
//   ADAPTER NONE          remove the current adapter
//   RESIDUALS <job.json>  capture last-token "<capture_node>-<il>" activations per layer
//                         (default node "hc_attn_pre" = deepseek4; override with the
//                          HERETIC_CAPTURE_NODE env var, e.g. "l_out" for qwen35moe)
//   LOGITS    <job.json>  last-token logits per prompt
//   GENERATE  <job.json>  greedy batched generation
//   QUIT                  exit(0)
//
// Job JSON: {"prompts": [{"system": "...", "user": "..."}], "prefix": "<think></think>",
//            "out": "/dev/shm/res.f32", "max_tokens": 100,
//            "routing_out": "/dev/shm/routing.u32", "n_experts": 256}   // optional, RESIDUALS only
//
// Routing statistics (optional RESIDUALS extension): when the job carries
// "routing_out" + "n_experts", every "ffn_moe_topk-<il>" node (the per-layer
// top-k expert ids, named via cb() in build_moe_ffn, src/llama-graph.cpp:1918)
// seen during the prefills is accumulated into a (n_layer, n_experts) u32
// histogram over ALL prompt tokens, written to routing_out with a {n_layer,
// n_experts} i32 header. Works for learned-gate layers (I32 view into the
// argsort result, read per-column) and hash layers (I32 get_rows of the
// tid2eid table, contiguous).
//
// Key implementation facts (verified against llama.cpp master, see LLAMA_CPP_PIN):
// - Tensor naming: model code calls cb(cur, "<node>", il) in its graph builder
//   (e.g. "hc_attn_pre" in src/models/deepseek4.cpp:1118, "l_out" in
//   src/models/qwen35moe.cpp:225 and src/models/gigachat35.cpp:328);
//   llm_graph_context::cb forwards to the graph callback installed in
//   llama_context::graph_get_cb, which names the tensor with
//   ggml_format_name(cur, "%s-%d", name, il) -> runtime name "<node>-<il>".
// - Per-layer capture: llama_context_params.cb_eval / cb_eval_user_data (include/llama.h),
//   forwarded by common_context_params_to_llama. Callback signature
//   bool (*)(ggml_tensor *, bool ask, void *); ask=true before compute, ask=false after compute,
//   when data is readable via ggml_backend_tensor_get (see examples/eval-callback, common/debug.cpp).
// - ggml_backend_tensor_get is a *linear* memcpy of size bytes from tensor->data + offset
//   -- it does NOT de-stride non-contiguous tensors. We therefore only capture
//   ggml_is_contiguous tensors; for a contiguous [n_embd, n_tokens] tensor the last-token
//   column is simply the last n_embd elements, readable with one strided call at offset
//   (nelements - n_embd) * elsize.
// - LoRA hot-swap: llama_adapter_lora_init(model, path) [2 args],
//   llama_set_adapters_lora(ctx, &a, 1, &scale), llama_adapter_lora_free. llama-server's
//   SET_LORA handler just updates the adapter list and applies it via
//   common_set_adapter_lora -> llama_set_adapters_lora; no KV clear is required -- the new
//   adapter applies on the next decode.
// - Decoding: the old llama_decode()/llama_batch pair was replaced by
//   llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, llama_batch_ext*). We use the
//   common_batch wrapper (common/common.h) and chunk via get_sub_batch(i, n),
//   halving n on failure (KV full) -- same approach as examples/parallel/parallel.cpp.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

#include <atomic>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// activation capture via cb_eval
// ---------------------------------------------------------------------------

// Residual capture node: the cb() node whose "<name>-<il>" instances hold the
// per-layer residual stream ("hc_attn_pre" on deepseek4, "l_out" on most
// other archs — grep src/models/<arch>.cpp for cb(). Overridable via the
// HERETIC_CAPTURE_NODE env var so one binary serves every arch.
static std::string g_capture_node = "hc_attn_pre";
static std::string g_capture_fmt  = "hc_attn_pre-%d%c";  // sscanf pattern

struct capture_state {
    std::atomic<bool> active{false};     // only capture while a RESIDUALS prefill is running

    int32_t n_embd  = 0;
    int32_t n_layer = 0;

    std::mutex mtx;
    std::map<int32_t, std::vector<float>> layers;  // il -> [n_embd] last-token activation
    int64_t n_skipped_noncontig = 0;
    int64_t n_skipped_badtype   = 0;

    // routing-statistics capture (ffn_moe_topk-<il> expert-id histograms)
    std::atomic<bool> routing_active{false};
    int32_t routing_n_experts = 0;
    std::vector<uint32_t> routing_hist;            // (n_layer, routing_n_experts), row-major
    int64_t n_skipped_topk_type = 0;
};

// convert one raw element to float (only the types a norm output can plausibly be)
static bool elem_to_float(const uint8_t * p, ggml_type type, float & out) {
    switch (type) {
        case GGML_TYPE_F32:  out = *(const float *)        p; return true;
        case GGML_TYPE_F16:  out = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p); return true;
        case GGML_TYPE_BF16: out = ggml_bf16_to_fp32(*(const ggml_bf16_t *) p); return true;
        default: return false;
    }
}

// accumulate one ffn_moe_topk-<il> node ([n_expert_used, n_tokens] I32 expert
// ids) into the routing histogram
static void capture_routing(capture_state * cs, ggml_tensor * t, int il) {
    if (!cs->routing_active.load(std::memory_order_relaxed)) {
        return;
    }
    if (il < 0 || il >= cs->n_layer) {
        return;
    }
    if (t->type != GGML_TYPE_I32) {
        cs->n_skipped_topk_type++;
        return;
    }

    const int64_t k        = t->ne[0];
    const int64_t n_tokens = k > 0 ? ggml_nelements(t) / k : 0; // expect [k, n_tokens]
    if (k < 1 || n_tokens < 1 || cs->routing_n_experts <= 0) {
        return;
    }

    std::vector<int32_t> ids(k * n_tokens);
    if (ggml_is_contiguous(t)) {
        // hash layers: get_rows of the tid2eid table is contiguous
        ggml_backend_tensor_get(t, ids.data(), 0, ids.size() * sizeof(int32_t));
    } else {
        // learned-gate layers: ffn_moe_topk is a view into the argsort result,
        // so token column j lives at offset j*nb[1] (ggml_backend_tensor_get
        // is a linear memcpy and cannot de-stride on its own)
        for (int64_t j = 0; j < n_tokens; ++j) {
            ggml_backend_tensor_get(t, ids.data() + j * k, j * t->nb[1], k * sizeof(int32_t));
        }
    }

    std::lock_guard<std::mutex> lock(cs->mtx);
    uint32_t * hist = cs->routing_hist.data() + (size_t) il * cs->routing_n_experts;
    for (const int32_t id : ids) {
        if (id >= 0 && id < cs->routing_n_experts) {
            hist[id]++;
        }
    }
}

static bool capture_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * cs = (capture_state *) user_data;

    if (ask) {
        return true; // always allow computation
    }

    if (!cs->active.load(std::memory_order_relaxed)) {
        return true;
    }

    // match "ffn_moe_topk-<il>" exactly (routing-statistics capture)
    {
        int il = -1;
        char tail = 0;
        if (sscanf(t->name, "ffn_moe_topk-%d%c", &il, &tail) == 1) {
            capture_routing(cs, t, il);
            return true;
        }
    }

    // match "<capture_node>-<il>" exactly (runtime naming: ggml_format_name(cur, "%s-%d", ...))
    int il = -1;
    char tail = 0;
    if (sscanf(t->name, g_capture_fmt.c_str(), &il, &tail) != 1) {
        return true;
    }
    if (il < 0 || il >= cs->n_layer) {
        return true; // e.g. MTP/nextn layers beyond n_layer
    }

    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = ggml_nelements(t) / n_embd; // expect [n_embd, n_tokens]

    if (n_embd != cs->n_embd || n_tokens < 1) {
        LOG_WRN("%s: unexpected shape for %s: ne = [%lld, %lld, %lld, %lld]\n", __func__,
                t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        return true;
    }

    if (!ggml_is_contiguous(t)) {
        // ggml_backend_tensor_get is a linear memcpy and cannot de-stride views/padded tensors.
        // If this ever triggers, the capture for this layer is left as NaN and the model's
        // graph builder should be patched to name a contiguous tensor instead.
        cs->n_skipped_noncontig++;
        return true;
    }

    const size_t elsize = ggml_type_size(t->type);
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        cs->n_skipped_badtype++;
        return true;
    }

    // contiguous => last-token column is the last n_embd elements
    std::vector<uint8_t> buf(n_embd * elsize);
    ggml_backend_tensor_get(t, buf.data(), (ggml_nelements(t) - n_embd) * elsize, buf.size());

    std::vector<float> vec(n_embd);
    for (int64_t i = 0; i < n_embd; ++i) {
        elem_to_float(buf.data() + i * elsize, t->type, vec[i]);
    }

    {
        std::lock_guard<std::mutex> lock(cs->mtx);
        cs->layers[il] = std::move(vec);
    }

    return true;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

struct prompt_pair {
    std::string system;
    std::string user;
};

struct job_spec {
    std::vector<prompt_pair> prompts;
    std::string prefix;
    std::string out;
    int32_t     max_tokens = 100;

    // optional routing-statistics capture (RESIDUALS only)
    std::string routing_out;
    int32_t     n_experts = 0;
};

static bool parse_job(const std::string & path, job_spec & job, std::string & err, bool need_max_tokens) {
    std::ifstream f(path);
    if (!f) {
        err = "cannot open job file: " + path;
        return false;
    }

    json j;
    try {
        j = json::parse(f);
    } catch (const std::exception & e) {
        err = std::string("invalid job JSON: ") + e.what();
        return false;
    }

    try {
        for (const auto & p : j.at("prompts")) {
            prompt_pair pp;
            pp.system = p.value("system", "");
            pp.user   = p.at("user").get<std::string>();
            job.prompts.push_back(std::move(pp));
        }
        job.prefix = j.value("prefix", "");
        job.out    = j.at("out").get<std::string>();
        if (need_max_tokens) {
            job.max_tokens = j.value("max_tokens", 100);
        }
        job.routing_out = j.value("routing_out", "");
        job.n_experts   = j.value("n_experts", 0);
    } catch (const std::exception & e) {
        err = std::string("malformed job JSON: ") + e.what();
        return false;
    }

    if (job.prompts.empty()) {
        err = "job has no prompts";
        return false;
    }

    return true;
}

// render one chat prompt with the jinja chat template + verbatim prefix,
// then tokenize. Matches llama-cli's chat path (tools/completion/completion.cpp:305).
static std::vector<llama_token> render_and_tokenize(
        llama_context * ctx,
        const common_chat_templates * tmpls,
        bool use_jinja,
        const prompt_pair & pp,
        const std::string & prefix) {

    common_chat_templates_inputs inputs;
    inputs.use_jinja = use_jinja;
    inputs.add_generation_prompt = true;

    if (!pp.system.empty()) {
        common_chat_msg sys;
        sys.role    = "system";
        sys.content = pp.system;
        inputs.messages.push_back(std::move(sys));
    }

    common_chat_msg user;
    user.role    = "user";
    user.content = pp.user;
    inputs.messages.push_back(std::move(user));

    std::string text = common_chat_templates_apply(tmpls, inputs).prompt.str();
    text += prefix;

    // parse_special = true so the prefix's "<think></think>" becomes special tokens
    return common_tokenize(ctx, text, /* add_special = */ true, /* parse_special = */ true);
}

// chunked prefill of a token sequence on seq_id 0 (memory must be cleared beforehand).
// logits are requested only for the final token.
static bool prefill(llama_context * ctx, const std::vector<llama_token> & tokens, std::string & err) {
    if ((int32_t) tokens.size() > (int32_t) llama_n_ctx(ctx)) {
        err = "prompt too long: " + std::to_string(tokens.size()) + " tokens > n_ctx " + std::to_string(llama_n_ctx(ctx));
        return false;
    }

    common_batch batch(ctx);
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        const bool last = (i == (int32_t) tokens.size() - 1);
        batch.add(tokens[i], i, 0, last);
    }

    // process in chunks of llama_n_batch, halving on failure (KV full)
    int32_t n_batch = (int32_t) llama_n_batch(ctx);
    for (int32_t i = 0; i < batch.size(); ) {
        const int32_t n = std::min(n_batch, batch.size() - i);
        const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(i, n));
        if (ret != 0) {
            if (n_batch == 1 || ret < 0) {
                err = "llama_process failed during prefill, ret = " + std::to_string(ret);
                return false;
            }
            n_batch /= 2;
            continue;
        }
        i += n;
        n_batch = (int32_t) llama_n_batch(ctx);
    }

    return true;
}

static bool write_f32_file(const std::string & path, const std::vector<int32_t> & header, const std::vector<float> & data, std::string & err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot open output file: " + path;
        return false;
    }
    f.write((const char *) header.data(), header.size() * sizeof(int32_t));
    f.write((const char *) data.data(),   data.size()   * sizeof(float));
    if (!f) {
        err = "failed writing output file: " + path;
        return false;
    }
    return true;
}

static bool write_u32_file(const std::string & path, const std::vector<int32_t> & header, const std::vector<uint32_t> & data, std::string & err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        err = "cannot open output file: " + path;
        return false;
    }
    f.write((const char *) header.data(), header.size() * sizeof(int32_t));
    f.write((const char *) data.data(),   data.size()   * sizeof(uint32_t));
    if (!f) {
        err = "failed writing output file: " + path;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

static std::string cmd_adapter(
        llama_context * ctx,
        llama_model * model,
        llama_adapter_lora *& cur_adapter,
        const std::string & arg) {

    if (arg == "NONE") {
        if (cur_adapter) {
            // mirror of llama-server hot-swap: empty adapter list, then free (no KV clear needed)
            llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
            llama_adapter_lora_free(cur_adapter);
            cur_adapter = nullptr;
            LOG_INF("adapter removed\n");
        }
        return "OK";
    }

    llama_adapter_lora * adapter = llama_adapter_lora_init(model, arg.c_str());
    if (!adapter) {
        return "ERR failed to load adapter: " + arg;
    }

    float scale = 1.0f;
    llama_adapter_lora * adapters[1] = { adapter };
    if (llama_set_adapters_lora(ctx, adapters, 1, &scale) != 0) {
        llama_adapter_lora_free(adapter);
        return "ERR llama_set_adapters_lora failed";
    }

    if (cur_adapter) {
        llama_adapter_lora_free(cur_adapter);
    }
    cur_adapter = adapter;

    LOG_INF("adapter applied: %s (scale 1.0)\n", arg.c_str());
    return "OK";
}

static std::string cmd_residuals(
        llama_context * ctx,
        capture_state & cs,
        const common_chat_templates * tmpls,
        bool use_jinja,
        const std::string & job_path) {

    job_spec job;
    std::string err;
    if (!parse_job(job_path, job, err, false)) {
        return "ERR " + err;
    }

    const int32_t n_prompts = (int32_t) job.prompts.size();
    const int32_t n_layer   = cs.n_layer;
    const int32_t n_embd    = cs.n_embd;

    const bool want_routing = !job.routing_out.empty();
    if (want_routing) {
        if (job.n_experts <= 0) {
            return "ERR routing_out requires n_experts > 0 in the job JSON";
        }
        std::lock_guard<std::mutex> lock(cs.mtx);
        cs.routing_n_experts = job.n_experts;
        cs.routing_hist.assign((size_t) n_layer * job.n_experts, 0);
        cs.n_skipped_topk_type = 0;
        cs.routing_active.store(true);
    }

    std::vector<float> out((size_t) n_prompts * n_layer * n_embd);

    for (int32_t p = 0; p < n_prompts; ++p) {
        auto tokens = render_and_tokenize(ctx, tmpls, use_jinja, job.prompts[p], job.prefix);

        LOG_INF("RESIDUALS: prompt %d/%d, %zu tokens\n", p + 1, n_prompts, tokens.size());

        llama_memory_clear(llama_get_memory(ctx), true);

        {
            std::lock_guard<std::mutex> lock(cs.mtx);
            cs.layers.clear();
        }
        cs.n_skipped_noncontig = 0;
        cs.n_skipped_badtype   = 0;

        cs.active.store(true);
        const bool ok = prefill(ctx, tokens, err);
        cs.active.store(false);

        if (!ok) {
            cs.routing_active.store(false);
            return "ERR " + err;
        }

        {
            std::lock_guard<std::mutex> lock(cs.mtx);
            for (int32_t il = 0; il < n_layer; ++il) {
                float * dst = out.data() + ((size_t) p * n_layer + il) * n_embd;
                auto it = cs.layers.find(il);
                if (it != cs.layers.end()) {
                    memcpy(dst, it->second.data(), n_embd * sizeof(float));
                } else {
                    // capture failed for this layer -> NaNs + warning (see header comment)
                    LOG_WRN("RESIDUALS: no capture for layer %d (prompt %d); writing NaNs\n", il, p);
                    for (int32_t i = 0; i < n_embd; ++i) {
                        dst[i] = std::nanf("");
                    }
                }
            }
        }

        if (cs.n_skipped_noncontig > 0) {
            LOG_WRN("RESIDUALS: skipped %lld non-contiguous %s tensors (ggml_backend_tensor_get cannot de-stride views)\n",
                    (long long) cs.n_skipped_noncontig, g_capture_node.c_str());
        }
        if (cs.n_skipped_badtype > 0) {
            LOG_WRN("RESIDUALS: skipped %lld %s tensors with unsupported element type\n",
                    (long long) cs.n_skipped_badtype, g_capture_node.c_str());
        }
    }

    if (!write_f32_file(job.out, { n_prompts, n_layer, n_embd }, out, err)) {
        cs.routing_active.store(false);
        return "ERR " + err;
    }

    LOG_INF("RESIDUALS: wrote %s (%d prompts x %d layers x %d embd)\n", job.out.c_str(), n_prompts, n_layer, n_embd);

    if (want_routing) {
        cs.routing_active.store(false);
        if (cs.n_skipped_topk_type > 0) {
            LOG_WRN("RESIDUALS: skipped %lld ffn_moe_topk tensors with non-I32 type\n",
                    (long long) cs.n_skipped_topk_type);
        }
        std::vector<uint32_t> hist;
        {
            std::lock_guard<std::mutex> lock(cs.mtx);
            hist = cs.routing_hist;
        }
        if (!write_u32_file(job.routing_out, { n_layer, job.n_experts }, hist, err)) {
            return "ERR " + err;
        }
        uint64_t total = 0;
        for (const auto c : hist) {
            total += c;
        }
        LOG_INF("RESIDUALS: wrote routing histogram %s (%d layers x %d experts, %llu total routed tokens)\n",
                job.routing_out.c_str(), n_layer, job.n_experts, (unsigned long long) total);
    }

    char dims[128];
    snprintf(dims, sizeof(dims), "DIMS %d %d %d", n_prompts, n_layer, n_embd);
    return std::string(dims) + "\nOK";
}

static std::string cmd_logits(
        llama_context * ctx,
        const llama_vocab * vocab,
        const common_chat_templates * tmpls,
        bool use_jinja,
        const std::string & job_path) {

    job_spec job;
    std::string err;
    if (!parse_job(job_path, job, err, false)) {
        return "ERR " + err;
    }

    const int32_t n_prompts = (int32_t) job.prompts.size();
    const int32_t n_vocab   = llama_vocab_n_tokens(vocab);

    std::vector<float> out((size_t) n_prompts * n_vocab);

    for (int32_t p = 0; p < n_prompts; ++p) {
        auto tokens = render_and_tokenize(ctx, tmpls, use_jinja, job.prompts[p], job.prefix);

        LOG_INF("LOGITS: prompt %d/%d, %zu tokens\n", p + 1, n_prompts, tokens.size());

        llama_memory_clear(llama_get_memory(ctx), true);

        if (!prefill(ctx, tokens, err)) {
            return "ERR " + err;
        }

        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            return "ERR llama_get_logits_ith returned NULL";
        }

        memcpy(out.data() + (size_t) p * n_vocab, logits, n_vocab * sizeof(float));
    }

    if (!write_f32_file(job.out, { n_prompts, n_vocab }, out, err)) {
        return "ERR " + err;
    }

    LOG_INF("LOGITS: wrote %s (%d prompts x %d vocab)\n", job.out.c_str(), n_prompts, n_vocab);

    char dims[128];
    snprintf(dims, sizeof(dims), "DIMS %d %d", n_prompts, n_vocab);
    return std::string(dims) + "\nOK";
}

// ---------------------------------------------------------------------------
// GENERATE: continuous batching over N slots, greedy argmax, no sampler chain.
// Adapted from examples/parallel/parallel.cpp.
// ---------------------------------------------------------------------------

struct gen_slot {
    llama_seq_id seq_id    = -1;
    int32_t      prompt_idx = -1;

    llama_token  sampled    = 0;
    int32_t      n_past     = 0;
    int32_t      n_decoded  = 0;
    int32_t      gen_limit  = 0;
    int32_t      i_batch    = -1;

    bool         active     = false;
    std::string  response;
};

static llama_token sample_argmax(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

static std::string cmd_generate(
        llama_context * ctx,
        const common_params & params,
        const llama_vocab * vocab,
        const common_chat_templates * tmpls,
        bool use_jinja,
        const std::string & job_path) {

    job_spec job;
    std::string err;
    if (!parse_job(job_path, job, err, true)) {
        return "ERR " + err;
    }

    const int32_t n_prompts = (int32_t) job.prompts.size();
    const int32_t n_vocab   = llama_vocab_n_tokens(vocab);
    const int32_t n_slots   = std::min(std::max(1, params.n_parallel), 32);
    const int32_t n_ctx_seq = llama_n_ctx_seq(ctx);

    auto * mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    // tokenize all prompts up front
    std::vector<std::vector<llama_token>> prompts_tokens(n_prompts);
    for (int32_t p = 0; p < n_prompts; ++p) {
        prompts_tokens[p] = render_and_tokenize(ctx, tmpls, use_jinja, job.prompts[p], job.prefix);
        if ((int32_t) prompts_tokens[p].size() >= n_ctx_seq) {
            return "ERR prompt " + std::to_string(p) + " too long: " +
                   std::to_string(prompts_tokens[p].size()) + " tokens >= n_ctx_seq " + std::to_string(n_ctx_seq);
        }
    }

    std::vector<std::string> responses(n_prompts);
    std::vector<gen_slot> slots(n_slots);
    for (int32_t i = 0; i < n_slots; ++i) {
        slots[i].seq_id = i;
    }

    int32_t next_prompt = 0;
    int32_t n_done      = 0;

    // batch rebuilt every iteration, processed in chunks of params.n_batch
    // (parallel.cpp approach)
    common_batch batch(ctx);

    const auto t_start = ggml_time_us();

    while (n_done < n_prompts) {
        batch.clear();

        // 1 token per active slot (decode step)
        for (auto & s : slots) {
            if (!s.active) {
                continue;
            }
            s.i_batch = batch.size();
            batch.add(s.sampled, s.n_past++, s.seq_id, true);
            s.n_decoded += 1;
        }

        // fill free slots with new prompts (prefill); the whole prompt goes into the
        // same batch and is processed by the chunked decode loop below
        for (auto & s : slots) {
            if (s.active || next_prompt >= n_prompts) {
                continue;
            }

            const auto & toks = prompts_tokens[next_prompt];

            s.prompt_idx = next_prompt;
            s.n_past     = 0;
            s.n_decoded  = 0;
            s.response.clear();

            s.gen_limit = std::min(job.max_tokens, n_ctx_seq - (int32_t) toks.size() - 1);
            if (s.gen_limit < job.max_tokens) {
                LOG_WRN("GENERATE: prompt %d generation clamped to %d tokens (n_ctx_seq = %d)\n",
                        next_prompt, s.gen_limit, n_ctx_seq);
            }

            for (size_t i = 0; i < toks.size(); ++i) {
                batch.add(toks[i], s.n_past++, s.seq_id, false);
            }
            // logits only for the last prompt token
            if (batch.size() > 0) {
                batch.set_output(batch.size() - 1, true);
            }
            s.i_batch = batch.size() - 1;
            s.active  = true;

            LOG_INF("GENERATE: slot %d <- prompt %d (%zu tokens)\n", s.seq_id, next_prompt, toks.size());

            next_prompt += 1;
        }

        if (batch.size() == 0) {
            break;
        }

        // process in chunks of n_batch, halving on failure (KV full) as in parallel.cpp
        int32_t n_batch = params.n_batch;
        int32_t i_next  = 0;
        for (int32_t i = 0; i < batch.size(); i = i_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - i);

            const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get_sub_batch(i, n_tokens));
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    return "ERR llama_process failed (KV cache full? increase -c), ret = " + std::to_string(ret);
                }
                n_batch /= 2;
                continue;
            }

            i_next = i + n_tokens;
            n_batch = params.n_batch;

            // sample every slot whose logits were computed in this chunk
            for (auto & s : slots) {
                if (!s.active || s.i_batch < i || s.i_batch >= i + n_tokens) {
                    continue;
                }

                const float * logits = llama_get_logits_ith(ctx, s.i_batch - i);
                const llama_token id = sample_argmax(logits, n_vocab);

                s.i_batch = -1;

                if (llama_vocab_is_eog(vocab, id) || s.n_decoded >= s.gen_limit) {
                    responses[s.prompt_idx] = s.response;
                    llama_memory_seq_rm(mem, s.seq_id, -1, -1);
                    s.active = false;
                    n_done += 1;

                    LOG_INF("GENERATE: slot %d done (prompt %d, %d tokens)\n", s.seq_id, s.prompt_idx, s.n_decoded);
                    continue;
                }

                s.response += common_token_to_piece(ctx, id, /* special = */ false);
                s.sampled   = id;
            }
        }
    }

    const double dt = (ggml_time_us() - t_start) / 1e6;
    LOG_INF("GENERATE: %d prompts in %.2f s\n", n_prompts, dt);

    json out;
    out["responses"] = responses;

    std::ofstream f(job.out, std::ios::trunc);
    if (!f) {
        return "ERR cannot open output file: " + job.out;
    }
    // Generated text can contain incomplete UTF-8 (byte-level BPE pieces cut
    // at the token limit, or lone byte tokens from a degenerate model), which
    // makes json::dump throw type_error.316 and kills the process. Replace
    // invalid sequences with U+FFFD instead — the Python scorer only does
    // keyword matching, so a replacement char is harmless.
    f << out.dump(2, ' ', false, json::error_handler_t::replace) << "\n";
    if (!f) {
        return "ERR failed writing output file: " + job.out;
    }

    return "OK";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.warmup = false; // no warmup pass: avoids spurious captures and speeds up startup

    common_init();

    // LLAMA_EXAMPLE_CLI: accept the same flag set as llama-cli, including
    // --jinja / --chat-template-file (arg.cpp:3424, arg.cpp:3502)
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // residual capture node override (arch-specific graph node name; see top comment)
    if (const char * env = getenv("HERETIC_CAPTURE_NODE")) {
        if (env[0] == '\0' || strchr(env, '%')) {
            LOG_ERR("%s: invalid HERETIC_CAPTURE_NODE value: '%s'\n", __func__, env);
            return 1;
        }
        g_capture_node = env;
        g_capture_fmt  = g_capture_node + "-%d%c";
    }

    // install the activation-capture callback before context creation
    // (common_context_params_to_llama forwards it into llama_context_params.cb_eval)
    capture_state cs;
    params.cb_eval           = capture_cb_eval;
    params.cb_eval_user_data = &cs;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    cs.n_embd  = llama_model_n_embd(model);
    cs.n_layer = llama_model_n_layer(model);

    // jinja chat templates (uses --chat-template-file / --chat-template override if given)
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
    const bool use_jinja = params.use_jinja;

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("heretic-eval ready: n_layer = %d, n_embd = %d, n_vocab = %d, n_ctx = %d, n_parallel = %d, capture_node = %s\n",
            cs.n_layer, cs.n_embd, llama_vocab_n_tokens(vocab), llama_n_ctx(ctx), params.n_parallel,
            g_capture_node.c_str());

    llama_adapter_lora * cur_adapter = nullptr;

    printf("READY\n");
    fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        // trim trailing whitespace/CR
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }

        std::string resp;

        const size_t sp = line.find(' ');
        const std::string cmd = sp == std::string::npos ? line : line.substr(0, sp);
        const std::string arg = sp == std::string::npos ? ""  : line.substr(sp + 1);

        if (cmd == "QUIT") {
            break;
        } else if (cmd == "ADAPTER") {
            if (arg.empty()) {
                resp = "ERR usage: ADAPTER <path.gguf|NONE>";
            } else {
                resp = cmd_adapter(ctx, model, cur_adapter, arg);
            }
        } else if (cmd == "RESIDUALS") {
            resp = cmd_residuals(ctx, cs, chat_templates.get(), use_jinja, arg);
        } else if (cmd == "LOGITS") {
            resp = cmd_logits(ctx, vocab, chat_templates.get(), use_jinja, arg);
        } else if (cmd == "GENERATE") {
            resp = cmd_generate(ctx, params, vocab, chat_templates.get(), use_jinja, arg);
        } else {
            resp = "ERR unknown command: " + cmd;
        }

        printf("%s\n", resp.c_str());
        fflush(stdout);
    }

    if (cur_adapter) {
        llama_adapter_lora_free(cur_adapter);
    }

    llama_backend_free();

    return 0;
}
