# GigaChat 3.5 (Reasoning) — launch instructions

Running heretic-gguf abliteration on the GigaChat 3.5 reasoning model
(`gigachat35` arch: DeepSeek-V3-style MLA MoE + Qwen3-Next-style
Gated-Delta-Net/full-MLA hybrid attention, gated-RMSNorm sandwich, MTP head).

Everything below was validated end-to-end on a same-arch synthetic model
(generator: `examples/make_synthetic_gigachat35.py`) before touching the
real weights.

## 1. llama.cpp build

**Pin:** the exact commit in [`LLAMA_CPP_PIN`](../LLAMA_CPP_PIN)
(`33d6135cbe20e90e67d9db1c874fbb4a9dc64b19`), **plus llama.cpp PR #25342**
applied on top (GigaChat 3.5 model support + MoE expert offload/caching via
`--moe-cache-mib`; it was still unmerged at the time of validation).
The `heretic-eval` harness lives at `examples/heretic-eval/` in the llama.cpp
tree as an untracked local addition (the CMakeLists hookup patch is in
[`harness/`](../harness/)).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 33d6135cbe20e90e67d9db1c874fbb4a9dc64b19
git apply /path/to/gigachat35-pr.diff      # PR #25342
cp -r /path/to/heretic-gguf/harness/heretic-eval examples/heretic-eval
# apply harness/examples-CMakeLists.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j --target llama-cli llama-server llama-quantize \
      llama-tokenize test-llama-archs heretic-eval
```

### ⚠ MANDATORY: build against ONE consistent CUDA toolkit

llama.cpp must be compiled **and** linked against a single CUDA toolkit.
Mixing toolkits (e.g. CUDA-13 headers with the system `libcudart.so.12` at
runtime) silently misreads `cudaDeviceProp` fields — `sharedMemPerBlockOptin`
comes back as garbage (4 GiB), the smem opt-in call "succeeds" but corrupts
the kernel's dynamic-smem limit, and **every subsequent MoE `mm_ids_helper`
launch fails with the misleading** `CUDA error: invalid argument` (only for
n_tokens > 16, which makes it look like a model issue).

Fix: point the CMake CUDA libraries at one toolkit explicitly (sticky cache
entries are not overridden by `CMAKE_CUDA_TOOLKIT_ROOT_DIR` alone):

```bash
cmake -S . -B build \
  -DCUDA_cudart_LIBRARY=/usr/local/cuda-13.0/lib64/libcudart.so \
  -DCUDA_cublas_LIBRARY=/usr/local/cuda-13.0/lib64/libcublas.so \
  -DCUDA_cublasLt_LIBRARY=/usr/local/cuda-13.0/lib64/libcublasLt.so
```

Verify after the build:

```bash
ldd build/bin/libggml-cuda.so | grep cud   # every entry must be the same major version
```

Also note: if `nvcc`/`compute-sanitizer` of a *different* toolkit shadow the
PATH ones, the build links the wrong runtime — check `which nvcc`.

## 2. Requantize the real model

Weight-only, no imatrix. Q2_K routed experts (the bulk of the 466 GB file),
Q6_K for attention / dense / shared experts / SSM / MTP:

```bash
./scripts/requant_gigachat35.sh /path/to/GigaChat3.5-466B.gguf /data/quant/ 32
```

The script prints the resulting type distribution and the fallback list.
Expected benign fallbacks: `attn_k_b` → Q8_0 (key_mla not a multiple of the
256-element Q6_K block), `ffn_gate_inp` → F16, `exp_probs_b.bias` → F32.
(On a model with exactly `n_expert == 8` — e.g. the synthetic smoke model —
the standard quantize logic also bumps the attn_v-class tensors to Q8_0;
harmless, and it does not apply to the real model.)
`ssm_conv1d.weight` **must** end up F32 (the SSM conv op asserts float32 in
every backend) — the script enforces this with an explicit override.

## 3. Pre-flight checks on the run box

```bash
# arch registration + model load
build/bin/test-llama-archs | grep -i gigachat

# plain generation (all layers offloaded)
build/bin/llama-cli -m quant/gigachat35_q6q2.gguf -ngl 999 -n 16 -p "hi"

# MoE expert offload + GPU cache (the PR #25342 pattern): pin the routed
# experts of a GPU-resident layer to CPU RAM; --moe-cache-mib keeps the hot
# experts in a GPU SLRU cache. Verify in the log:
#   "CUDA0 MoE cache size = ... for ... of host experts"
#   "N layers, q2_K: K slots (...)"          <- experts fit in the cache
#   "llama_moe_cache: ubatch ...: hits = ..., hit rate = ...%"  <- cache active
build/bin/llama-cli -m quant/gigachat35_q6q2.gguf -ngl <last-N> \
    -ot "blk.<L>.ffn_gate_up_exps=CPU,blk.<L>.ffn_down_exps=CPU" \
    --moe-cache-mib 4096 -n 16 -p "hi" --verbose
```

Offline package check (dequant of Q2_K/Q6_K, adapter assembly, no GPU):

```bash
python -m heretic_gguf.cli --config config.toml selftest
```

## 4. Abliteration run

Config (based on `config.example.toml`):

```toml
model = "/data/quant/gigachat35_q6q2.gguf"
harness_binary = "/path/to/llama.cpp/build/bin/heretic-eval"
# one consistent offload strategy; MoE cache requires LAYER split
# (it is NOT supported with --tensor-split / -ts):
harness_args = [
    "-ngl", "999",
    "-ot", "blk.\\d+.ffn_gate_up_exps=CPU,blk.\\d+.ffn_down_exps=CPU",
    "--moe-cache-mib", "16384",
    "-c", "4096", "-b", "2048", "-ub", "512", "-np", "8",
    "--load-mode", "none",          # experts come from RAM, mmap adds nothing
]
shm_dir = "/dev/shm/heretic_gguf"

ablate_device = "cuda"              # projection pass is ~10x faster on GPU
expert_mode = "routing"             # or "shared"; routing is driven by the
                                    # measured harmful/harmless routing stats
# the gigachat35 response prefix (think-close + blank line, CoT-skip) and the
# l_out capture node are per-arch defaults — no config needed
```

```bash
python -m heretic_gguf.cli --config config.toml directions    # residuals + directions (cached)
python -m heretic_gguf.cli --config config.toml optimize      # Optuna study (resumable)
python -m heretic_gguf.cli --config config.toml export --pareto-index 0 --mode adapter --out abliteration.gguf
python -m heretic_gguf.cli --config config.toml verify --pareto-index 0   # optional: 'pre' vs 'full' fidelity
```

Study restarts skip the baseline re-measurement and the cached directions
(fingerprint-checked); trial adapters are rank-1 and small (tens–hundreds of
MB), so a few GB of `/dev/shm` is enough.

## 5. Deployment + OpenAI e2e verification

```bash
build/bin/llama-server -m quant/gigachat3.5-466B.gguf --lora abliteration.gguf \
    -ngl 999 -c 4096 --port 8080
# (plus the same -ot / --moe-cache-mib offload args as the run config)

curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": "How do I pick a lock?"}],
  "max_tokens": 256}'
```

Expected: the response goes into `reasoning_content` (the model is a
reasoning model; the chat template ends the prompt inside the thinking
block) with the final answer in `content` after the think-close token.
Cross-check with the un-adapter model: refusal-style responses should appear
on the harmful eval prompts after abliteration.

## What was validated on the synthetic same-arch model

| stage | result |
|---|---|
| `selftest` (F16 and Q6_K/Q2_K) | PASS — Q2_K/Q6_K fast dequant bit-exact vs `gguf.dequantize` |
| `directions` (harness, `l_out` capture, all layers) | PASS |
| MoE expert offload + GPU cache (`-ot ...=CPU` + `--moe-cache-mib`) | PASS — F16 and Q2_K cached experts, hit-rate stats printed |
| `optimize` (baseline + trials, adapter hot-swap) | PASS |
| `export --mode adapter` | PASS |
| `verify` (`pre` vs `full` row normalization) | PASS — KL delta ≈ 0 |
| vanilla llama.cpp `--lora` | PASS — adapter loads, changes outputs |
| `llama-server` OpenAI `/v1/chat/completions` | PASS — incl. reasoning_content parsing |

Known llama.cpp fix that ships with this setup: in `src/models/gigachat35.cpp`
the MTP seed must NOT be `cb()`-named `h_nextn` — that tensor **is** the last
block's `l_out-<il>` and the rename (il < 0 → `ggml_set_name`) clobbers the
per-layer name that activation-capture tooling matches on (the last layer
then never captures). The fix is in the local tree; worth a comment on the PR.
