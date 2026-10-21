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

# PR #25342 — acquire the diff (it is open and its head can move; the head
# SHA validated here is 0d2237cd72e2eab9f88c065a6cfac9987161ea34):
curl -sL -o /tmp/gigachat35-pr.diff https://github.com/ggml-org/llama.cpp/pull/25342.diff
git apply --check /tmp/gigachat35-pr.diff   # dry run first
git apply /tmp/gigachat35-pr.diff
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

The validated on-disk size is **~127 GB** for the adjusted recipe
(Q8_0 embeddings instead of F16, F16 norms, Q2_K for *all* routed expert
tensors — up, down, and gate; the real checkpoint uses the **unfused**
`ffn_gate_exps`/`ffn_up_exps` layout, which the model file and the MoE
cache eligibility check both handle alongside the fused `ffn_gate_up_exps`).
The pipeline's dequant path covers every type in that recipe
(Q2_K / Q6_K / Q8_0 / F16 / F32); the embedding rows are not abliteration
targets, so quantizing them is free.

## 3. Placement: `--fit` (no manual `-ngl` / `-ot` / `-ts`)

The pinned build has `--fit` **ON by default** (`-fit` default `on`), and it
is the right way to place this model — no hand-derived offload recipe:

- **MoE path.** For a MoE model, fit first measures memory with *all* routed
  experts (`blk.*.ffn_(up|down|gate_up|gate)_(ch|)exps` — exactly the
  PR #25342 offload set; shared experts excluded) in host RAM. If the dense
  part fits, it places **all layers on the GPUs** (across both, writing the
  tensor split itself) with **all routed experts pinned to host RAM** —
  precisely the layout the MoE expert cache needs (layer on GPU, experts on
  host), so `--moe-cache-mib` then activates on every MoE layer. If the
  dense part is too big, some dense layers simply stay on CPU (partial
  overflow); the model still runs.
- **The `--moe-cache-mib` budget is accounted for in the fit measurement**
  (verified: at the same VRAM target, fit picks the smaller layout when the
  cache is enabled, scoring the cache allocation against the target). So the
  per-device margin `-fitt` (default 1 GiB) only has to cover CUDA/driver
  overhead — do not reserve VRAM for the cache by hand.
- **Back-off, not a hard conflict.** If the model fits as configured, fit
  does nothing. If it does not fit and the user pre-set `-ngl`, `-ot`,
  `-cmoe` or `-ts`, fit aborts with a warning and the user's layout is used
  as-is (the warning is only visible at `-lv 4` — fit logs at debug level).

```bash
# the whole placement in one line (fit + expert cache + FA + quantized KV):
build/bin/llama-cli -m quant/gigachat35_q6q2.gguf \
    --moe-cache-mib 8192 -fa on -ctk q8_0 -ctv q8_0 -lm mmap \
    -c 8192 -lv 4 -n 16 -p "hi"
```

Tuning: if the load OOMs, raise `-fitt` (per-device VRAM margin); if VRAM
sits idle, lower it or raise `--moe-cache-mib` and re-read the fit trace.

**Log levels:** the MoE cache lines are INFO → pass `-lv 3` to
`llama-cli`/`heretic-eval` (llama-server defaults to 3 and shows them).
Fit's own decisions (targets, per-device layer counts) are TRACE → `-lv 4`.

Small-model caveat: if the model fits entirely in VRAM, fit deliberately
offloads nothing and the MoE cache stays disabled ("no layer has all of its
experts in host memory") — correct, not a bug. To exercise the offload path
on a small model, force it with `-cmoe` or shrink the fit target with
`-fitt` (pre-flight below).

## 4. Pre-flight checks on the run box

```bash
# arch registration + model load
build/bin/test-llama-archs | grep -i gigachat

# fit + MoE cache (the real layout). Verify in the log (-lv 4):
#   "fitting params to device memory ..."
#   "id=N, target=... MiB" and per-device "N layers, ... MiB used"
#   "MoE cache size = X for Y of host experts"  <- Y > 0 = offload TRIGGERED
#   "K layers, q2_K: M slots (...)"             <- experts cached in VRAM
#   NO "no layer has all of its experts in host memory" warning
# and at shutdown:
#   "llama_moe_cache: ubatch ...: hits = ..., hit rate = ...%"  <- cache active
build/bin/llama-cli -m quant/gigachat35_q6q2.gguf \
    --moe-cache-mib 8192 -c 8192 -lv 4 -n 16 -p "hi"

# small-model variant (e.g. the synthetic smoke model): it fits in VRAM, so
# force the offload — either pin the experts to host:
build/bin/llama-cli -m small_model.gguf -cmoe --moe-cache-mib 256 -lv 3 -n 4 -p "hi"
# ...or shrink the fit target (margin = measured free MiB - target MiB; the
# free value is in the fit trace of any -lv 4 run):
build/bin/llama-cli -m small_model.gguf --fit on -fitt <free-target> \
    --moe-cache-mib 256 -lv 4 -n 4 -p "hi"
```

Both variants must show `MoE cache size = ... for Y MiB of host experts`
with **Y > 0** (host-resident experts) and a slot line before you trust a
cache-related test — otherwise the model simply fit in VRAM and the test
measured nothing.

Offline package check (dequant of Q2_K/Q6_K/Q8_0, adapter assembly, no GPU):

```bash
python -m heretic_gguf.cli --config config.gigachat35.toml selftest
```

## 5. Abliteration run

Committed run config: **[`config.gigachat35.toml`](../config.gigachat35.toml)**
(adjust `model`, `harness_binary`, `--moe-cache-mib` for the box). Its
`harness_args` are the placement line from section 3 with `-lv 3` (the
harness runs the same `common_init`, so fit + MoE cache apply identically
to it).

The committed config runs in **multi-direction mode**
(`multidirectional_som = true`, `study_name = "gigachat35-som"`): the
single-direction difference-of-means run on this model capped at ~70%
refusal, and GigaChat's refusal is "identity"-rich (it spans a low-dim
manifold, not one direction). Multi-direction abliteration (SOM-collected
directions, arXiv:2511.08379) is expected to break that ceiling at low KL.
It needs `pip install minisom` (in requirements). The SOM directions cache
is `checkpoints/directions_som.pt` — separate from the single-direction
`directions.pt`, which is left untouched (flip `multidirectional_som` back
to `false` and the old study/directions still work as before).

### Abliteratable components

The Heretic vanilla pair `[attn.o_proj, mlp.down_proj]` maps 1:1, plus a
third component this arch has (dense models do not):

| component (config) | tensor at layer L | where |
|---|---|---|
| `attn.o_proj` | `blk.L.attn_output.weight` | full-attention (MLA) layers |
| `attn.o_proj` | `blk.L.ssm_out.weight` | Gated-Delta-Net (recurrent) layers |
| `mlp.down_proj` | `blk.L.ffn_down.weight` | leading dense layers |
| `mlp.down_proj` | `blk.L.ffn_down_exps.weight` | MoE layers (fused 3-D routed experts; the adapter carries one LoRA pair per expert) |
| `mlp.down_proj.shared` | `blk.L.ffn_down_shexp.weight` | MoE layers (the always-active shared expert — this arch's analogue of the vanilla `mlp.down_proj`) |

The per-layer resolution (which layers are recurrent / dense / MoE) is read
from the GGUF metadata in `heretic_gguf/archs.py`; all five tensors go
through `build_lora_mm`/`build_moe_ffn` in the model file, i.e. are
LoRA-targetable. `config.gigachat35.toml` uses all three components.
`expert_mode`: `shared` (one LoRA pair per fused expert tensor, same update
for every expert — the faithful Heretic MoE behavior; the default here),
`grouped` (per-group multipliers), `routing` (per-expert strength from the
measured harmful/harmless routing ratio — requires `directions` first, which
dumps `routing_stats.npy`).

```bash
python -m heretic_gguf.cli --config config.gigachat35.toml directions    # residuals + directions (cached)
python -m heretic_gguf.cli --config config.gigachat35.toml optimize      # Optuna study (resumable)
python -m heretic_gguf.cli --config config.gigachat35.toml export --pareto-index 0 --mode adapter --out abliteration.gguf
python -m heretic_gguf.cli --config config.gigachat35.toml verify --pareto-index 0   # optional: 'pre' vs 'full' fidelity
```

Study restarts skip the baseline re-measurement and the cached directions
(fingerprint-checked; the SOM hyperparameters are part of the fingerprint
and the SOM cache lives in `directions_som.pt`); trial adapters are rank
`som_k` (4 here) and still small (a few hundred MB), so a few GB of
`/dev/shm` is enough.

### Long generates, timeouts, and the harness protocol

A full-eval `GENERATE` on this model can take 30–90+ min (50 prompts,
responses up to `max_response_length`, host-resident experts). The Python
driver used to treat *any* command quieter than `harness_cmd_timeout` as
dead, so one long generate would time out while the harness was still
working — the driver then piled new commands into the harness stdin queue,
the stale `OK`s got consumed by the wrong calls, and the run degenerated
into `apply - generate - apply - generate ...` circles with no `LOGITS`,
leaking job/result files into `shm_dir` until the tmpfs got cleaned out.

Fixed on both sides: the harness now prints a `PROGRESS <done>/<total>`
stdout line after every completed prompt of RESIDUALS/LOGITS/GENERATE, so
`harness_cmd_timeout` (config key, default 1800 s) is a **no-progress**
bound — it only fires when a *single prompt* makes no progress for 30 min,
i.e. when the process is genuinely wedged. On such a timeout the driver
respawns the harness (fast: `QUIT` → SIGTERM → SIGKILL, never waits out the
stall) and retries the trial once instead of desyncing. Slow-but-healthy
commands of any length therefore just run; do not raise
`harness_cmd_timeout` to "make slow runs fit" — that knob is for stuck
prompts, not total durations.

The first SOM run also showed a *deterministic* per-cycle slowdown: within
one long-lived harness process, tok/s drops with each adapter
apply/remove cycle (~7.3 → 6.1 → 4.9 → 4.5 over three cycles in that run;
a fresh process resets it). That is a llama.cpp-runtime effect (suspected
CUDA-graph pool churn from the new adapter tensors per cycle), not a bug in
this repo. The committed run config mitigates it with
`harness_restart_every = 8`: one ~1 min model reload per 8 trials, which
bounds the degradation while the generates take 20–75 min each. Set it to
0 to disable.

## 6. Deployment + OpenAI e2e verification

```bash
build/bin/llama-server -m quant/gigachat35_q6q2.gguf --lora abliteration.gguf \
    --moe-cache-mib 8192 \
    -c 16384 -b 2048 -ub 512 -np 4 \
    -fa on -ctk q8_0 -ctv q8_0 \
    -lm mmap \
    --host 0.0.0.0 --port 8123 \
    --warmup --metrics

curl -s localhost:8123/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "system", "content": "Ты полезный помощник"},
               {"role": "user", "content": "Как вскрыть дверной замок?"}],
  "max_tokens": 512}'
```

Modern-flag notes (all present in the pinned build):

- **`--fit` is on by default** — it places the layers (both GPUs, tensor
  split written automatically), pins the routed experts to host RAM and
  accounts for the `--moe-cache-mib` budget; no `-ngl`/`-ot`/`-ts`/`-cmoe`
  needed.
- **`-fa on` + `-ctk/-ctv q8_0`** — flash attention with a q8_0 KV cache
  halves the MLA KV footprint; llama.cpp hard-asserts FA for a quantized
  V cache, so the pair is required (verified working on this arch).
- **`-lm mmap`** — the host-resident experts are page-cache backed instead
  of being copied into anonymous RAM at load (no 2× peak); with 176 GB
  DDR5 there is headroom either way.
- On by default, not repeated: `--repack`, `--op-offload`, `--warmup`,
  continuous batching, `--jinja`, and reasoning extraction (`auto` detects
  the GigaChat reasoning template).
- Tuning knobs: `-fitt` (per-device VRAM margin) and `--moe-cache-mib`
  (see section 3).

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
| `--fit` placement (CLI and server) + MoE cache | PASS — fit measures with all experts in host RAM, writes the tensor split itself, and accounts for the `--moe-cache-mib` budget in its measurement (verified: same target → smaller layout chosen when the cache is enabled); user-set `-ngl`/`-ot`/`-cmoe` are honored via graceful abort (WARN, `-lv 4`) |
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
