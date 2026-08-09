# heretic-gguf

[Heretic](https://github.com/p-e-w/heretic)-style, Optuna-optimized directional
ablation ("abliteration") **directly on quantized GGUF models**, using
llama.cpp as the inference runtime.

Stock Heretic loads models through `transformers` — which means full bf16
weights (or a bitsandbytes 4-bit copy that still needs far more memory than
the quantized size suggests) and, for some recent architectures, no
`transformers` implementation at all. heretic-gguf instead works on the GGUF
you already run: the base file is **never modified and never rewritten during
the search**. Every Optuna trial's ablation is expressed as a small
**LoRA adapter GGUF** (rank 1; per-expert 3-D for MoE tensors) written to
tmpfs and hot-swapped into a running llama.cpp harness — the same mathematical
overlay Heretic uses (its `abliterate()` writes rank-1 LoRA A/B into PEFT
adapters), applied here by llama.cpp's native LoRA path, which supports
**per-expert A/B slices** on fused expert tensors. Only the single final
configuration is materialized — either as a merged GGUF (copy-and-patch,
requantized in place) or as a permanent adapter you can pass to
`llama-server --lora`.

Primary development target: **DeepSeek-V4-Flash-0731** (`deepseek4`, 284B
total / 13B active MoE, UD-Q4_K_XL), where loading the original weights is not
an option and no `transformers` implementation exists. Also validated
end-to-end on **Qwen3.6-35B-A3B** (`qwen35moe`); other architectures can be
added via a small `ArchSpec` (see `heretic_gguf/archs.py`).

## Why heretic-gguf: measured memory footprint

Peak memory while abliterating the **same model** (Qwen3.6-35B-A3B, a 35B
MoE; stock Heretic @ 7675b90 vs this port, 60 direction + 50 eval prompts,
a few trials):

| | Stock Heretic, bf16 | Stock Heretic, `bnb_4bit` | heretic-gguf (UD-Q4_K_XL) |
|---|---|---|---|
| VRAM after load | ~70 GiB (4 GPUs) | ~68 GiB (4 GPUs) | ~24 GiB (1 GPU) |
| Peak VRAM (directions + trials) | **96 GiB** | **86 GiB** | **26 GiB** |
| Peak RAM (process RSS) | 43 GiB | 66 GiB | 21 GiB harness (mmap, page-cache backed) + 9.5 GiB driver |

(raw `nvidia-smi` sums, including a ~3 GiB desktop baseline on GPU 0.)

The `bnb_4bit` option saves almost nothing here: bitsandbytes only
quantizes `nn.Linear` modules, and transformers implements the routed
experts of recent MoE models as fused 3-D `nn.Parameter` tensors — so all
~32B expert parameters (the bulk of the model) stay in bf16, and the
"4-bit" load trims just 2.5 GiB. On top of the weights, Heretic's auto-tuned
batch size drives large eval transients (full-sequence logits over the
248k-token vocab, all-layer hidden states during residual capture, KV
cache), the bnb load itself peaks at 66 GiB RAM from CPU staging, and
finishing a study reloads the entire model in bf16 on CPU for the final
merge. heretic-gguf runs the same search on the 22 GB GGUF you already
inference with — one GPU, no second copy of the weights, ever.

## Features

- Heretic-faithful port: difference-of-means residual directions, optional
  winsorization and orthogonalization, the full Optuna parameter space
  (`direction_scope`, float `direction_index`, per-component weight kernels),
  row normalization modes `none` / `pre` / `full`, keyword-rate + KL
  multi-objective TPE with journal checkpointing.
- Per-expert ablation strength for MoE models: `shared`, `grouped`
  (Optuna-controlled expert groups), or `routing` (driven by measured
  harmful/harmless expert-routing frequencies, zero extra Optuna dimensions).
- Two-stage evaluation: rank all trials on a cheap prompt set, then re-score
  only the Pareto top-K with the full eval set.
- Lossless output option: export the winning trial as a tiny LoRA adapter
  GGUF (recommended), or as a fully merged, requantized GGUF.
- Resumable studies (Optuna journal + cached baseline — a restart skips the
  baseline re-measurement and goes straight to trials), per-trial JSONL logs,
  provenance embedded in exported adapters, live status dashboard.
- Supported architectures: `deepseek4`, `qwen35moe` (more via
  `heretic_gguf/archs.py`).

## Requirements

- Linux with enough VRAM+RAM to run your GGUF in llama.cpp (the same hardware
  you already inference on — no extra copy of the weights is ever made).
- Python ≥ 3.11 with `torch` (CUDA recommended — the one-time projection pass
  is ~10× faster on GPU), plus `numpy`, `gguf`, `optuna`, `datasets`, `rich`
  (installed via pyproject, see below).
- A llama.cpp checkout with the `heretic-eval` harness built (next section).
- A few GB free in `/dev/shm` (tmpfs) for trial adapters.
- For `export --mode merged`: free disk roughly equal to the model size.

## Installation

### 1. Python package

```bash
git clone <this repo> && cd heretic-gguf
python -m venv .venv && source .venv/bin/activate   # or use conda

# 1a. torch: a CUDA build is strongly recommended (the one-time projection
# pass is ~10x faster on GPU). If your pip default is CPU-only, install the
# wheel matching your CUDA version from https://pytorch.org FIRST —
# requirements.txt just says "torch" and would pull the CPU build.

# 1b. the remaining dependencies, then the package itself
pip install -r requirements.txt
pip install .
```

`pip install .` puts the `heretic-gguf` command on your PATH (inside the
venv/conda env — activate it in each new shell). If the command is not found,
`python -m heretic_gguf.cli` is a drop-in equivalent for every command in
this README.

### 2. Build llama.cpp with the `heretic-eval` harness

The harness (`harness/heretic-eval/`) is an in-tree llama.cpp example: a
long-lived process that keeps the model loaded and is driven over
stdin/stdout (capture per-layer residuals, dump logits, batched greedy
generation, LoRA hot-swap). It touches semi-internal llama.cpp APIs
(`cb_eval`, the LoRA adapter init signature) that change between upstream
commits, so **build against the commit pinned in
[`LLAMA_CPP_PIN`](LLAMA_CPP_PIN)** unless you are prepared to re-validate:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout $(grep -E '^[0-9a-f]{40}$' /path/to/heretic-gguf/LLAMA_CPP_PIN)

# install the harness as an example
cp -r /path/to/heretic-gguf/harness/heretic-eval examples/
git apply /path/to/heretic-gguf/harness/examples-CMakeLists.patch
# (the patch is one line: add_subdirectory(heretic-eval) in examples/CMakeLists.txt)

# build (CUDA; drop -DGGML_CUDA=ON for CPU-only)
cmake -B build -DGGML_CUDA=ON
cmake --build build --target heretic-eval -j
```

The binary ends up at `build/bin/heretic-eval`; point `harness_binary` in your
config at it.

## Quickstart

```bash
cp config.example.toml config.toml   # adjust model path, harness_binary, GPU flags

# one command does it all: sanity-checks config/paths, computes the refusal
# directions (skipped automatically when a cached directions.pt matches the
# current model/datasets/template — recompute with --force-directions), runs
# the resumable Optuna study with the live dashboard, and prints export
# instructions at the end
heretic-gguf --config config.toml run

# export the chosen trial as a permanent LoRA adapter (tiny, lossless, instant)
heretic-gguf --config config.toml export --mode adapter \
    --trial 137 --out my-model-heretic-lora.gguf
#    run:  llama-server -m <base gguf> --lora my-model-heretic-lora.gguf <flags>
```

`export` without `--trial` uses `--pareto-index N` into the sorted Pareto
front; `--mode merged --out <dir>` writes a full requantized GGUF instead
(needs model-sized free disk).

The individual steps `run` composes are also available on their own:

```bash
# offline sanity check (no harness; dequant + ablation math + adapter write)
heretic-gguf --config config.toml selftest

# collect residuals and compute refusal directions (one-time; with
# expert_mode = "routing" this also dumps routing stats for that mode)
heretic-gguf --config config.toml directions

# run the Optuna study (resumable via checkpoints/*.journal)
heretic-gguf --config config.toml optimize

# optional: fidelity check — rebuild the winning trial with the exact
# row_normalization="full" path and compare keyword rate + KL
heretic-gguf --config config.toml verify --trial 137
```

### Dataset caching

`hf:` dataset specs are downloaded once and cached as jsonl (one file per
full spec — repo, split, slice, column) under `dataset_cache_dir` (default:
`<study_checkpoint_dir>/../datasets`). Warm starts read the cache and make
zero network calls, so `HF_HUB_OFFLINE=1` works once the cache is populated;
each spec logs whether it was loaded from cache or fetched.
`--refresh-datasets` (or `refresh_datasets = true` in the config) forces a
re-download. `file:` specs are local already and are never cached.

### Custom prompt sets

Every dataset key (`good_prompts`, `bad_prompts`, `eval_harmful_prompts`,
`eval_harmless_prompts`) accepts either a single spec string or a **list of
specs**, mixed freely between `hf:` and `file:`:

```toml
eval_harmful_prompts = [
    "hf:mlabonne/harmful_behaviors:test[:100]:text",
    "file:my_jailbreaks.txt",          # one prompt per line, UTF-8
]
```

A spec's own `[:N]` slice applies to that spec only; the loaded prompts are
concatenated and deduplicated (first occurrence wins). When the eval set
mixes origins, the refusal display splits per source — e.g.
`23/52 = 44.23% [13/32 (base), 10/20 (user)]` (`base` = `hf:` specs,
`user` = `file:` specs). See the annotated example in
[`config.example.toml`](config.example.toml) and the template file
[`examples/custom_prompts.txt`](examples/custom_prompts.txt). Note: the
cached directions/baseline fingerprint the spec *strings* — editing the
contents of an existing `file:` path does not invalidate them (rerun
`directions` / delete `<study>.baseline.pt` after editing a prompt file).

### Live status screen

`run` and `optimize` open a persistent rich dashboard (banner, trial progress
with elapsed/ETA, a live top-10 Pareto table, and the last few event lines)
instead of scrolling one line per trial — the directions phase reports its
progress in the same dashboard. While the dashboard is up, the harness's
verbose llama.cpp stderr is teed to `checkpoints/<study>.harness.log`. Use
`--plain` (or redirect stdout / run without a tty) to get the old line-based
output — both modes write the same `checkpoints/*.trials.jsonl` records.

### Two-stage evaluation

Generation dominates trial wall time, so the study can rank every trial on a
cheap **Stage A** (default: 32 harmful prompts × 64 tokens) and then
automatically re-score a selection of trials with the full eval set in a
**Stage B** pass after the study. KL is prefill-only and always uses the full
harmless set. Stage B scores are stored in the `stage_b_scores` study
attribute and shown next to Stage A scores in trial summaries. Enable in
`config.toml`:

```toml
two_stage = true
stage_a_harmful_prompts = "hf:mlabonne/harmful_behaviors:test[:32]:text"
stage_a_max_response_length = 64
stage_b_top_k = 10
```

Stage B selection is configurable via `stage_b_select`: `"pareto"` (default)
fills non-dominated fronts in order up to `stage_b_top_k`; `"threshold"`
re-scores every trial whose Stage A scores pass `stage_b_max_keyword_rate`
and `stage_b_max_kl` (best first, still capped at `stage_b_top_k`) — useful
because Stage A keyword rates are optimistic and the full-set ranking can
differ markedly from the cheap one. Specific trials can always be re-scored
on demand, regardless of selection mode:

```bash
heretic-gguf --config config.toml optimize --stage-b-trials 116,103
```

If a two-stage study is stopped before `n_trials`, run Stage B on its own
with `optimize --stage-b-only` (skips the search, re-scores the currently
selected trials with the full keyword set).

(Optuna pruners don't support multi-objective studies, so this is a manual
second pass rather than `trial.report`-based pruning.)

## Per-expert ablation (`expert_mode`)

- `shared` — Heretic-classic: one strength for all experts.
- `grouped` — experts split into contiguous groups, each with its own Optuna
  multiplier ∈ [0, 1.25]. Cheap, TPE-friendly, usually the right start.
- `routing` — strengths ∝ measured harmful/harmless routing-frequency ratio,
  **zero extra Optuna dimensions**. The harness accumulates per-layer
  `ffn_moe_topk` expert-id histograms over all prompt tokens during residual
  collection; running `directions` with `expert_mode = "routing"` writes the
  ratio to `routing_stats_file` (default `checkpoints/routing_stats.npy`),
  which `optimize`/`export`/`verify` then consume.

Note the residual *direction* is a property of the pre-MoE collapsed stream,
so per-expert *directions* are not measurable; per-expert **strength** is the
meaningful axis, and it is fully supported end-to-end (3-D adapter slices).

## What is faithfully ported from Heretic

- Difference-of-means residual directions on first-token hidden states,
  optional winsorization, optional orthogonalization against the good mean
  (grimjim projected abliteration).
- The full Optuna parameter space: `direction_scope` (global / per layer),
  float `direction_index` with lerp between adjacent layer directions, and the
  per-component kernel `max_weight / max_weight_position / min_weight /
  min_weight_distance` with the MLP lower bound at −0.25 clamped to 0
  (Heretic issue #202 behavior). The `max_weight` search ranges default to
  Heretic's hardcoded values but are configurable
  (`attn_max_weight_range` / `mlp_max_weight_range`), and a fresh study can
  be seeded with the top trials of a previous one (`seed_from_*` /
  `seed_top_k`).
- Row normalization modes `none` / `pre` / `full` (norm-preserving biprojected
  abliteration with `svd_lowrank`, rank configurable) applied to
  `W` dequantized from the GGUF.
- Scorers: refusal keyword rate (Heretic's marker list, extended with
  simplified/traditional Chinese refusal markers) on held-out harmful prompts
  with greedy ≤100-token responses; first-token KL divergence against the
  unablated baseline on held-out harmless prompts.
- Multi-objective TPE (`multivariate=True`, 60 startup trials), journal
  checkpointing, Pareto-front trial selection.

## Deliberate deviations

1. **Residual capture point (hyper-connection archs).** DeepSeek-V4 widens
   the residual to 4 streams; we capture the collapsed d_model block input
   (`hc_attn_pre-N`) instead of a layer-output hidden state, so per-layer
   scope pairs layer L with `directions[L+1]` and skips the last layer.
2. **Quantized numerics.** Directions and Ŵ come from dequantized weights;
   the KL baseline is measured on the quantized model, so the objective is
   self-consistent. Absolute KL values are not comparable to a bf16 Heretic
   run (they never are across runtimes anyway).
3. **Merged export requantizes** edited tensors (one extra quantization step
   vs Heretic's fp16 export). `--mode adapter` avoids this entirely and is
   the recommended output.
4. Default `row_normalization = "pre"` (fast, cacheable) instead of Heretic's
   `"full"` (supported, but reprojects weights per trial — slower).

## Reproducibility & bookkeeping

- Every trial is appended as one greppable JSON line (params, scores,
  duration, stage) to `checkpoints/<study>.trials.jsonl`, alongside the
  Optuna journal: `grep '"trial": 137' checkpoints/*.trials.jsonl`.
- The study's user attributes record the exact **git commit of llama.cpp**
  and of this repo, a config snapshot, and the baseline scores.
- Exported adapter GGUFs embed the same provenance as `adapter.heretic.*`
  KV keys (study name, trial number, `direction_index`, the full parameter
  set, scores, commit hashes). llama.cpp ignores unknown keys; inspect with
  `strings adapter.gguf | grep adapter.heretic`.
- The compatible llama.cpp commit is pinned in
  [`LLAMA_CPP_PIN`](LLAMA_CPP_PIN).

## Repository layout

| Path | What |
|---|---|
| `heretic_gguf/` | Python package: GGUF I/O, fast dequant (torch), direction computation, ablation assembly, Optuna loop, scorers, CLI, live status UI |
| `harness/heretic-eval/` | C++ llama.cpp example: the evaluation harness (see Installation) |
| `harness/examples-CMakeLists.patch` | one-line patch registering the harness in llama.cpp's example build |
| `config.example.toml` | annotated configuration template |
| `requirements.txt` | pip dependencies (mirror of `pyproject.toml`) |
| `examples/` | template user prompt file (custom dataset specs) |

## Known limitations

- Trial wall time is dominated by generation: 100 prompts × 100 tokens takes
  minutes per trial on large models. Use `two_stage = true` (Stage A ranks
  cheaply, Stage B re-scores only the top-K) before committing to a full
  200-trial study.
- `full` row normalization is exact but slow (per-trial streaming dequant +
  SVD). Use `pre` for the study, then run `verify --trial N` on the winner to
  confirm `pre` vs `full` makes no material difference.
- The study CLI is non-interactive (Heretic's chat/menu UX was not ported);
  pick trials via `export --trial/--pareto-index`.
- llama.cpp applies LoRA scale as `alpha/rank` per adapter file; we set
  `alpha = rank` (strength exactly 1). Per-tensor scaling is baked into B.
- Only output projections are ablated (`attn.o_proj`, `mlp.down_proj`,
  optional shared expert) — same scope as Heretic.
- Quantization support: MXFP4 and Q8_0 are on the fast torch path (verified,
  byte-lossless requant); other K-quants fall back to the slower pip `gguf`
  dequant; imatrix quants (IQ*/TQ*) are unsupported.

## License

heretic-gguf is a derivative work of
[Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann —
its algorithm, parameter space, scorer logic, and refusal keyword list were
ported directly from Heretic's source. Like Heretic, it is licensed under
the **GNU Affero General Public License v3.0 or later**; see
[LICENSE](LICENSE). The name is used in reference to the original project;
all credit for the method goes to its author.
