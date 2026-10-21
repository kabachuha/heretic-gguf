# SPDX-License-Identifier: AGPL-3.0-or-later
# CLI entry point: run | directions | optimize | export | verify | selftest

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .ablate import Ablater, routing_ratio
from .archs import COMPONENT_ATTN_OUT, COMPONENT_MLP_DOWN, ArchSpec, get_arch_spec
from . import config
from .config import Settings
from .directions import compute_directions, compute_som_directions, winsorize
from .gguf_io import GGUFModel
from .harness import Harness, read_routing_counts
from .optimize import Optimizer, pareto_trials, stage_b_scores, trial_summary
from .scorers import DEFAULT_KEYWORDS, KLDivergence, KeywordRate, Score


def _load_settings(args) -> Settings:
    s = Settings.from_toml(args.config) if args.config else Settings()
    if getattr(args, "model", None):
        s.model = args.model
    if getattr(args, "refresh_datasets", False):
        s.refresh_datasets = True
    return s


def _open_model(settings: Settings) -> GGUFModel:
    if not settings.model:
        sys.exit("error: model path required (--model or config)")
    return GGUFModel(settings.model)


def _arch_spec(settings: Settings, model: GGUFModel) -> ArchSpec:
    """Resolve the model's ArchSpec and apply per-arch defaults for config
    keys left empty (currently just response_prefix)."""
    spec = get_arch_spec(model.arch, model)
    if not settings.response_prefix and spec.response_prefix:
        settings.response_prefix = spec.response_prefix
    return spec


def _open_harness(settings: Settings, spec: ArchSpec, stderr=None) -> Harness:
    args = ["-m", settings.model] + list(settings.harness_args)
    if settings.chat_template_file:
        args += ["--jinja", "--chat-template-file", settings.chat_template_file]
    # The residual capture node travels via env var so one harness binary
    # serves every registered arch (no rebuild per arch).
    capture_node = settings.capture_node or spec.capture_node
    return Harness(settings.harness_binary, args, shm_dir=settings.shm_dir,
                   stderr=stderr, env_extra={"HERETIC_CAPTURE_NODE": capture_node},
                   cmd_timeout=settings.harness_cmd_timeout)


def _make_ablater(settings: Settings, model: GGUFModel,
                  directions: torch.Tensor) -> Ablater:
    return Ablater(
        model,
        directions,
        row_normalization=settings.row_normalization,
        full_rank=settings.full_normalization_lora_rank,
        components=settings.components,
        device=settings.ablate_device,
        expert_chunk=settings.expert_chunk,
        seed=settings.seed,
    )


def _make_kw_scorer(settings: Settings, harness: Harness,
                    specs: str | list[str], max_tokens: int) -> KeywordRate:
    """KeywordRate over one dataset spec or a list of them. Mixed lists
    (hf: + file:) get a per-origin refusal breakdown in the score display,
    e.g. "23/52 = 44% [13/32 (base), 10/20 (user)]"."""
    return KeywordRate(
        harness, settings.prompts(specs),
        keywords=settings.keyword_markers,
        prefix=settings.response_prefix,
        max_tokens=max_tokens,
        groups=settings.prompt_groups(specs),
    )


def _directions_path(settings: Settings) -> Path:
    # The two collection modes produce different tensors (single direction
    # vs. per-layer direction SET), so each mode gets its own cache file and
    # they never invalidate each other.
    stem = "directions_som" if settings.multidirectional_som else "directions"
    return Path(settings.study_checkpoint_dir) / f"{stem}.pt"


def _directions_meta_path(settings: Settings) -> Path:
    return _directions_path(settings).with_suffix(".meta.json")


def _directions_fingerprint(settings: Settings, model: GGUFModel,
                            spec: ArchSpec) -> dict:
    """The inputs that determine the contents of directions.pt (and of the
    routing stats, which are collected in the same pass). If any of these
    change, cached directions must be recomputed. Call AFTER _arch_spec so
    per-arch defaults (response_prefix) are already resolved."""
    return {
        "model": str(Path(settings.model).resolve()),
        "good_prompts": settings.good_prompts,
        "bad_prompts": settings.bad_prompts,
        "system_prompt": settings.system_prompt,
        "response_prefix": settings.response_prefix,
        "chat_template_file": settings.chat_template_file,
        "capture_node": settings.capture_node or spec.capture_node,
        "expert_mode": settings.expert_mode,
        "orthogonalize_direction": settings.orthogonalize_direction,
        "winsorization_quantile": settings.winsorization_quantile,
        # Multi-direction mode changes what the cache stores (a per-layer
        # direction SET) and the SOM hyperparameters change its contents.
        "multidirectional_som": settings.multidirectional_som,
        "som_x": settings.som_x,
        "som_y": settings.som_y,
        "som_iterations": settings.som_iterations,
        "som_lr": settings.som_lr,
        "som_sigma": settings.som_sigma,
        "som_k": settings.som_k,
        # Residuals numerically depend on the harness compute flags
        # (e.g. -ctk q8_0, -fa on) and on the harness build itself.
        "harness_binary": settings.harness_binary,
        "harness_args": list(settings.harness_args),
    }


def _fingerprint_hash(fingerprint: dict) -> str:
    return hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def _write_directions_meta(settings: Settings, fingerprint: dict) -> None:
    meta = {
        "sha256": _fingerprint_hash(fingerprint),
        "inputs": fingerprint,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _directions_meta_path(settings).write_text(json.dumps(meta, indent=2))


def _directions_cache_status(settings: Settings,
                             fingerprint: dict) -> tuple[bool, str]:
    """(True, "") if the cached directions.pt may be reused, else
    (False, human-readable reason to recompute)."""
    if not _directions_path(settings).exists():
        return False, "no cached directions.pt"
    meta_path = _directions_meta_path(settings)
    if not meta_path.exists():
        return False, ("directions.pt predates input metadata — cannot "
                       "verify it matches the current config")
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, "unreadable directions metadata sidecar"
    if meta.get("sha256") != _fingerprint_hash(fingerprint):
        old = meta.get("inputs", {})
        changed = [k for k in fingerprint if old.get(k) != fingerprint[k]]
        return False, "inputs changed: " + ", ".join(changed or ["unknown"])
    if (settings.expert_mode == "routing"
            and not settings.routing_stats_path().exists()):
        return False, "routing stats missing for expert_mode='routing'"
    return True, ""


def _baseline_path(settings: Settings) -> Path:
    return (Path(settings.study_checkpoint_dir)
            / f"{settings.study_name}.baseline.pt")


def _baseline_fingerprint(settings: Settings) -> dict:
    """The inputs that determine the optimize baseline (keyword-rate
    responses and KL first-token logits measured on the unablated model).
    If any of these change, a cached baseline must be recomputed. Call AFTER
    _arch_spec so per-arch defaults (response_prefix) are already resolved."""
    return {
        "model": str(Path(settings.model).resolve()),
        "eval_harmful_prompts": settings.eval_harmful_prompts,
        "eval_harmless_prompts": settings.eval_harmless_prompts,
        "system_prompt": settings.system_prompt,
        "response_prefix": settings.response_prefix,
        "chat_template_file": settings.chat_template_file,
        "max_response_length": settings.max_response_length,
        "keyword_markers": settings.keyword_markers or DEFAULT_KEYWORDS,
        # The harness configuration (incl. --jinja / chat template source)
        # affects the measured logits, so it invalidates the cache too.
        "harness_binary": settings.harness_binary,
        "harness_args": list(settings.harness_args),
    }


def _baseline_cache_status(settings: Settings,
                           fingerprint: dict) -> tuple[dict | None, str]:
    """(cache, "") if the cached baseline may be reused, else
    (None, human-readable reason to recompute)."""
    path = _baseline_path(settings)
    if not path.exists():
        return None, "no cached baseline"
    try:
        cache = torch.load(path, weights_only=True)
    except Exception:
        return None, "unreadable baseline cache"
    if cache.get("sha256") != _fingerprint_hash(fingerprint):
        old = cache.get("inputs", {})
        changed = [k for k in fingerprint if old.get(k) != fingerprint[k]]
        return None, "inputs changed: " + ", ".join(changed or ["unknown"])
    return cache, ""


def _write_baseline_cache(settings: Settings, fingerprint: dict,
                          baseline_kw: Score, responses: list[str],
                          kl_logits: np.ndarray) -> Path:
    """Persist everything needed to reconstruct the baseline without the
    harness: keyword score + per-prompt baseline responses + the baseline
    first-token logits (fp16 — ~2 bytes/token of vocab per prompt)."""
    path = _baseline_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sha256": _fingerprint_hash(fingerprint),
        "inputs": fingerprint,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "keyword": {"value": baseline_kw.value, "display": baseline_kw.display,
                    "responses": list(responses)},
        "kl_logits": torch.from_numpy(np.asarray(kl_logits)).to(torch.float16),
    }, path)
    return path


def _load_study(settings: Settings):
    """Re-open the Optuna study from its journal (read-only usage)."""
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    journal = Path(settings.study_checkpoint_dir) / f"{settings.study_name}.journal"
    if not journal.exists():
        sys.exit(f"error: no study journal at {journal} (run `optimize` first)")
    storage = JournalStorage(JournalFileBackend(str(journal)))
    return optuna.load_study(study_name=settings.study_name, storage=storage)


def _pick_trial(study, trial_number: int | None, pareto_index: int):
    if trial_number is not None:
        for t in study.trials:
            if t.number == trial_number:
                return t
        sys.exit(f"trial #{trial_number} not found")
    return pareto_trials(study)[pareto_index]


def _load_routing_scale(settings: Settings, ablater: Ablater) -> np.ndarray:
    """Per-expert strength map for expert_mode='routing', from the stats file
    written by `directions` (fail with a clear message if it is missing)."""
    path = settings.routing_stats_path()
    if not path.exists():
        sys.exit(f"error: expert_mode='routing' needs routing stats at {path} "
                 f"— run `directions` first (it collects them when "
                 f"expert_mode='routing'), or set routing_stats_file")
    return ablater.expert_scale_from_routing(np.load(path))


def _trial_ablation_args(settings: Settings, ablater: Ablater, trial):
    """Reconstruct (direction_index, parameters, expert_scale) of a finished
    trial from its user attrs / params."""
    from .ablate import AbliterationParameters

    # from_dict accepts both the legacy scalar form (pre-SOM studies) and the
    # per-direction list form.
    parameters = {
        k: AbliterationParameters.from_dict(v)
        for k, v in trial.user_attrs["parameters"].items()
    }
    direction_index = trial.user_attrs["direction_index"]
    expert_scale = None
    if settings.expert_mode == "grouped":
        mults = np.array(
            [trial.params.get(f"expert_group.{i}", 1.0)
             for i in range(settings.expert_groups)],
            dtype=np.float32,
        )
        expert_scale = ablater.expert_scale_from_groups(mults, settings.expert_groups)
    elif settings.expert_mode == "routing":
        expert_scale = _load_routing_scale(settings, ablater)
    return direction_index, parameters, expert_scale


def _adapter_metadata(settings: Settings, study, trial) -> dict:
    """Trial provenance embedded into exported adapter GGUFs as
    `adapter.heretic.*` keys (llama.cpp ignores unknown KV keys)."""
    from .version import heretic_gguf_commit, llama_cpp_commit

    meta: dict[str, str | int | float | None] = {
        "study_name": settings.study_name,
        "trial_number": trial.number,
        "trial_index": trial.user_attrs.get("index"),
        "direction_scope": trial.params.get("direction_scope"),
        "direction_index": trial.user_attrs.get("direction_index"),
        "row_normalization": settings.row_normalization,
        "expert_mode": settings.expert_mode,
        "components": ",".join(settings.components),
        "seed": settings.seed,
        "llama_cpp_commit": study.user_attrs.get("llama_cpp_commit")
                            or llama_cpp_commit(settings.harness_binary),
        "heretic_gguf_commit": study.user_attrs.get("heretic_gguf_commit")
                               or heretic_gguf_commit(),
    }
    for name, value in trial.user_attrs.get("scores", {}).items():
        meta[f"score.{name}"] = value
    for name, value in stage_b_scores(study).get(trial.number, {}).items():
        meta[f"stage_b_score.{name}"] = value
    for key, value in trial.params.items():
        if key == "direction_scope":
            continue
        meta[f"param.{key}"] = value
    return meta


def _compute_directions(settings: Settings, model: GGUFModel, spec: ArchSpec,
                        log=print, phase=None, harness_stderr=None) -> None:
    """Collect residuals with the harness and write directions.pt (plus the
    routing stats when expert_mode='routing'). Shared by `directions` and
    `run`; in live-dashboard mode log/phase go to the reporter and the
    harness's stderr is teed away from the display."""
    phase = phase or log
    log(f"model: {model.arch}, {model.n_layers} layers, "
        f"{model.n_experts} experts, d={model.n_embd} "
        f"(capture: {settings.capture_node or spec.capture_node}-<layer>)")
    want_routing = settings.expert_mode == "routing"
    if want_routing and model.n_experts <= 0:
        sys.exit("error: expert_mode='routing' but the model has no experts")
    with _open_harness(settings, spec, stderr=harness_stderr) as harness:
        good = settings.prompts(settings.good_prompts)
        bad = settings.prompts(settings.bad_prompts)
        phase(f"collecting residuals: {len(good)} good, {len(bad)} bad prompts"
              + (" (+ routing histograms)" if want_routing else ""))
        if want_routing:
            shm = Path(settings.shm_dir)
            counts_g_path = shm / f"routing_good_{os.getpid()}.u32"
            counts_b_path = shm / f"routing_bad_{os.getpid()}.u32"
            res_g = harness.residuals(good, settings.response_prefix,
                                      routing_out=counts_g_path,
                                      n_experts=model.n_experts)
            res_b = harness.residuals(bad, settings.response_prefix,
                                      routing_out=counts_b_path,
                                      n_experts=model.n_experts)
        else:
            res_g = harness.residuals(good, settings.response_prefix)
            res_b = harness.residuals(bad, settings.response_prefix)
    g = winsorize(torch.from_numpy(res_g), settings.winsorization_quantile)
    b = winsorize(torch.from_numpy(res_b), settings.winsorization_quantile)
    if settings.multidirectional_som:
        try:
            from .som import SOMCalculator  # noqa: F401 (import check)
        except ModuleNotFoundError:
            sys.exit("error: multidirectional_som = true requires the "
                     "'minisom' package (pip install minisom)")
        log(f"multi-direction mode: {settings.som_k} directions/layer via a "
            f"{settings.som_x}x{settings.som_y} SOM "
            f"({settings.som_iterations} iters, lr={settings.som_lr}, "
            f"sigma={settings.som_sigma})")
        directions = compute_som_directions(
            g, b,
            k=settings.som_k, som_x=settings.som_x, som_y=settings.som_y,
            iterations=settings.som_iterations, lr=settings.som_lr,
            sigma=settings.som_sigma,
            orthogonalize=settings.orthogonalize_direction,
            progress=phase,
        )
    else:
        directions = compute_directions(g, b, settings.orthogonalize_direction)
    path = _directions_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, path)
    # Geometry summary (reduced port of Heretic's print_residual_geometry).
    cos = torch.nn.functional.cosine_similarity(
        g.mean(0), b.mean(0), dim=1)
    log(f"good/bad mean cosine per layer: min={cos.min():.3f} "
        f"max={cos.max():.3f} (layer {cos.argmin()} most separable)")
    log(f"saved directions {tuple(directions.shape)} -> {path}")

    if want_routing:
        counts_g = read_routing_counts(counts_g_path)
        counts_b = read_routing_counts(counts_b_path)
        counts_g_path.unlink(missing_ok=True)
        counts_b_path.unlink(missing_ok=True)
        ratio = routing_ratio(counts_g, counts_b)
        stats_path = settings.routing_stats_path()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(stats_path, ratio)
        log(f"routing histograms: {int(counts_g.sum())} good / "
            f"{int(counts_b.sum())} bad routed tokens; "
            f"ratio min={ratio.min():.3f} max={ratio.max():.3f}")
        log(f"saved routing stats {ratio.shape} -> {stats_path}")


def cmd_directions(args) -> None:
    settings = _load_settings(args)
    model = _open_model(settings)
    spec = _arch_spec(settings, model)
    _compute_directions(settings, model, spec)
    # Refresh the cache sidecar so `run` can reuse these directions.
    _write_directions_meta(
        settings, _directions_fingerprint(settings, model, spec))


def _open_harness_log(settings: Settings, reporter):
    """In live mode llama.cpp's verbose stderr would wreck the dashboard;
    tee it to a log file instead (line protocol is unaffected)."""
    if not reporter.live:
        return None
    logdir = Path(settings.study_checkpoint_dir)
    logdir.mkdir(parents=True, exist_ok=True)
    return open(logdir / f"{settings.study_name}.harness.log", "a")


def _study(settings: Settings, model: GGUFModel, spec: ArchSpec,
           ablater: Ablater, reporter, harness_log,
           stage_b_only: bool = False,
           stage_b_trials: list[int] | None = None):
    """Run (or resume) the Optuna study under the given reporter. Shared by
    `optimize` and `run`."""
    try:
        # The factory lets the Optimizer respawn the harness mid-study
        # after a process-level crash (segfault in libcuda after many
        # adapter hot-swaps) instead of dying with the study.
        current: dict[str, Harness] = {}

        def harness_factory() -> Harness:
            h = _open_harness(settings, spec, stderr=harness_log)
            current["h"] = h
            return h

        with harness_factory() as harness:
            if settings.row_normalization != "full":
                reporter.phase("precomputing per-layer projections "
                               "(one streaming pass)...")
                ablater.precompute(
                    progress=lambda m: reporter.phase(f"  {m}"))
            kw_scorer = _make_kw_scorer(
                settings, harness, settings.eval_harmful_prompts,
                settings.max_response_length,
            )
            kw_scorer_a = None
            if settings.two_stage:
                stage_a_prompts = (settings.stage_a_harmful_prompts
                                   or settings.eval_harmful_prompts)
                kw_scorer_a = _make_kw_scorer(
                    settings, harness, stage_a_prompts,
                    settings.stage_a_max_response_length,
                )
                if settings.stage_b_select == "threshold":
                    b_sel = (f"trials with stage A kw <= "
                             f"{settings.stage_b_max_keyword_rate} and KL <= "
                             f"{settings.stage_b_max_kl} (cap "
                             f"{settings.stage_b_top_k})")
                else:
                    b_sel = f"top {settings.stage_b_top_k} trials (pareto)"
                reporter.log(
                    f"two-stage eval: stage A = {len(kw_scorer_a.prompts)} "
                    f"prompts x {settings.stage_a_max_response_length} "
                    f"tokens, stage B = full set, {b_sel}")
            kl_scorer = KLDivergence(
                harness, settings.prompts(settings.eval_harmless_prompts),
                prefix=settings.response_prefix,
            )
            reporter.kw_denominators = {
                "full": len(kw_scorer.prompts),
                "a": len(kw_scorer_a.prompts) if kw_scorer_a else 0,
            }
            harness.set_adapter(None)
            fingerprint = _baseline_fingerprint(settings)
            cache, why = _baseline_cache_status(settings, fingerprint)
            if cache is not None:
                kl_scorer.set_baseline_logits(cache["kl_logits"])
                baseline_kw = Score(cache["keyword"]["value"],
                                    cache["keyword"]["display"])
                reporter.log(f"baseline: loaded cached baseline from "
                             f"{_baseline_path(settings)} (keyword "
                             f"{baseline_kw.display}) — skipping the "
                             f"baseline generation/logits passes")
                reporter.baseline(baseline_kw.display)
            else:
                reporter.log(f"baseline: measuring ({why})")
                reporter.phase("measuring baseline (no adapter)...")
                kl_logits = kl_scorer.init_baseline()
                baseline_kw = kw_scorer.get_score()
                reporter.baseline(baseline_kw.display)
                path = _write_baseline_cache(
                    settings, fingerprint, baseline_kw,
                    kw_scorer.last_responses or [], kl_logits)
                reporter.log(f"baseline: cached -> {path} "
                             f"(restarts will skip the baseline passes)")

            opt = Optimizer(
                settings, model, ablater, harness, kw_scorer, kl_scorer,
                baseline_scores={"keyword_rate": baseline_kw.value,
                                 "kl_divergence": 0.0},
                stage_a_keyword_scorer=kw_scorer_a,
                reporter=reporter,
                harness_factory=harness_factory,
            )
            study = opt.run(stage_b_only=stage_b_only,
                            stage_b_trials=stage_b_trials)
    finally:
        # Close whichever harness is current (a respawned one, if any);
        # the `with` above only closes the original.
        h = current.get("h")
        if h is not None and h.proc.poll() is None:
            h.close()
    return study


def cmd_optimize(args) -> None:
    from .status import make_reporter

    settings = _load_settings(args)
    stage_b_trials = None
    if args.stage_b_trials:
        try:
            stage_b_trials = [int(x) for x in
                              args.stage_b_trials.split(",") if x.strip()]
        except ValueError:
            sys.exit("error: --stage-b-trials takes comma-separated trial "
                     "numbers")
        if not stage_b_trials:
            sys.exit("error: --stage-b-trials got no trial numbers")
        if not settings.two_stage:
            sys.exit("error: --stage-b-trials needs two_stage = true in the "
                     "config")
    if args.stage_b_only and not (settings.two_stage and settings.stage_b_top_k > 0):
        sys.exit("error: --stage-b-only needs two_stage = true and "
                 "stage_b_top_k > 0 in the config")
    if settings.expert_mode == "routing" and not settings.routing_stats_path().exists():
        sys.exit(f"error: expert_mode='routing' needs routing stats at "
                 f"{settings.routing_stats_path()} — run `directions` first "
                 f"(it collects them when expert_mode='routing'), or set "
                 f"routing_stats_file")
    model = _open_model(settings)
    spec = _arch_spec(settings, model)
    directions = torch.load(_directions_path(settings), weights_only=True)
    ablater = _make_ablater(settings, model, directions)

    with make_reporter(settings, plain=args.plain) as reporter:
        config.log = reporter.log  # keep dataset logs off the live display
        harness_log = _open_harness_log(settings, reporter)
        try:
            study = _study(settings, model, spec, ablater, reporter,
                           harness_log, stage_b_only=args.stage_b_only,
                           stage_b_trials=stage_b_trials)
        finally:
            if harness_log is not None:
                harness_log.close()

    print("\nPareto-optimal trials:")
    for t in pareto_trials(study):
        print("  " + trial_summary(t, study))


def _sanity_check(settings: Settings) -> None:
    """Cheap pre-flight for `run`: verify the paths the pipeline needs before
    starting the dashboard (the offline `selftest` covers the math)."""
    problems = []
    if not settings.model:
        problems.append("model path is empty (set 'model' in the config "
                        "or pass --model)")
    elif not Path(settings.model).exists():
        problems.append(f"model not found: {settings.model}")
    hb = Path(settings.harness_binary)
    if not settings.harness_binary:
        problems.append("harness_binary is empty (build heretic-eval and set "
                        "harness_binary in the config, see README)")
    elif not hb.exists():
        problems.append(f"harness_binary not found: {hb} "
                        f"(build heretic-eval, see README)")
    elif not os.access(hb, os.X_OK):
        problems.append(f"harness_binary is not executable: {hb}")
    if settings.chat_template_file \
            and not Path(settings.chat_template_file).exists():
        problems.append(
            f"chat_template_file not found: {settings.chat_template_file}")
    for d, what in ((settings.shm_dir, "shm_dir"),
                    (settings.study_checkpoint_dir, "study_checkpoint_dir")):
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            problems.append(f"{what} is not usable: {d} ({e})")
    if problems:
        sys.exit("configuration problems:\n  - " + "\n  - ".join(problems))
    print(f"sanity check OK: model {settings.model}")
    print(f"  harness {settings.harness_binary}")
    print(f"  shm_dir {settings.shm_dir}, "
          f"checkpoints {settings.study_checkpoint_dir}")


def _print_next_steps(args, settings: Settings) -> None:
    cfg = args.config
    trials_log = (f"{settings.study_checkpoint_dir}/"
                  f"{settings.study_name}.trials.jsonl")
    print(f"""
Next steps — export a winning trial as a permanent LoRA adapter (lossless):
  heretic-gguf --config {cfg} export --mode adapter --pareto-index 0 \\
      --out my-model-heretic-lora.gguf
  (--pareto-index 0 is the top of the front above; use --trial N for a
   specific trial — see the list or {trials_log})

Then run it with llama.cpp:
  llama-server -m {settings.model} --lora my-model-heretic-lora.gguf

Optional:
  heretic-gguf --config {cfg} verify --pareto-index 0
      # fidelity check: re-measure the trial with row_normalization="full"
  heretic-gguf --config {cfg} export --mode merged --trial N --out <dir>
      # full merged GGUF instead of an adapter (needs model-sized free disk)""")


def cmd_run(args) -> None:
    """One-command pipeline: sanity check -> directions (reused when the
    cached copy matches the current inputs) -> Optuna study, all under the
    same live dashboard `optimize` uses."""
    from .status import make_reporter

    settings = _load_settings(args)
    _sanity_check(settings)
    model = _open_model(settings)
    spec = _arch_spec(settings, model)

    with make_reporter(settings, plain=args.plain) as reporter:
        config.log = reporter.log  # keep dataset logs off the live display
        harness_log = _open_harness_log(settings, reporter)
        try:
            fingerprint = _directions_fingerprint(settings, model, spec)
            fresh, why = _directions_cache_status(settings, fingerprint)
            if fresh and not args.force_directions:
                reporter.log(f"directions: {_directions_path(settings)} is up "
                             f"to date — skipping "
                             f"(use --force-directions to recompute)")
            else:
                reason = ("forced by --force-directions"
                          if fresh else why)
                reporter.log(f"directions: recomputing ({reason})")
                reporter.phase("collecting residuals / computing "
                               "directions...")
                _compute_directions(settings, model, spec,
                                    log=reporter.log, phase=reporter.phase,
                                    harness_stderr=harness_log)
                _write_directions_meta(settings, fingerprint)
            directions = torch.load(_directions_path(settings),
                                    weights_only=True)
            ablater = _make_ablater(settings, model, directions)
            study = _study(settings, model, spec, ablater, reporter,
                           harness_log, stage_b_only=False)
        finally:
            if harness_log is not None:
                harness_log.close()

    print("\nPareto-optimal trials:")
    for t in pareto_trials(study):
        print("  " + trial_summary(t, study))
    _print_next_steps(args, settings)


def cmd_export(args) -> None:
    settings = _load_settings(args)
    model = _open_model(settings)
    directions = torch.load(_directions_path(settings), weights_only=True)
    ablater = _make_ablater(settings, model, directions)

    study = _load_study(settings)
    trial = _pick_trial(study, args.trial, args.pareto_index)
    print("exporting " + trial_summary(trial, study))

    direction_index, parameters, expert_scale = _trial_ablation_args(
        settings, ablater, trial)

    if settings.row_normalization != "full":
        print("precomputing per-layer projections...")
        ablater.precompute(progress=lambda m: print(f"  {m}"))

    if args.mode == "adapter":
        dtype = np.float16 if settings.adapter_dtype == "f16" else np.float32
        rank = (max(settings.full_normalization_lora_rank, ablater.n_dirs)
                if settings.row_normalization == "full" else ablater.n_dirs)
        tensors = ablater.build_adapter(direction_index, parameters, expert_scale,
                                        adapter_dtype=dtype)
        GGUFModel.write_adapter(
            args.out, arch=model.arch, rank=rank, tensors=tensors,
            metadata=_adapter_metadata(settings, study, trial),
        )
        print(f"wrote adapter -> {args.out}")
        print(f"run with: llama-server -m <base.gguf> --lora {args.out} ...")
    else:
        deltas = ablater.build_adapter(direction_index, parameters, expert_scale,
                                       merged=True)
        written = model.export_merged(args.out, deltas, requant=args.requant)
        for p in written:
            print(f"wrote {p}")


def cmd_verify(args) -> None:
    """Exact-fidelity check of the `pre` row-normalization default: rebuild a
    finished trial with row_normalization="full" (the exact Heretic port;
    per-tensor dequant + SVD, slow) and compare keyword rate + KL against the
    scores recorded during the study."""
    settings = _load_settings(args)
    model = _open_model(settings)
    spec = _arch_spec(settings, model)
    directions = torch.load(_directions_path(settings), weights_only=True)

    study = _load_study(settings)
    trial = _pick_trial(study, args.trial, args.pareto_index)
    sb = stage_b_scores(study).get(trial.number)
    recorded = sb or trial.user_attrs.get("scores", {})
    recorded_src = ("stage B" if sb
                    else "stage A" if trial.user_attrs.get("eval_stage") == "a"
                    else "study")
    print("verifying " + trial_summary(trial, study))
    print(f"recorded scores are from the {recorded_src} evaluation "
          f"(row_normalization={settings.row_normalization!r}); the fresh "
          f"measurement below uses the full eval sets with 'full'.")

    settings_full = dataclasses.replace(settings, row_normalization="full")
    ablater = _make_ablater(settings_full, model, directions)
    direction_index, parameters, expert_scale = _trial_ablation_args(
        settings, ablater, trial)

    with _open_harness(settings, spec) as harness:
        kw_scorer = _make_kw_scorer(
            settings, harness, settings.eval_harmful_prompts,
            settings.max_response_length,
        )
        kl_scorer = KLDivergence(
            harness, settings.prompts(settings.eval_harmless_prompts),
            prefix=settings.response_prefix,
        )
        harness.set_adapter(None)
        print("measuring baseline (no adapter)...")
        kl_scorer.init_baseline()
        baseline_kw = kw_scorer.get_score()
        print(f"baseline keyword rate: {baseline_kw.display}")

        t0 = time.time()
        print(f"rebuilding trial #{trial.number} with row_normalization='full' "
              f"(rank {max(settings.full_normalization_lora_rank, ablater.n_dirs)}"
              f"; per-tensor dequant + SVD, slow)...")
        dtype = np.float16 if settings.adapter_dtype == "f16" else np.float32
        tensors = ablater.build_adapter(direction_index, parameters, expert_scale,
                                        adapter_dtype=dtype)
        adapter_path = str(Path(settings.shm_dir) / "verify_adapter.gguf")
        GGUFModel.write_adapter(adapter_path, arch=model.arch,
                                rank=max(settings.full_normalization_lora_rank,
                                         ablater.n_dirs),
                                tensors=tensors)
        print(f"  adapter built in {time.time() - t0:.0f}s; evaluating...")
        harness.set_adapter(adapter_path)
        kw = kw_scorer.get_score()
        kl = kl_scorer.get_score()
        harness.set_adapter(None)

    rec_kw = recorded.get("keyword_rate", float("nan"))
    rec_kl = recorded.get("kl_divergence", float("nan"))
    print(f"\nfidelity check, trial #{trial.number} "
          f"('{settings.row_normalization}' recorded vs 'full' re-measured):")
    print(f"  keyword rate: {rec_kw:.3f} -> {kw.value:.3f} "
          f"(delta {kw.value - rec_kw:+.3f}; baseline {baseline_kw.value:.3f})")
    print(f"  KL:           {rec_kl:.4f} -> {kl.value:.4f} "
          f"(delta {kl.value - rec_kl:+.4f})")
    if abs(kw.value - rec_kw) > 0.05 or abs(kl.value - rec_kl) > 0.05:
        print("  NOTE: material divergence — consider running the study with "
              "row_normalization = \"full\" or document the gap.")
    else:
        print("  OK: 'pre' stays a justified default for this configuration.")


def cmd_rescore(args) -> None:
    """Break Stage-A ties after a two-stage study: re-score every finished
    trial whose Stage-A keyword rate is <= --threshold (default 0.0, i.e. the
    whole 0/32 crowd) with the FULL keyword eval set, merging results into the
    study's stage_b_scores attr — the same store run_stage_b uses, so
    trial_summary/export pick the scores up automatically. Trials already
    re-scored (by the automatic Stage B or an earlier rescore run) are
    skipped, so the command is resumable. KL needs no re-measurement: it is
    prefill-only and every Stage-A trial already used the full harmless set,
    so the recorded value is carried over."""
    from optuna.trial import TrialState

    settings = _load_settings(args)
    study = _load_study(settings)
    done_b = stage_b_scores(study)
    candidates = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.number not in done_b
        and t.user_attrs.get("scores", {}).get("keyword_rate", 1.0)
        <= args.threshold
    ]
    # Best (kw, kl) first: an interrupted run re-scores the most promising
    # trials first.
    candidates.sort(key=lambda t: tuple(t.values))
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"{len(candidates)} trials with stage-A keyword_rate <= "
          f"{args.threshold} not yet re-scored "
          f"(~{len(candidates) * 5}-{len(candidates) * 7} min on DS4 at "
          f"~5 min/trial)")
    if args.dry_run or not candidates:
        for t in candidates:
            print("  " + trial_summary(t, study))
        return

    model = _open_model(settings)
    spec = _arch_spec(settings, model)
    directions = torch.load(_directions_path(settings), weights_only=True)
    ablater = _make_ablater(settings, model, directions)
    if settings.row_normalization != "full":
        print("precomputing per-layer projections...")
        ablater.precompute(progress=lambda m: print(f"  {m}"))

    with _open_harness(settings, spec) as harness:
        kw_scorer = _make_kw_scorer(
            settings, harness, settings.eval_harmful_prompts,
            settings.max_response_length,
        )
        kl_scorer = KLDivergence(harness, [], prefix=settings.response_prefix)
        opt = Optimizer(settings, model, ablater, harness, kw_scorer, kl_scorer)
        stage_b = dict(done_b)
        for t in candidates:
            t0 = time.time()
            opt.restore_trial(t)
            kw = kw_scorer.get_score()
            harness.set_adapter(None)
            kl = t.values[1]  # prefill-only; stage A already used the full set
            stage_b[str(t.number)] = {"keyword_rate": kw.value,
                                      "kl_divergence": kl}
            study.set_user_attr("stage_b_scores", stage_b)
            stage_a_kw = t.user_attrs.get("scores", {}).get("keyword_rate",
                                                            float("nan"))
            duration = time.time() - t0
            print(f"trial #{t.number}: stage A kw={stage_a_kw:.3f} -> "
                  f"full kw={kw.display}, kl={kl:.4f} ({duration:.0f}s)",
                  flush=True)
            opt.log_trial_jsonl({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "trial": t.number,
                "stage": "b",
                "params": t.params,
                "direction_index": t.user_attrs.get("direction_index"),
                "scores": {"keyword_rate": kw.value, "kl_divergence": kl},
                "duration_s": round(duration, 1),
                "rescore": True,
            })
    print(f"rescore complete: {len(candidates)} trials merged into "
          f"stage_b_scores")


def cmd_selftest(args) -> None:
    """Offline smoke tests: GGUF access, dequant, ablation assembly, adapter write."""
    settings = _load_settings(args)
    model = _open_model(settings)
    spec = _arch_spec(settings, model)
    print(f"model: {model.arch}, layers={model.n_layers}, "
          f"experts={model.n_experts}, d={model.n_embd}")
    print(f"  arch spec: capture={settings.capture_node or spec.capture_node}-<layer>, "
          f"components={list(spec.all_components)}")

    # 1. Dequant sanity of each component tensor on a representative layer:
    #    layer 5 on large models, the last existing one on small ones.
    def _probe_exists(layer: int) -> bool:
        return any((n := spec.tensor_name(c, layer)) is not None and n in model.tensors
                   for c in spec.all_components)

    probe_layer = 5 if _probe_exists(5) else next(
        l for l in range(model.n_layers - 1, -1, -1) if _probe_exists(l))
    probe_expert = min(3, max(0, model.n_experts - 1))
    for comp in spec.all_components:
        name = spec.tensor_name(comp, probe_layer)
        if name is None:
            print(f"  {comp}: no tensor on layer {probe_layer} (hybrid arch), skipped")
            continue
        ref = model.info(name)
        if ref.n_experts:
            W = model.dequant(name, expert=probe_expert)
        else:
            W = model.dequant(name)
        print(f"  {name}: qtype={ref.qtype.name} shape={W.shape} "
              f"absmean={np.abs(W).mean():.5f}")

    # 2. Ablation assembly with synthetic directions.
    torch.manual_seed(0)
    directions = torch.nn.functional.normalize(
        torch.randn(model.n_layers, model.n_embd), dim=1)
    ablater = _make_ablater(settings, model, directions)
    print("precomputing projections (this streams all down_exps once)...")
    ablater.precompute(progress=lambda m: print(f"  {m}"))

    from .ablate import AbliterationParameters

    params = {
        c: AbliterationParameters([1.2], 0.75 * (model.n_layers - 1), [0.3], 8.0)
        for c in settings.components
    }
    tensors = ablater.build_adapter(0.7 * (model.n_layers - 1), params)
    n_bytes = sum(t.nbytes for t in tensors.values())
    print(f"  adapter tensors: {len(tensors)} entries, {n_bytes / 1e6:.1f} MB")
    for k in list(tensors)[:4]:
        print(f"    {k}: {tensors[k].shape} {tensors[k].dtype}")

    out = Path(settings.shm_dir) / "selftest_adapter.gguf"
    GGUFModel.write_adapter(out, arch=model.arch, rank=1, tensors=tensors,
                            metadata={"study_name": "selftest", "trial_number": 7,
                                      "param.direction_index": 30.25,
                                      "llama_cpp_commit": None})
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # 2b. Multi-direction (SOM mode) assembly: a direction SET of k=2 must
    #     yield an exact rank-2 adapter (global scope; precompute supplies
    #     the row norms, the direction projections are recomputed on the fly).
    k = 2
    directions_k = torch.nn.functional.normalize(
        torch.randn(model.n_layers, k, model.n_embd), dim=1)
    ablater_k = _make_ablater(settings, model, directions_k)
    params_k = {
        c: AbliterationParameters([1.2, 0.9], 0.75 * (model.n_layers - 1),
                                  [0.3, 0.4], 8.0)
        for c in settings.components
    }
    print(f"  multi-direction check: {k} directions/layer (global scope, "
          f"streams the tensors once more)...")
    ablater_k.precompute(progress=lambda m: None)
    tensors_k = ablater_k.build_adapter(0.7 * (model.n_layers - 1), params_k)
    # rank k: lora_a (k, in) / (E, k, in); lora_b (out, k) / (E, out, k)
    bad = {
        tname: t.shape for tname, t in tensors_k.items()
        if (tname.endswith(".lora_a")
            and not ((t.ndim == 2 and t.shape[0] == k)
                     or (t.ndim == 3 and t.shape[1] == k)))
        or (tname.endswith(".lora_b")
            and not ((t.ndim == 2 and t.shape[1] == k)
                     or (t.ndim == 3 and t.shape[2] == k)))
    }
    assert tensors_k and not bad, f"rank-{k} assembly shapes wrong: {bad}"
    for k2 in list(tensors_k)[:4]:
        print(f"    {k2}: {tensors_k[k2].shape} {tensors_k[k2].dtype}")
    out_k = Path(settings.shm_dir) / "selftest_adapter_k.gguf"
    GGUFModel.write_adapter(out_k, arch=model.arch, rank=k, tensors=tensors_k,
                            metadata={"study_name": "selftest",
                                      "multidirectional_som": True,
                                      "param.direction_index": 30.25})
    print(f"  wrote {out_k} ({out_k.stat().st_size / 1e6:.1f} MB)")

    # 3. Round-trip: read the adapter back (tensors + heretic metadata keys).
    from gguf import GGUFReader

    r = GGUFReader(str(out))
    names = [t.name for t in r.tensors]
    mlp_names = {f"{spec.tensor_name(COMPONENT_MLP_DOWN, l)}.lora_a"
                 for l in range(model.n_layers)}
    attn_names = {f"{n}.lora_b" for l in range(model.n_layers)
                  if (n := spec.tensor_name(COMPONENT_ATTN_OUT, l)) is not None}
    assert any(n in mlp_names for n in names), names[:5]
    assert any(n in attn_names for n in names), names[:5]
    kv = set(r.fields)
    assert "adapter.heretic.study_name" in kv
    assert "adapter.heretic.trial_number" in kv
    assert "adapter.heretic.param.direction_index" in kv
    assert "adapter.heretic.llama_cpp_commit" not in kv  # None values skipped
    print(f"  adapter round-trip OK ({len(names)} tensors, "
          f"{sum(1 for k in kv if k.startswith('adapter.heretic.'))} metadata keys)")
    print("SELFTEST PASSED")


def main() -> None:
    p = argparse.ArgumentParser(prog="heretic-gguf")
    p.add_argument("--config", default="config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    def refresh_flag(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--refresh-datasets", action="store_true",
                        help="re-download hf: prompt datasets even when a "
                             "cached copy exists on disk")

    sp = sub.add_parser("directions", help="collect residuals, compute directions")
    sp.add_argument("--model")
    refresh_flag(sp)
    sp.set_defaults(fn=cmd_directions)

    sp = sub.add_parser("run", help="all-in-one: sanity check, directions "
                        "(reused if the cached copy matches the current "
                        "inputs), then the Optuna study")
    sp.add_argument("--model")
    sp.add_argument("--plain", action="store_true",
                    help="line-based scrolling output instead of the live "
                         "status screen (also automatic when not a tty)")
    sp.add_argument("--force-directions", action="store_true",
                    help="recompute directions even if the cached "
                         "directions.pt matches the current inputs")
    refresh_flag(sp)
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("optimize", help="run the Optuna study")
    sp.add_argument("--model")
    sp.add_argument("--plain", action="store_true",
                    help="line-based scrolling output instead of the live "
                         "status screen (also automatic when not a tty)")
    sp.add_argument("--stage-b-only", action="store_true",
                    help="skip the search and run only the Stage-B re-score "
                         "of the top stage_b_top_k trials (e.g. to finish a "
                         "two-stage study that was stopped early)")
    sp.add_argument("--stage-b-trials", default="", metavar="N[,N...]",
                    help="comma-separated trial numbers to Stage-B re-score "
                         "(implies --stage-b-only; ignores stage_b_select "
                         "and stage_b_top_k)")
    refresh_flag(sp)
    sp.set_defaults(fn=cmd_optimize)

    sp = sub.add_parser("export", help="export a chosen trial")
    sp.add_argument("--model")
    sp.add_argument("--mode", choices=["adapter", "merged"], default="adapter")
    sp.add_argument("--trial", type=int, default=None, help="trial number")
    sp.add_argument("--pareto-index", type=int, default=0,
                    help="index into sorted Pareto front (default 0)")
    sp.add_argument("--requant", default="orig",
                    choices=["orig"],
                    help="merged-mode requantization target (only 'orig' is "
                         "supported; use --mode adapter for other dtypes)")
    sp.add_argument("--out", required=True,
                    help="adapter mode: output .gguf FILE; merged mode: "
                         "output DIRECTORY (shards are written inside it "
                         "keeping their split names)")
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("verify", help="rebuild a trial with row_normalization "
                        "'full' and compare scores (fidelity check of 'pre')")
    sp.add_argument("--model")
    sp.add_argument("--trial", type=int, default=None, help="trial number")
    sp.add_argument("--pareto-index", type=int, default=0,
                    help="index into sorted Pareto front (default 0)")
    refresh_flag(sp)
    sp.set_defaults(fn=cmd_verify)

    sp = sub.add_parser("selftest", help="offline smoke test (no harness)")
    sp.add_argument("--model")
    sp.set_defaults(fn=cmd_selftest)

    sp = sub.add_parser("rescore", help="re-score Stage-A-tied finished trials "
                        "with the full keyword eval set (tie-breaker after a "
                        "two-stage study; resumable)")
    sp.add_argument("--model")
    sp.add_argument("--threshold", type=float, default=0.0,
                    help="re-score trials with stage-A keyword_rate <= this "
                         "(default 0.0 = the 0/32 crowd; 1/32 = 0.03125)")
    sp.add_argument("--limit", type=int, default=0,
                    help="re-score at most this many trials (best (kw, kl) "
                         "first; 0 = no limit)")
    sp.add_argument("--dry-run", action="store_true",
                    help="only list the selected trials, don't score")
    refresh_flag(sp)
    sp.set_defaults(fn=cmd_rescore)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
