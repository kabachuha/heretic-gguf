# SPDX-License-Identifier: AGPL-3.0-or-later
# Optuna study — port of Heretic's objective (src/heretic/main.py:591-709)
# with the per-expert parameter extensions.

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import Trial

from .ablate import Ablater, AbliterationParameters
from .archs import COMPONENT_MLP_DOWN
from .config import Settings
from .gguf_io import GGUFModel
from .harness import Harness, HarnessError
from .scorers import KLDivergence, KeywordRate
from .status import Reporter
from .version import heretic_gguf_commit, llama_cpp_commit

# After this many CONSECUTIVE trials dying on harness errors (each already
# retried once on a fresh harness process), something is systematically wrong
# — stop the study loudly instead of burning all trials as FAILs.
MAX_CONSECUTIVE_HARNESS_FAILURES = 5

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_trial_parameters(trial: Trial, ablater: Ablater,
                             settings: Settings) -> tuple[
                                 float | None,
                                 dict[str, AbliterationParameters],
                                 np.ndarray | None]:
    """Mirror Heretic's parameter space (main.py:596-669) + expert multipliers."""
    last_layer = ablater.n_layers - 1

    direction_scope = trial.suggest_categorical("direction_scope", ["global", "per layer"])
    direction_index = trial.suggest_float(
        "direction_index", 0.4 * last_layer, 0.9 * last_layer
    )
    if direction_scope == "per layer":
        direction_index = None

    parameters: dict[str, AbliterationParameters] = {}
    for component in ablater.components:
        # MLP components get a negative lower bound clamped to 0 so the
        # optimizer can disable their ablation entirely (Heretic, issue #202).
        lo, hi = (settings.mlp_max_weight_range
                  if component.startswith("mlp.")
                  else settings.attn_max_weight_range)
        max_weight = max(
            0.0,
            trial.suggest_float(f"{component}.max_weight", lo, hi),
        )
        max_weight_position = trial.suggest_float(
            f"{component}.max_weight_position", 0.6 * last_layer, 1.0 * last_layer
        )
        min_weight = trial.suggest_float(f"{component}.min_weight", 0.0, 1.0)
        min_weight_distance = trial.suggest_float(
            f"{component}.min_weight_distance", 1.0, max(0.6 * last_layer, 1.0)
        )
        parameters[component] = AbliterationParameters(
            max_weight=max_weight,
            max_weight_position=max_weight_position,
            min_weight=min_weight * max_weight,
            min_weight_distance=min_weight_distance,
        )

    # --- per-expert extension (constraint 2) -------------------------------
    expert_scale = None
    if settings.expert_mode == "grouped" and COMPONENT_MLP_DOWN in ablater.components:
        g = settings.expert_groups
        mults = np.array(
            [trial.suggest_float(f"expert_group.{i}", 0.0, 1.25) for i in range(g)],
            dtype=np.float32,
        )
        expert_scale = ablater.expert_scale_from_groups(mults, g)
    elif settings.expert_mode == "routing" and COMPONENT_MLP_DOWN in ablater.components:
        stats = np.load(settings.routing_stats_path())
        expert_scale = ablater.expert_scale_from_routing(stats)

    return direction_index, parameters, expert_scale


def study_meta(settings: Settings,
               baseline_scores: dict | None = None) -> dict[str, object]:
    """Provenance/config snapshot stored in the study's user attributes."""
    meta: dict[str, object] = {
        "llama_cpp_commit": llama_cpp_commit(settings.harness_binary),
        "heretic_gguf_commit": heretic_gguf_commit(),
        "config": {
            "row_normalization": settings.row_normalization,
            "components": list(settings.components),
            "expert_mode": settings.expert_mode,
            "max_response_length": settings.max_response_length,
            "eval_harmful_prompts": settings.eval_harmful_prompts,
            "eval_harmless_prompts": settings.eval_harmless_prompts,
            "seed": settings.seed,
        },
    }
    if baseline_scores:
        meta["baseline_scores"] = baseline_scores
    return meta


class Optimizer:
    def __init__(self, settings: Settings, model: GGUFModel, ablater: Ablater,
                 harness: Harness, keyword_scorer: KeywordRate,
                 kl_scorer: KLDivergence,
                 baseline_scores: dict | None = None,
                 stage_a_keyword_scorer: KeywordRate | None = None,
                 reporter: Reporter | None = None,
                 harness_factory=None):
        self.settings = settings
        self.model = model
        self.ablater = ablater
        self.harness = harness
        self.keyword_scorer = keyword_scorer
        self.kl_scorer = kl_scorer
        self.baseline_scores = baseline_scores
        self.stage_a_keyword_scorer = stage_a_keyword_scorer
        self.reporter = reporter or Reporter()
        # Callable returning a fresh, READY Harness (same model/args). When
        # set, a trial whose harness PROCESS died (segfault etc.) is retried
        # once on a respawned harness instead of killing the whole study.
        self.harness_factory = harness_factory
        self._consecutive_harness_failures = 0
        self.adapter_path = str(Path(settings.shm_dir) / "trial_adapter.gguf")
        self.trial_index = 0
        self.jsonl_path = (Path(settings.study_checkpoint_dir)
                           / f"{settings.study_name}.trials.jsonl")

    # --------------------------------------------------------------- logging
    def log_trial_jsonl(self, record: dict) -> None:
        """Append one greppable JSON line per trial to
        checkpoints/<study>.trials.jsonl (alongside the Optuna journal)."""
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # ------------------------------------------------------------------ eval
    def apply_trial(self, direction_index, parameters, expert_scale) -> None:
        dtype = np.float16 if self.settings.adapter_dtype == "f16" else np.float32
        rank = (self.settings.full_normalization_lora_rank
                if self.settings.row_normalization == "full" else 1)
        tensors = self.ablater.build_adapter(
            direction_index, parameters, expert_scale, adapter_dtype=dtype
        )
        if not tensors:
            self.harness.set_adapter(None)
            return
        GGUFModel.write_adapter(
            self.adapter_path, arch=self.model.arch, rank=rank, tensors=tensors
        )
        self.harness.set_adapter(self.adapter_path)

    def evaluate(self, stage_a: bool = False) -> tuple[float, float, dict]:
        kw_scorer = (self.stage_a_keyword_scorer
                     if stage_a and self.stage_a_keyword_scorer is not None
                     else self.keyword_scorer)
        kw = kw_scorer.get_score()
        kl = self.kl_scorer.get_score()
        return kw.value, kl.value, {"keyword": kw.display, "kl": kl.display}

    # --------------------------------------------- harness crash resilience
    def _harness_dead(self) -> bool:
        return self.harness.proc.poll() is not None

    def _restart_harness(self) -> None:
        """Respawn the harness process and re-point all scorers at it. The
        KL baseline needs no re-measurement: it lives in the scorer and the
        respawned harness loads the same model deterministically. The new
        process starts with NO adapter, which matches the per-trial
        apply-then-clear flow."""
        assert self.harness_factory is not None
        try:
            self.harness.close()
        except Exception:
            pass
        self.reporter.log("respawning harness (model reload, ~1 min)...")
        self.harness = self.harness_factory()
        for scorer in (self.keyword_scorer, self.stage_a_keyword_scorer,
                       self.kl_scorer):
            if scorer is not None:
                scorer.harness = self.harness

    def _with_harness_retry(self, fn, what: str):
        """Run fn() (which talks to the harness). If the harness PROCESS died
        (EOF on the pipe — e.g. the libcuda segfault seen after ~500 adapter
        hot-swaps), respawn it and retry once. A HarnessError from a LIVE
        harness (an ERR protocol response) is re-raised immediately, as is a
        second death on the same fn."""
        try:
            return fn()
        except HarnessError:
            if self.harness_factory is None or not self._harness_dead():
                raise
            self.reporter.log(f"harness process died during {what}; "
                              f"retrying once on a fresh process")
            self._restart_harness()
            return fn()

    def _clear_adapter(self) -> None:
        try:
            self.harness.set_adapter(None)
        except Exception:
            pass  # cleanup only; the harness may already be dead

    # ----------------------------------------------------------------- study
    def objective(self, trial: Trial) -> tuple[float, float]:
        self.trial_index += 1
        two_stage = self.settings.two_stage and self.stage_a_keyword_scorer
        trial.set_user_attr("index", self.trial_index)
        direction_index, parameters, expert_scale = suggest_trial_parameters(
            trial, self.ablater, self.settings
        )
        trial.set_user_attr("direction_index", direction_index)
        trial.set_user_attr("parameters", {k: asdict(v) for k, v in parameters.items()})
        trial.set_user_attr("eval_stage", "a" if two_stage else "full")

        t0 = time.time()
        stage = "a" if two_stage else "full"
        self.reporter.trial_started(self.trial_index, trial.number,
                                    self.settings.n_trials)
        self.reporter.phase("building adapter")

        def run_trial():
            self.apply_trial(direction_index, parameters, expert_scale)
            self.reporter.phase(f"evaluating (stage {stage}: generation + KL)")
            return self.evaluate(stage_a=bool(two_stage))

        try:
            kw, kl, displays = self._with_harness_retry(
                run_trial, f"trial {self.trial_index}")
        except HarnessError:
            # Retried already; count consecutive harness failures and stop
            # the study loudly when they pile up (systematic problem),
            # otherwise let optuna mark just this trial FAIL (catch= below).
            self._consecutive_harness_failures += 1
            if self._consecutive_harness_failures >= \
                    MAX_CONSECUTIVE_HARNESS_FAILURES:
                raise RuntimeError(
                    f"{self._consecutive_harness_failures} consecutive trials "
                    f"failed on harness errors (each retried on a fresh "
                    f"harness process) — aborting the study; check "
                    f"checkpoints/{self.settings.study_name}.harness.log")
            raise
        self._consecutive_harness_failures = 0
        duration = time.time() - t0
        trial.set_user_attr("scores", {"keyword_rate": kw, "kl_divergence": kl})
        self.reporter.trial_finished(self.trial_index, trial.number,
                                     self.settings.n_trials,
                                     displays["keyword"], kw, kl, stage,
                                     duration)
        self._clear_adapter()
        self.log_trial_jsonl({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trial": trial.number,
            "index": self.trial_index,
            "stage": "a" if two_stage else "full",
            "params": trial.params,
            "direction_index": direction_index,
            "scores": {"keyword_rate": kw, "kl_divergence": kl},
            "duration_s": round(duration, 1),
        })
        return kw, kl

    def run(self, stage_b_only: bool = False,
            stage_b_trials: list[int] | None = None) -> optuna.Study:
        ckpt = Path(self.settings.study_checkpoint_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        journal = ckpt / f"{self.settings.study_name}.journal"
        storage = JournalStorage(
            JournalFileBackend(str(journal), lock_obj=JournalFileOpenLock(str(journal)))
        )
        study = optuna.create_study(
            study_name=self.settings.study_name,
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(
                n_startup_trials=self.settings.n_startup_trials,
                multivariate=True,
                seed=self.settings.seed,
            ),
            directions=["minimize", "minimize"],
        )
        for key, value in study_meta(self.settings, self.baseline_scores).items():
            study.set_user_attr(key, value)
        self.trial_index = len(study.trials)
        if not study.trials and self.settings.seed_top_k > 0:
            self._seed_study(study)
        # Waiting (enqueued seed) trials run first inside study.optimize, so
        # they must not reduce the budget twice: count only trials that have
        # already consumed one.
        n_waiting = sum(1 for t in study.trials
                        if t.state == optuna.trial.TrialState.WAITING)
        remaining = max(0, self.settings.n_trials
                        - (len(study.trials) - n_waiting))
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        self.reporter.study_started(len(study.trials), self.settings.n_trials,
                                    completed, stage_b_scores(study))
        if remaining and not stage_b_only and stage_b_trials is None:
            # HarnessError (after the in-objective respawn+retry) marks just
            # that trial FAIL and the study continues; anything else aborts.
            study.optimize(self.objective, n_trials=remaining,
                           catch=(HarnessError,))
        if self.settings.two_stage and (self.settings.stage_b_top_k > 0
                                        or stage_b_trials):
            self.run_stage_b(study, trial_numbers=stage_b_trials)
        return study

    def _seed_study(self, study: optuna.Study) -> None:
        """Enqueue the top trials of a previous study into this (empty) one.

        The seeds RE-RUN as ordinary trials (they count toward n_trials) so
        TPE's history is built from evaluations in THIS study's parameter
        space — important when ranges were widened between the studies,
        since enqueued trials record the new distributions. The source study
        is opened read-only; its journal is never touched."""
        settings = self.settings
        journal = (Path(settings.seed_from_checkpoint_dir)
                   / f"{settings.seed_from_study}.journal")
        if not journal.exists():
            raise FileNotFoundError(
                f"seed_top_k > 0 but no journal at {journal}")
        source_storage = JournalStorage(JournalFileBackend(str(journal)))
        source = optuna.load_study(study_name=settings.seed_from_study,
                                   storage=source_storage)
        done = [t for t in source.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
        # Stage B scores (full eval set) are strictly better evidence than the
        # cheap Stage A values the objectives hold: when the source study has
        # them, seed from the Stage B ranking first (a Stage A front ranks
        # do-nothing low-KL trials alongside aggressive ones whose Stage A kw
        # is optimistic; followup2: t201 A kw 0.071 -> B kw 0.217 while
        # t247 A 0.095 -> B 0.142). Trials without a Stage B score fill the
        # remaining slots from the Stage A non-dominated fronts as before.
        stage_b = source.user_attrs.get("stage_b_scores", {})
        selected: list = []
        if stage_b:
            scored = [t for t in done if str(t.number) in stage_b]
            scored.sort(key=lambda t: (stage_b[str(t.number)]["keyword_rate"],
                                       stage_b[str(t.number)]["kl_divergence"]))
            selected.extend(scored[:settings.seed_top_k])
        if len(selected) < settings.seed_top_k:
            chosen = {t.number for t in selected}
            rest = [t for t in done if t.number not in chosen]
            for front in non_dominated_fronts(rest):
                selected.extend(sorted(front, key=lambda t: tuple(t.values)))
                if len(selected) >= settings.seed_top_k:
                    break
        selected = selected[:settings.seed_top_k]
        for t in selected:
            study.enqueue_trial(t.params)
        self.reporter.log(
            f"seeded {len(selected)} trials from {journal} (top "
            f"{len(selected)} by stage B scores where available, else "
            f"non-dominated fronts): "
            + ", ".join(
                f"#{t.number} (kw={stage_b.get(str(t.number), {}).get('keyword_rate', t.values[0]):.3f}"
                f"{'B' if str(t.number) in stage_b else 'A'}, "
                f"kl={t.values[1]:.4f})" for t in selected))

    # ------------------------------------------------------------- stage B
    def run_stage_b(self, study: optuna.Study,
                    trial_numbers: list[int] | None = None) -> None:
        """Full-set keyword re-evaluation of selected trials of a two-stage
        study. Stage A scores (cheap) drove the search; this manual second
        pass re-scores the selected trials with the full eval set and
        stores the result in the `stage_b_scores` STUDY user attr (journal
        storage forbids updating finished trials, so a per-trial attr is not
        possible; multi-objective pruners are not supported by Optuna either,
        hence a second pass rather than trial.report-based pruning).

        Selection: `trial_numbers` (explicit list, from --stage-b-trials)
        wins; otherwise `stage_b_select` picks "pareto" (non-dominated
        fronts, filled in order) or "threshold" (every trial whose Stage A
        scores pass stage_b_max_keyword_rate/stage_b_max_kl), both capped at
        stage_b_top_k."""
        k = self.settings.stage_b_top_k
        stage_b = dict(study.user_attrs.get("stage_b_scores", {}))
        complete = [t for t in study.trials
                    if t.state == optuna.trial.TrialState.COMPLETE]
        done = [t for t in complete if str(t.number) not in stage_b]
        if trial_numbers is not None:
            wanted = set(trial_numbers)
            have = {t.number for t in complete}
            missing = sorted(wanted - have)
            if missing:
                self.reporter.log(f"stage B: requested trials not completed "
                                  f"(skipped): {missing}")
            selected = sorted((t for t in done if t.number in wanted),
                              key=lambda t: t.number)
            already = len(wanted & have) - len(selected)
            if already:
                self.reporter.log(f"stage B: {already} requested trial(s) "
                                  f"already have a Stage B score (skipped)")
        elif self.settings.stage_b_select == "threshold":
            max_kw = self.settings.stage_b_max_keyword_rate
            max_kl = self.settings.stage_b_max_kl
            self.reporter.log(f"stage B selection: threshold mode — Stage A "
                              f"kw <= {max_kw} and KL <= {max_kl}, "
                              f"capped at top {k}")
            selected = sorted(
                (t for t in done
                 if t.values[0] <= max_kw and t.values[1] <= max_kl),
                key=lambda t: tuple(t.values))[:k]
        else:
            if self.settings.stage_b_select != "pareto":
                raise ValueError(
                    f"unknown stage_b_select "
                    f"{self.settings.stage_b_select!r} "
                    f"(expected 'pareto' or 'threshold')")
            selected = []
            for front in non_dominated_fronts(done):
                selected.extend(sorted(front, key=lambda t: tuple(t.values)))
                if len(selected) >= k:
                    break
            selected = selected[:k]
        if not selected:
            self.reporter.log("stage B: no trials to re-score")
            return
        self.reporter.stage_b_started(len(selected),
                                      len(self.keyword_scorer.prompts),
                                      self.keyword_scorer.max_tokens)
        for t in selected:
            t0 = time.time()

            def run_b(trial=t):
                self.restore_trial(trial)
                return self.keyword_scorer.get_score()

            try:
                kw = self._with_harness_retry(run_b, f"stage B trial {t.number}")
            except HarnessError as e:
                # Already retried on a fresh harness; skip this trial rather
                # than killing stage B (scores simply stay absent).
                self.reporter.log(f"stage B trial {t.number} skipped: {e}")
                continue
            self._clear_adapter()
            kl = t.values[1]  # KL is prefill-only; Stage A already used the full set
            scores_b = {"keyword_rate": kw.value, "kl_divergence": kl}
            stage_b[str(t.number)] = scores_b
            study.set_user_attr("stage_b_scores", stage_b)
            stage_a_kw = t.user_attrs.get("scores", {}).get("keyword_rate", float("nan"))
            duration = time.time() - t0
            self.reporter.stage_b_trial(t.number, stage_a_kw, kw.value,
                                        kw.display, kl, duration)
            self.log_trial_jsonl({
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "trial": t.number,
                "stage": "b",
                "params": t.params,
                "direction_index": t.user_attrs.get("direction_index"),
                "scores": scores_b,
                "duration_s": round(duration, 1),
            })

    def restore_trial(self, trial) -> None:
        """Re-apply a finished trial's ablation (for export or inspection)."""
        parameters = {
            k: AbliterationParameters(**v)
            for k, v in trial.user_attrs["parameters"].items()
        }
        expert_scale = None
        if self.settings.expert_mode == "grouped":
            g = self.settings.expert_groups
            mults = np.array(
                [trial.params.get(f"expert_group.{i}", 1.0) for i in range(g)],
                dtype=np.float32,
            )
            expert_scale = self.ablater.expert_scale_from_groups(mults, g)
        elif self.settings.expert_mode == "routing":
            expert_scale = self.ablater.expert_scale_from_routing(
                np.load(self.settings.routing_stats_path()))
        self.apply_trial(trial.user_attrs["direction_index"], parameters, expert_scale)


def _dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Pareto dominance for minimize objectives."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def non_dominated_fronts(trials) -> list[list]:
    """Non-dominated sort (NSGA-II style) of trials by their objective values.
    Returns fronts as lists of trials, best front first."""
    remaining = list(trials)
    fronts: list[list] = []
    while remaining:
        front = [
            t for t in remaining
            if not any(u is not t and _dominates(tuple(u.values), tuple(t.values))
                       for u in remaining)
        ]
        fronts.append(front)
        remaining = [t for t in remaining if t not in front]
    return fronts


def pareto_trials(study: optuna.Study):
    """Pareto-optimal trials, sorted like Heretic's menu (by all objectives)."""
    return sorted(study.best_trials, key=lambda t: tuple(t.values))


def stage_b_scores(study: optuna.Study) -> dict[int, dict]:
    """trial_number -> full-eval scores from a two-stage study (empty if none)."""
    return {int(n): s for n, s in study.user_attrs.get("stage_b_scores", {}).items()}


def trial_summary(trial, study: optuna.Study | None = None) -> str:
    s = trial.user_attrs.get("scores", {})
    line = (f"trial #{trial.number} (index {trial.user_attrs.get('index')}): "
            f"keyword={s.get('keyword_rate', float('nan')):.3f} "
            f"kl={s.get('kl_divergence', float('nan')):.4f}")
    if trial.user_attrs.get("eval_stage") == "a":
        line += " [stage A]"
    sb = stage_b_scores(study).get(trial.number) if study is not None else None
    if sb:
        line += (f"  | stage B: keyword={sb['keyword_rate']:.3f} "
                 f"kl={sb['kl_divergence']:.4f}")
    return line
