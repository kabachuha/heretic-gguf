# SPDX-License-Identifier: AGPL-3.0-or-later
# Configuration — TOML-driven, mirroring the relevant subset of Heretic's
# config.default.toml plus GGUF/harness-specific settings.

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# One-line cache/fetch logs. The CLI replaces this with the reporter's log
# when the live dashboard is up so prints don't corrupt the display.
log = print


def _dataset_cache_file(cache_dir: Path, spec: str) -> Path:
    """Cache file for a full hf: spec (repo, split, slice, column are all in
    the spec string): a readable sanitized form plus a short hash suffix so
    distinct specs can never collide."""
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec).strip("_")
    digest = hashlib.sha1(spec.encode()).hexdigest()[:8]
    return cache_dir / f"{name}-{digest}.jsonl"


@dataclass
class DatasetSpec:
    """'hf:<repo>:<split>:<column>' or 'file:<path>[:<slice>]'."""
    source: str

    def _load_hf(self) -> list[str]:
        # hf:<repo>:<split[slice]>:<column> — the slice itself contains
        # colons, so split with a regex instead of naive ":" splitting.
        m = re.fullmatch(r"hf:(.+):([A-Za-z0-9_]+)(\[[^\]]*\])?:([^:]+)",
                         self.source)
        if not m:
            raise ValueError(f"bad hf dataset spec: {self.source}")
        repo, name, sl_str, column = m.groups()
        from datasets import load_dataset

        ds = load_dataset(repo)
        sl = slice(None)
        if sl_str:
            parts = [int(p) if p else None for p in sl_str[1:-1].split(":")]
            sl = slice(*parts)
        rows = ds[name][column]
        return [str(x) for x in rows[sl]]

    def load(self, cache_dir: Path | None = None,
             refresh: bool = False) -> list[str]:
        if self.source.startswith("hf:"):
            cache_file = (_dataset_cache_file(cache_dir, self.source)
                          if cache_dir is not None else None)
            if cache_file is not None and cache_file.exists() and not refresh:
                prompts = [json.loads(ln)
                           for ln in cache_file.read_text().splitlines()]
                log(f"dataset cache: {self.source} loaded from {cache_file} "
                    f"({len(prompts)} prompts)")
                return prompts
            prompts = self._load_hf()
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_text("".join(json.dumps(p) + "\n" for p in prompts))
                tmp.rename(cache_file)
                log(f"dataset cache: {self.source} fetched from the HF hub "
                    f"({len(prompts)} prompts) -> {cache_file}")
            return prompts
        if self.source.startswith("file:"):
            # file:<path>[:<slice>] — one prompt per line, UTF-8, blank lines
            # and '#' comment lines skipped. The optional trailing slice takes
            # the same forms as hf: specs and applies to THIS spec only:
            # ":50" / "[:50]" / ":[:50]" (first 50), ":10:20",
            # ":-20:" / "[-20:]" / ":[-20:]" (last 20). Colons inside
            # the path stay intact: the slice only matches a trailing
            # all-digits/colons (optionally bracketed) segment.
            m = re.fullmatch(r"file:(.+?)(?::?\[([-0-9:]*)\])?", self.source)
            if m is not None and m.group(2) is None:
                m = re.fullmatch(r"file:(.+?)(?::([-0-9:]*))?", self.source)
            if not m:
                raise ValueError(f"bad file dataset spec: {self.source}")
            path, sl_str = m.group(1), m.group(2)
            sl = slice(None)
            if sl_str:
                parts = [int(p) if p else None for p in sl_str.split(":")]
                sl = slice(*parts)
            lines = [
                ln.strip()
                for ln in Path(path).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#")
            ]
            return lines[sl]
        raise ValueError(f"unknown dataset source: {self.source}")


def load_prompts(specs: str | list[str], cache_dir: Path | None = None,
                 refresh: bool = False) -> tuple[list[str], list[tuple[str, int]]]:
    """Load one dataset spec or concatenate a LIST of them (hf:/file: specs
    mixed freely). Each spec's own [:N] slice is applied per spec, then the
    combined prompts are deduplicated preserving first-occurrence order.
    Returns (prompts, groups) where groups is a per-origin breakdown
    [("base"|"user", count)] in first-appearance order — "base" for hf:
    specs, "user" for file: specs — used for split refusal-rate displays."""
    spec_list = [specs] if isinstance(specs, str) else list(specs)
    seen: set[str] = set()
    prompts: list[str] = []
    counts: dict[str, int] = {}
    order: list[str] = []
    for spec in spec_list:
        label = "base" if spec.startswith("hf:") else "user"
        if label not in counts:
            order.append(label)
            counts[label] = 0
        loaded = DatasetSpec(spec).load(cache_dir=cache_dir, refresh=refresh)
        kept = 0
        for p in loaded:
            if p not in seen:
                seen.add(p)
                prompts.append(p)
                kept += 1
        counts[label] += kept
        if kept < len(loaded):
            log(f"dataset {spec}: {len(loaded) - kept} duplicate prompt(s) "
                f"skipped (dedup)")
    groups = [(label, counts[label]) for label in order]
    return prompts, groups


@dataclass
class Settings:
    # Model / harness
    model: str = ""                       # path to first GGUF shard
    harness_binary: str = ""              # path to the built heretic-eval binary
    harness_args: list[str] = field(default_factory=list)  # extra llama.cpp args
    chat_template_file: str = ""
    shm_dir: str = "/dev/shm/heretic_gguf"
    capture_node: str = ""                # override for the ArchSpec residual
                                          # capture node (HERETIC_CAPTURE_NODE)

    # Ablation
    row_normalization: str = "pre"        # "none" | "pre" | "full"
    full_normalization_lora_rank: int = 3
    adapter_dtype: str = "f16"            # "f32" | "f16"
    ablate_device: str = "cuda:0"         # dequant/projection device ("cpu" = slow, no VRAM)
    expert_chunk: int = 8                 # experts per dequant chunk (VRAM transient)
    components: list[str] = field(default_factory=lambda: [
        "attn.o_proj", "mlp.down_proj"])  # + optional "mlp.down_proj.shared"
    orthogonalize_direction: bool = True
    winsorization_quantile: float = 1.0
    seed: int = 42

    # Expert ablation (constraint 2)
    expert_mode: str = "shared"           # "shared" | "grouped" | "routing"
    expert_groups: int = 8
    routing_stats_file: str = ""          # .npy (n_layers, n_experts) ratios;
                                          # empty -> <study_checkpoint_dir>/routing_stats.npy

    # Optimization
    n_trials: int = 200
    n_startup_trials: int = 60
    study_checkpoint_dir: str = "checkpoints"
    study_name: str = "heretic-gguf"
    # Per-component max_weight search ranges [lo, hi] (Heretic hardcodes
    # 0.8..1.5 for attn, -0.25..1.5 for MLP). MLP lo < 0 is clamped to 0 so
    # TPE can switch the component off (Heretic issue #202 behavior).
    attn_max_weight_range: list[float] = field(default_factory=lambda: [0.8, 1.5])
    mlp_max_weight_range: list[float] = field(default_factory=lambda: [-0.25, 1.5])
    # Seed a FRESH study with the top trials of a previous one: the top
    # seed_top_k completed trials (non-dominated fronts, filled in order) of
    # seed_from_study in seed_from_checkpoint_dir are enqueued before the
    # first trial runs (they re-run and count toward n_trials). No effect
    # once the target study has any trials (resume-safe).
    seed_from_checkpoint_dir: str = ""
    seed_from_study: str = ""
    seed_top_k: int = 0

    # Evaluation
    system_prompt: str = "You are a helpful assistant."
    response_prefix: str = ""             # e.g. "<think></think>" for reasoning models
    max_response_length: int = 100
    keyword_markers: list[str] | None = None

    # Two-stage evaluation (generation dominates trial cost):
    # Stage A runs for EVERY trial (cheap: fewer prompts, shorter responses)
    # and is what Optuna ranks/prunes on; after the study, selected trials
    # are re-evaluated with the full keyword scorer (Stage B) and their
    # scores_stage_b stored in the journal.
    # KL is prefill-only (cheap) and always uses the full harmless set.
    two_stage: bool = False
    stage_a_harmful_prompts: str = ""     # empty -> same set as eval_harmful_prompts
    stage_a_max_response_length: int = 64
    stage_b_top_k: int = 10               # 0 -> skip Stage B entirely
    # Stage B trial selection: "pareto" = non-dominated fronts, filled in
    # order until stage_b_top_k; "threshold" = every trial whose STAGE A
    # scores pass both cutoffs below (still capped at stage_b_top_k, best
    # first). Stage A kw is optimistic (trials can rerank on the full set),
    # so a Pareto top-K can miss good export candidates — threshold mode
    # casts a wider, user-controlled net.
    stage_b_select: str = "pareto"        # "pareto" | "threshold"
    stage_b_max_keyword_rate: float = 1.0     # threshold mode: Stage A kw <= this
    stage_b_max_kl: float = float("inf")      # threshold mode: Stage A KL <= this

    # Datasets. Each key takes a single spec string or a LIST of specs
    # (hf:/file: mixed; concatenated and deduplicated, order-preserving —
    # see load_prompts). A spec's own [:N] slice applies to that spec only.
    good_prompts: str | list[str] = "hf:mlabonne/harmless_alpaca:train[:400]:text"
    bad_prompts: str | list[str] = "hf:mlabonne/harmful_behaviors:train[:400]:text"
    eval_harmful_prompts: str | list[str] = "hf:mlabonne/harmful_behaviors:test[:100]:text"
    eval_harmless_prompts: str | list[str] = "hf:mlabonne/harmless_alpaca:test[:100]:text"
    # "hf:" specs are downloaded once and cached as jsonl on disk; later runs
    # read the cache file and make zero network calls. Empty cache dir ->
    # <study_checkpoint_dir>/../datasets.
    dataset_cache_dir: str = ""
    refresh_datasets: bool = False      # re-download even if cached
                                          # (also: --refresh-datasets)

    @staticmethod
    def from_toml(path: str | Path) -> "Settings":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        s = Settings()
        for key, value in data.items():
            if hasattr(s, key):
                setattr(s, key, value)
            else:
                raise ValueError(f"unknown config key: {key}")
        enum_keys = {
            "row_normalization": ("none", "pre", "full"),
            "expert_mode": ("shared", "grouped", "routing"),
            "adapter_dtype": ("f16", "f32"),
            "stage_b_select": ("pareto", "threshold"),
        }
        for key, valid in enum_keys.items():
            if getattr(s, key) not in valid:
                raise ValueError(
                    f"bad value for config key {key}: {getattr(s, key)!r} "
                    f"(valid: {', '.join(valid)})")
        return s

    def dataset_cache_path(self) -> Path:
        """Where cached hf: prompt lists live (one jsonl per spec)."""
        if self.dataset_cache_dir:
            return Path(self.dataset_cache_dir)
        return Path(self.study_checkpoint_dir).parent / "datasets"

    def prompts(self, specs: str | list[str]) -> list[dict]:
        prompts, _ = load_prompts(specs, cache_dir=self.dataset_cache_path(),
                                  refresh=self.refresh_datasets)
        return [
            {"system": self.system_prompt, "user": u} for u in prompts
        ]

    def prompt_groups(self, specs: str | list[str]
                      ) -> list[tuple[str, int]] | None:
        """Per-origin prompt counts ([("base", n), ("user", m)]) for split
        refusal-rate displays; None when a single origin supplies everything
        (the display then stays a plain "matches/N")."""
        _, groups = load_prompts(specs, cache_dir=self.dataset_cache_path(),
                                 refresh=self.refresh_datasets)
        return groups if len(groups) > 1 else None

    def routing_stats_path(self) -> Path:
        """Where the (n_layers, n_experts) routing-ratio .npy lives: written by
        `directions` and read by `optimize`/`export` when expert_mode=routing."""
        if self.routing_stats_file:
            return Path(self.routing_stats_file)
        return Path(self.study_checkpoint_dir) / "routing_stats.npy"
