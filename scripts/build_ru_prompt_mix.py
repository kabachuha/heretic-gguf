#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the combined EN+RU prompt pools for GigaChat 3.5 abliteration.

Sources (all read from the HF hub, read-only — the outputs below are saved
LOCALLY ONLY and must NOT be pushed back to the hub):

  harmful  = mlabonne/harmful_behaviors (train+test, `text`)
           + heretic-org/Multilingual-Harmless-Harmful
             (config "russian", split "harmful", `text` column)
             NOTE: the "harmless" split of that dataset is broken (it is a
             second harmful set) and is NOT used.
  harmless = mlabonne/harmless_alpaca (train+test, `text`)
           + zjkarina/Vikhr_instruct (train+validation), RU entries only —
             filtered by the exact system prompt
             "Ты полезный помощник по имени Вихрь"; the first user turn of
             each row is the prompt. In the saved structured artifact the
             system prompt is replaced with the generic
             "Ты полезный помощник" (matching the abliteration run's
             system_prompt setting).

Outputs (under --outdir, default <repo>/data):
  vikhr_instruct_ru.jsonl  {"system": "Ты полезный помощник", "user": ...}
                           — the filtered RU subset, one row per line
  combined_harmful.txt     one prompt per line, deduped, shuffled
  combined_harmless.txt    one prompt per line, deduped, shuffled,
                           downsampled to the harmful pool's size AND
                           RU/EN composition (parity)

Usage:  python scripts/build_ru_prompt_mix.py [--outdir DIR] [--seed 42]
Needs:  datasets (pip). Requires network on first run (HF downloads).

The outputs live under <repo>/data/, which is git-ignored: the pools are
local artifacts and must NOT be pushed to the hub or the repository.

Config usage (both pools are the same size, so equal slices stay in
parity; use DISJOINT slices for directions vs. eval):

  good_prompts          = "file:data/combined_harmless.txt[:400]"
  bad_prompts           = "file:data/combined_harmful.txt[:400]"
  eval_harmless_prompts = "file:data/combined_harmless.txt[400:500]"
  eval_harmful_prompts  = "file:data/combined_harmful.txt[400:500]"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

Vikhr_RU_SYSTEM = "Ты полезный помощник по имени Вихрь"
Vikhr_GENERIC_SYSTEM = "Ты полезный помощник"


def log(msg: str) -> None:
    print(msg, flush=True)


def _vikhr_items(row: dict) -> list[dict]:
    """`items` is a message list; some rows store it as a JSON string."""
    it = row["items"]
    if isinstance(it, str):
        return json.loads(it)
    return it


def load_vikhr_ru() -> list[tuple[str, str]]:
    """-> [(system_replaced, first_user_turn)] for the RU subset."""
    from datasets import load_dataset, concatenate_datasets

    ds = load_dataset("zjkarina/Vikhr_instruct")
    rows = concatenate_datasets([ds["train"], ds["validation"]])
    out: list[tuple[str, str]] = []
    skipped = 0
    for row in rows:
        if row["system"] != Vikhr_RU_SYSTEM:
            continue
        try:
            users = [str(m["content"]).strip()
                     for m in _vikhr_items(row) if m["role"] == "user"]
            first = users[0] if users else ""
        except (json.JSONDecodeError, KeyError, TypeError):
            first = ""
        if not first:
            skipped += 1
            continue
        out.append((Vikhr_GENERIC_SYSTEM, first))
    if skipped:
        log(f"vikhr: skipped {skipped} RU rows with no parseable user turn")
    return out


def load_texts(specs: list[tuple[str, str, str, str | None]]) -> list[str]:
    """specs: [(repo, config|None, [splits...], column|None)] — concatenates
    the `column` (or the whole row if None) of each split as strings."""
    from datasets import load_dataset, concatenate_datasets

    out: list[str] = []
    for repo, config, splits, column in specs:
        ds = load_dataset(repo, config) if config else load_dataset(repo)
        part = concatenate_datasets([ds[s] for s in splits])
        if column is None:
            texts = [str(r) for r in part]
        else:
            texts = [str(x) for x in part[column]]
        n0 = len(out)
        out.extend(t for t in texts if t and t.strip())
        log(f"{repo}{':' + config if config else ''} {splits}: "
            f"{len(out) - n0} prompt(s)")
    return out


def normalize(prompt: str) -> str:
    """The file: loader is line-based — flatten internal newlines."""
    return " ".join(prompt.split())


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in items:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def write_pool(path: Path, prompts: list[str]) -> None:
    path.write_text("".join(p + "\n" for p in prompts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # ------------------------------------------------------------- harmful
    log("== harmful ==")
    harmful = load_texts([
        ("mlabonne/harmful_behaviors", None, ["train", "test"], "text"),
        ("heretic-org/Multilingual-Harmless-Harmful", "russian",
         ["harmful"], "text"),
    ])
    harmful = dedupe(harmful)
    log(f"harmful combined (deduped): {len(harmful)}")

    # ------------------------------------------------------------- harmless
    log("== harmless ==")
    vikhr = load_vikhr_ru()
    log(f"vikhr RU subset: {len(vikhr)}")
    (outdir / "vikhr_instruct_ru.jsonl").write_text(
        "".join(json.dumps({"system": s, "user": u}, ensure_ascii=False) + "\n"
                for s, u in vikhr),
        encoding="utf-8",
    )
    harmless = load_texts([
        ("mlabonne/harmless_alpaca", None, ["train", "test"], "text"),
    ])
    harmless.extend(u for _, u in vikhr)
    harmless = dedupe(harmless)
    log(f"harmless combined (deduped, before parity): {len(harmless)}")

    # ------------------------------------------------- parity + shuffle + out
    # Parity with the harmful pool: same total size AND same RU/EN
    # composition (a random downsample would leave only ~8% RU harmless).
    ru = re.compile(r"[а-яА-ЯёЁ]")
    n_ru_h = sum(1 for p in harmful if ru.search(p))
    n_en_h = len(harmful) - n_ru_h
    ru_pool = [p for p in harmless if ru.search(p)]
    en_pool = [p for p in harmless if not ru.search(p)]
    ru_take = min(n_ru_h, len(ru_pool))
    en_take = min(n_en_h, len(en_pool)) + (n_ru_h - ru_take)  # top up shortfall
    harmless = rng.sample(ru_pool, ru_take) + rng.sample(en_pool, en_take)
    log(f"harmless downsampled to {len(harmless)} "
        f"({ru_take} RU / {en_take} EN, matching harmful "
        f"{n_ru_h} RU / {n_en_h} EN)")

    harmful = [normalize(p) for p in harmful]
    harmless = [normalize(p) for p in harmless]
    harmful = dedupe(harmful)
    harmless = dedupe(harmless)
    rng.shuffle(harmful)
    rng.shuffle(harmless)

    write_pool(outdir / "combined_harmful.txt", harmful)
    write_pool(outdir / "combined_harmless.txt", harmless)

    n_ru_h = sum(1 for p in harmful if re.search(r"[а-яА-ЯёЁ]", p))
    n_ru_hl = sum(1 for p in harmless if re.search(r"[а-яА-ЯёЁ]", p))
    log(f"wrote {outdir / 'combined_harmful.txt'}: {len(harmful)} "
        f"({n_ru_h} RU / {len(harmful) - n_ru_h} EN)")
    log(f"wrote {outdir / 'combined_harmless.txt'}: {len(harmless)} "
        f"({n_ru_hl} RU / {len(harmless) - n_ru_hl} EN)")
    log(f"wrote {outdir / 'vikhr_instruct_ru.jsonl'}: {len(vikhr)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
