# SPDX-License-Identifier: AGPL-3.0-or-later
# Terminal UI for the optimize study: an ASCII banner plus a live, in-place
# status dashboard (rich) showing trial progress, elapsed/ETA and the current
# top-10 Pareto trials. A line-based Reporter preserves the old scrolling
# behavior for --plain / non-tty / scriptable runs.

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import ProgressBar
    from rich.table import Table
    from rich.text import Text

    HAVE_RICH = True
except ImportError:  # --plain still works without rich installed
    HAVE_RICH = False

# 5-row block font for the banner (only the glyphs HERETIC-GGUF needs).
_FONT = {
    "H": ["█   █", "█   █", "█████", "█   █", "█   █"],
    "E": ["█████", "█    ", "████ ", "█    ", "█████"],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
    "I": ["█████", "  █  ", "  █  ", "  █  ", "█████"],
    "C": [" ████", "█    ", "█    ", "█    ", " ████"],
    "G": [" ████", "█    ", "█  ██", "█   █", " ████"],
    "U": ["█   █", "█   █", "█   █", "█   █", " ███ "],
    "F": ["█████", "█    ", "████ ", "█    ", "█    "],
    "-": ["     ", "     ", " ███ ", "     ", "     "],
}

SUBTITLE = "Optuna-driven directional ablation on quantized GGUF"


def _render_word(word: str) -> list[str]:
    return [" ".join(_FONT[ch][row] for ch in word) for row in range(5)]


def banner_lines() -> list[str]:
    left = _render_word("HERETIC")
    right = _render_word("-GGUF")
    return [l + " " + r for l, r in zip(left, right)]


def banner_text() -> str:
    return "\n".join(banner_lines())


def _fmt_dur(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


class Reporter:
    """Line-based reporter: reproduces the original scrolling output."""

    live = False

    def __init__(self) -> None:
        # stage -> number of harmful prompts, for "matches/N" displays when
        # seeding from a resumed study. Set by the CLI once scorers exist.
        self.kw_denominators: dict[str, int] = {}

    def __enter__(self) -> "Reporter":
        print(banner_text())
        print()
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def log(self, msg: str) -> None:
        print(msg)

    def phase(self, msg: str) -> None:
        print(msg)

    def baseline(self, kw_display: str) -> None:
        self.log(f"baseline keyword rate: {kw_display}, KL: 0 (by definition)")

    def study_started(self, done: int, total: int, trials: list,
                      stage_b: dict | None = None) -> None:
        print(f"Study: {done} trials done, running {max(0, total - done)} more")

    def trial_started(self, index: int, number: int, total: int) -> None:
        print(f"\n=== Trial {index}/{total} (#{number}) ===")

    def trial_finished(self, index: int, number: int, total: int,
                       kw_display: str, kw: float, kl: float, stage: str,
                       duration: float) -> None:
        print(f"    keyword rate{' [A]' if stage == 'a' else ''}: "
              f"{kw_display}   KL: {kl:.4f}   ({duration:.0f}s)")

    def stage_b_started(self, k: int, n_prompts: int, max_tokens: int) -> None:
        print(f"\n=== Stage B: full keyword evaluation of top {k} trials "
              f"({n_prompts} prompts x {max_tokens} tokens) ===")

    def stage_b_trial(self, number: int, kw_a: float, kw_b: float,
                      kw_display: str, kl: float, duration: float) -> None:
        print(f"  trial #{number}: keyword {kw_a:.3f} [A] -> "
              f"{kw_display} [B]   KL: {kl:.4f}   ({duration:.0f}s)")


class LiveReporter(Reporter):
    """Persistent rich dashboard: banner + progress + top-10 Pareto table,
    updated in place. The instance itself is the Live renderable so elapsed
    time/ETA tick on every refresh."""

    live = True
    _N_TOP = 10
    _N_LOG = 4

    def __init__(self, n_total: int) -> None:
        if not HAVE_RICH:
            raise RuntimeError("rich is not installed")
        super().__init__()
        self.total = n_total
        self.start_time = time.time()
        self.initial_done = 0
        self.finished_this_run = 0
        self.current: tuple[int, int] | None = None  # (index, trial number)
        self.phase_msg = "starting up"
        self.baseline_display = ""
        self.durations: list[float] = []
        self.logs: list[str] = []
        # trial number -> entry; .values is the (kw, kl) objective tuple the
        # Pareto helpers in optimize.py operate on.
        self.entries: dict[int, SimpleNamespace] = {}
        self.console = Console()
        self.live_obj = Live(self, console=self.console, refresh_per_second=1,
                             transient=False)

    def __enter__(self) -> "LiveReporter":
        self.live_obj.__enter__()
        return self

    def __exit__(self, *exc) -> bool:
        self.live_obj.__exit__(*exc)
        return False

    # ------------------------------------------------------------- events
    def log(self, msg: str) -> None:
        self.logs.append(msg)

    def phase(self, msg: str) -> None:
        self.phase_msg = msg

    def baseline(self, kw_display: str) -> None:
        self.baseline_display = kw_display
        self.log(f"baseline refusals: {kw_display}")

    def study_started(self, done: int, total: int, trials: list,
                      stage_b: dict | None = None) -> None:
        self.initial_done = done
        stage_b = stage_b or {}
        for t in trials:
            kw, kl = t.values
            stage = t.user_attrs.get("eval_stage", "full")
            sb = stage_b.get(t.number)
            if sb is not None:
                # full-set score already measured: show it (same as when the
                # stage B pass ran in-process), rank by it in _top()
                stage = "a+b"
                kw_display = self._kw_display(sb["keyword_rate"], "full")
                b_kw = sb["keyword_rate"]
            else:
                kw_display = self._kw_display(kw, stage)
                b_kw = None
            self.entries[t.number] = SimpleNamespace(
                number=t.number, values=(kw, kl), stage=stage,
                kw_display=kw_display, b_kw=b_kw)
        if stage_b:
            self.log(f"loaded {len(stage_b)} existing stage B scores")
        self.log(f"study: {done} trials done, "
                 f"{max(0, total - done)} remaining")

    def trial_started(self, index: int, number: int, total: int) -> None:
        self.current = (index, number)
        self.phase_msg = "building adapter"

    def trial_finished(self, index: int, number: int, total: int,
                       kw_display: str, kw: float, kl: float, stage: str,
                       duration: float) -> None:
        self.entries[number] = SimpleNamespace(
            number=number, values=(kw, kl), stage=stage, kw_display=kw_display,
            b_kw=None)
        self.durations.append(duration)
        self.finished_this_run += 1
        self.current = None
        tag = " [A]" if stage == "a" else ""
        self.log(f"trial #{number} ({index}/{total}): refusals {kw_display} "
                 f"KL {kl:.4f} ({duration:.0f}s){tag}")
        self.phase_msg = "between trials"

    def stage_b_started(self, k: int, n_prompts: int, max_tokens: int) -> None:
        self.phase_msg = (f"stage B: full-set re-score of top {k} trials "
                          f"({n_prompts} x {max_tokens} tokens)")

    def stage_b_trial(self, number: int, kw_a: float, kw_b: float,
                      kw_display: str, kl: float, duration: float) -> None:
        e = self.entries.get(number)
        if e is not None:
            e.stage = "a+b"
            e.b_kw = kw_b
            e.kw_display = kw_display  # full-set score is the meaningful one
        self.log(f"stage B trial #{number}: {kw_a:.3f} [A] -> "
                 f"{kw_display} [B] ({duration:.0f}s)")

    # ----------------------------------------------------------- rendering
    def _kw_display(self, kw: float, stage: str) -> str:
        den = (self.kw_denominators.get(stage)
               or self.kw_denominators.get("full") or 0)
        if den:
            return f"{round(kw * den)}/{den} = {kw:.2%}"
        return f"{kw:.2%}"

    def _top(self) -> list[SimpleNamespace]:
        from .optimize import non_dominated_fronts

        # Rank every trial by its best-known score: the full-set stage B kw
        # where measured, else the stage A objective values — so a strong
        # unmeasured stage A trial shows in its would-be position instead of
        # below all measured ones. Stage A kw is optimistic (B-A averaged
        # +0.085 in followup2), so a stage A row's position is best-case
        # until stage B measures it; the stage column discloses the evidence.
        proxies = [
            SimpleNamespace(
                values=(e.b_kw if e.b_kw is not None else e.values[0],
                        e.values[1]),
                entry=e)
            for e in self.entries.values()
        ]
        rows: list[SimpleNamespace] = []
        for front in non_dominated_fronts(proxies):
            rows.extend(p.entry for p in sorted(front, key=lambda p: p.values))
            if len(rows) >= self._N_TOP:
                break
        return rows[: self._N_TOP]

    def _banner(self):
        left = _render_word("HERETIC")
        right = _render_word("-GGUF")
        lines = []
        for l, r in zip(left, right):
            t = Text()
            t.append(l, style="bold bright_red")
            t.append(" ")
            t.append(r, style="bold grey93")
            lines.append(t)
        lines.append(Text(SUBTITLE, style="dim italic"))
        return Panel(Align.center(Group(*lines)), border_style="red",
                     padding=(0, 1))

    def _status(self):
        done = self.initial_done + self.finished_this_run
        line = Text()
        if self.current is not None:
            index, number = self.current
            line.append(f" Trial {index}/{self.total} ", style="bold")
            line.append(f"(#{number})  ", style="dim")
        else:
            line.append(f" Trials {done}/{self.total}   ", style="bold")
        line.append(f"elapsed {_fmt_dur(time.time() - self.start_time)}  ")
        if self.durations and done < self.total:
            eta = (sum(self.durations[-20:]) / len(self.durations[-20:])) \
                * (self.total - done)
            line.append(f"ETA ~{_fmt_dur(eta)}  ")
        line.append(f"· {self.phase_msg}", style="cyan")
        bar = ProgressBar(total=max(self.total, 1), completed=done,
                          complete_style="bright_red", finished_style="green")
        return Panel(Group(line, bar), border_style="grey37", padding=(0, 1))

    def _top_table(self):
        table = Table(box=box.SIMPLE_HEAD, expand=True)
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Trial", justify="right")
        table.add_column("Refusals")
        table.add_column("KL", justify="right")
        table.add_column("Stage", style="dim")
        rows = self._top()
        for rank, e in enumerate(rows, 1):
            stage = {"a": "A", "a+b": "A→B", "full": "full"}.get(e.stage, e.stage)
            table.add_row(str(rank), f"#{e.number}", e.kw_display,
                          f"{e.values[1]:.4f}", stage)
        if not rows:
            table.add_row("—", "—", "no completed trials yet", "—", "")
        title = (f"top {self._N_TOP} trials (best-known score; "
                 f"stage A rows are optimistic until B-measured)")
        if self.baseline_display:
            title += f" · baseline refusals {self.baseline_display}"
        return Panel(table, title=title, border_style="grey37", padding=(0, 1))

    def _recent(self):
        # fixed panels above take banner 8 + status 4 + table (_N_TOP + 6);
        # show as many log lines as the stretched panel can hold.
        avail = max(self._N_LOG,
                    self.console.size.height - (8 + 4 + self._N_TOP + 6) - 2)
        body = Text("\n".join(self.logs[-avail:]) or "…", style="dim")
        return Panel(body, title="recent", border_style="grey30",
                     padding=(0, 1))

    def __rich__(self):
        layout = Layout()
        layout.split_column(
            Layout(self._banner(), name="banner", size=8),
            Layout(self._status(), name="status", size=4),
            # fixed height: the top-10 table is the key info and must never be
            # squeezed on short terminals; the log below absorbs overflow.
            # (_N_TOP rows + 2 header lines + 2 panel borders + 2 padding)
            Layout(self._top_table(), name="top", size=self._N_TOP + 6),
            Layout(self._recent(), name="recent", ratio=1),
        )
        return layout


def make_reporter(settings, plain: bool = False) -> Reporter:
    """Live dashboard on a real terminal; scrolling lines otherwise."""
    if plain or not sys.stdout.isatty() or not HAVE_RICH:
        return Reporter()
    return LiveReporter(settings.n_trials)
