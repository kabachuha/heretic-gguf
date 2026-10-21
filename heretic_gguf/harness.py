# SPDX-License-Identifier: AGPL-3.0-or-later
# Driver for the heretic-eval C++ harness (llama.cpp in-tree example).
# Protocol: line-based commands on stdin, "OK"/"ERR ..." (+ optional
# "DIMS ..." and per-prompt "PROGRESS <i>/<n>" keep-alive lines) on stdout;
# bulk data via files (ideally in /dev/shm).

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np


class HarnessError(RuntimeError):
    pass


class HarnessTimeout(HarnessError):
    """No stdout line within cmd_timeout.

    The harness prints a "PROGRESS <done>/<total>" stdout line after every
    completed prompt of RESIDUALS/LOGITS/GENERATE, so a timeout means a
    SINGLE prompt made no progress for cmd_timeout seconds: the process is
    probably wedged and its stdin/stdout state is desynced (it may still be
    executing the command). It must be restarted before any further command
    is sent — see Optimizer._with_harness_retry."""


def read_routing_counts(path: str | Path) -> np.ndarray:
    """Read a routing histogram written by the harness (i32 header
    {n_layers, n_experts} + u32 counts) -> (n_layers, n_experts) u32 array."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(8), dtype=np.int32)
        n_l, n_e = int(hdr[0]), int(hdr[1])
        data = np.frombuffer(f.read(), dtype="<u4")
    if data.size != n_l * n_e:
        raise HarnessError(f"{path}: expected {n_l * n_e} counts, got {data.size}")
    return data.reshape(n_l, n_e).copy()


class Harness:
    def __init__(self, binary: str, model_args: list[str], shm_dir: str = "/dev/shm/heretic_gguf",
                 stderr=None, env_extra: dict[str, str] | None = None,
                 cmd_timeout: float = 1800.0):
        self.shm_dir = Path(shm_dir)
        self.shm_dir.mkdir(parents=True, exist_ok=True)
        # Max seconds allowed WITHOUT any stdout line. Long commands emit
        # per-prompt PROGRESS lines, so this bounds a single stuck prompt —
        # it is NOT a cap on total command duration. (Note: with
        # n_prompts <= n_parallel all prompts decode in one batch, so the
        # first PROGRESS line arrives when the first prompt completes: the
        # timeout must exceed the longest single prompt's generation time.)
        self.cmd_timeout = cmd_timeout
        # Result file of a command we gave up on (HarnessTimeout): the
        # harness may still be running it and write the file after we did —
        # close() sweeps it once the process is actually dead.
        self._abandoned_out: Path | None = None
        self._buf = b""  # raw stdout bytes not yet consumed as lines
        cmd = [binary, *model_args]
        env = dict(os.environ)
        # Stable PCI-bus device order so -ts/-ot device indices mean the same
        # thing on every run (default CUDA order is fastest-first).
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        env.update(env_extra or {})
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,  # None = inherit: harness logs go to our stderr
            text=True,
            bufsize=1,
            env=env,
        )
        # Wait for READY (model load of a 155 GB GGUF takes minutes).
        try:
            t0 = time.time()
            while True:
                line = self._readline(max(1.0, 3600 - (time.time() - t0)),
                                      "READY")
                if not line:
                    raise HarnessError("harness exited during startup")
                if line.strip() == "READY":
                    break
                if time.time() - t0 > 3600:
                    raise HarnessError("harness startup timeout")
        except Exception:
            # Reap the process so a failed startup leaves no zombie behind.
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
            raise

    # ------------------------------------------------------------------ io
    def _readline(self, timeout: float, what: str) -> str:
        """One stdout line (without the trailing newline; "" on EOF) with a
        deadline. The raw fd is read via select so a live-but-wedged harness
        cannot block the read forever.

        Raises HarnessTimeout (not HarnessError) on the deadline: with
        per-prompt PROGRESS lines a deadline hit means the harness is wedged,
        which callers must handle by restarting it, not by continuing."""
        fd = self.proc.stdout.fileno()
        deadline = time.time() + timeout
        while True:
            if b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                return line.decode(errors="replace")
            remaining = deadline - time.time()
            if remaining <= 0:
                raise HarnessTimeout(
                    f"timeout after {timeout:.0f}s without a stdout line "
                    f"waiting for {what}")
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                raise HarnessTimeout(
                    f"timeout after {timeout:.0f}s without a stdout line "
                    f"waiting for {what}")
            chunk = os.read(fd, 65536)
            if not chunk:  # EOF
                line, self._buf = self._buf, b""
                return line.decode(errors="replace")
            self._buf += chunk

    def _cmd(self, line: str) -> list[str]:
        assert self.proc.stdin and self.proc.stdout
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as e:
            raise HarnessError(f"harness died on command: {line}") from e
        lines = []
        while True:
            resp = self._readline(self.cmd_timeout, f"response to: {line}")
            if not resp:
                raise HarnessError(f"harness died on command: {line}")
            if resp.startswith("ERR"):
                raise HarnessError(f"{line}: {resp}")
            if resp == "OK":
                return lines
            lines.append(resp)  # e.g. "DIMS ..."

    def _shm(self, name: str) -> Path:
        return self.shm_dir / name

    # ------------------------------------------------------------- commands
    def set_adapter(self, path: str | None) -> None:
        self._cmd(f"ADAPTER {path if path else 'NONE'}")

    def _write_job(self, kind: str, prompts: list[dict], prefix: str,
                   out_path: Path, extra: dict | None = None) -> Path:
        job = {"prompts": prompts, "prefix": prefix, "out": str(out_path)}
        if extra:
            job.update(extra)
        job_path = self._shm(f"job_{kind}_{os.getpid()}_{time.time_ns()}.json")
        job_path.write_text(json.dumps(job))
        return job_path

    def residuals(self, prompts: list[dict], prefix: str = "",
                  routing_out: str | Path | None = None,
                  n_experts: int = 0) -> np.ndarray:
        """-> f32 array (n_prompts, n_layers, n_embd), last-token collapsed
        residual at each layer's block input.

        If routing_out is given, the harness additionally accumulates per-layer
        top-k expert-id histograms (ffn_moe_topk nodes) over ALL prompt tokens
        and writes (n_layers, n_experts) u32 counts there (see
        read_routing_counts)."""
        out = self._shm(f"res_{time.time_ns()}.f32")
        extra = None
        if routing_out is not None:
            if n_experts <= 0:
                raise ValueError("routing_out requires n_experts > 0")
            extra = {"routing_out": str(routing_out), "n_experts": n_experts}
        job = self._write_job("residuals", prompts, prefix, out, extra)
        try:
            try:
                lines = self._cmd(f"RESIDUALS {job}")
            except HarnessTimeout:
                self._abandoned_out = out  # close() sweeps it after death
                raise
            n_p, n_l, d = self._parse_dims(lines, 3)
            try:
                with open(out, "rb") as f:
                    hdr = np.frombuffer(f.read(12), dtype=np.int32)
                    if tuple(hdr) != (n_p, n_l, d):
                        raise HarnessError(
                            f"RESIDUALS {job}: {out}: header {tuple(hdr)} != "
                            f"DIMS {(n_p, n_l, d)}")
                    data = np.frombuffer(f.read(), dtype="<f4").reshape(n_p, n_l, d)
            except (OSError, ValueError) as e:
                raise HarnessError(
                    f"RESIDUALS {job}: bad result file {out}: {e}") from e
            return data.copy()
        finally:
            # Never leak job/result files into the shm dir: after a timeout
            # the harness may still be running the command (or was abandoned).
            out.unlink(missing_ok=True)
            job.unlink(missing_ok=True)

    def logits(self, prompts: list[dict], prefix: str = "") -> np.ndarray:
        """-> f32 array (n_prompts, n_vocab): raw logits at the last token."""
        out = self._shm(f"log_{time.time_ns()}.f32")
        job = self._write_job("logits", prompts, prefix, out)
        try:
            try:
                lines = self._cmd(f"LOGITS {job}")
            except HarnessTimeout:
                self._abandoned_out = out  # close() sweeps it after death
                raise
            n_p, n_vocab = self._parse_dims(lines, 2)
            try:
                with open(out, "rb") as f:
                    hdr = np.frombuffer(f.read(8), dtype=np.int32)
                    if tuple(hdr) != (n_p, n_vocab):
                        raise HarnessError(
                            f"LOGITS {job}: {out}: header {tuple(hdr)} != "
                            f"DIMS {(n_p, n_vocab)}")
                    data = np.frombuffer(f.read(), dtype="<f4").reshape(n_p, n_vocab)
            except (OSError, ValueError) as e:
                raise HarnessError(
                    f"LOGITS {job}: bad result file {out}: {e}") from e
            return data.copy()
        finally:
            # Never leak job/result files into the shm dir (see residuals()).
            out.unlink(missing_ok=True)
            job.unlink(missing_ok=True)

    def generate(self, prompts: list[dict], prefix: str = "",
                 max_tokens: int = 100) -> list[str]:
        out = self._shm(f"gen_{time.time_ns()}.json")
        job = self._write_job("generate", prompts, prefix, out,
                              {"max_tokens": max_tokens})
        try:
            try:
                self._cmd(f"GENERATE {job}")
            except HarnessTimeout:
                self._abandoned_out = out  # close() sweeps it after death
                raise
            try:
                responses = json.loads(out.read_text())["responses"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
                raise HarnessError(
                    f"GENERATE {job}: bad result file {out}: {e}") from e
            if len(responses) != len(prompts):
                raise HarnessError(
                    f"GENERATE {job}: {out}: got {len(responses)} responses "
                    f"for {len(prompts)} prompts")
            return responses
        finally:
            # Never leak job/result files into the shm dir (see residuals()).
            out.unlink(missing_ok=True)
            job.unlink(missing_ok=True)

    @staticmethod
    def _parse_dims(lines: list[str], n: int) -> tuple[int, ...]:
        for line in lines:
            if line.startswith("DIMS"):
                try:
                    vals = tuple(int(x) for x in line.split()[1:])
                except ValueError as e:
                    raise HarnessError(f"bad DIMS line: {line!r}") from e
                if len(vals) != n:
                    raise HarnessError(
                        f"bad DIMS line: {line!r} (expected {n} values)")
                return vals
        raise HarnessError("no DIMS line in harness response")

    def close(self) -> None:
        # Never block for cmd_timeout here: a WEDGED harness (stuck inside a
        # long command, the HarnessTimeout case) will not read QUIT until it
        # un-sticks, and a respawn must not inherit that stall. Fire QUIT
        # best-effort, then escalate terminate -> kill on short budgets.
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write("QUIT\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        if self._abandoned_out is not None:
            self._abandoned_out.unlink(missing_ok=True)
            self._abandoned_out = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
