# SPDX-License-Identifier: AGPL-3.0-or-later
# Driver for the heretic-eval C++ harness (llama.cpp in-tree example).
# Protocol: line-based commands on stdin, "OK"/"ERR ..." (+ optional
# "DIMS ...") on stdout; bulk data via files (ideally in /dev/shm).

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
        self.cmd_timeout = cmd_timeout
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
        cannot block the read forever."""
        fd = self.proc.stdout.fileno()
        deadline = time.time() + timeout
        while True:
            if b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                return line.decode(errors="replace")
            remaining = deadline - time.time()
            if remaining <= 0:
                raise HarnessError(
                    f"timeout after {timeout:.0f}s waiting for {what}")
            r, _, _ = select.select([fd], [], [], remaining)
            if not r:
                raise HarnessError(
                    f"timeout after {timeout:.0f}s waiting for {what}")
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
        lines = self._cmd(f"RESIDUALS {job}")
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
        out.unlink(missing_ok=True)
        job.unlink(missing_ok=True)
        return data.copy()

    def logits(self, prompts: list[dict], prefix: str = "") -> np.ndarray:
        """-> f32 array (n_prompts, n_vocab): raw logits at the last token."""
        out = self._shm(f"log_{time.time_ns()}.f32")
        job = self._write_job("logits", prompts, prefix, out)
        lines = self._cmd(f"LOGITS {job}")
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
        out.unlink(missing_ok=True)
        job.unlink(missing_ok=True)
        return data.copy()

    def generate(self, prompts: list[dict], prefix: str = "",
                 max_tokens: int = 100) -> list[str]:
        out = self._shm(f"gen_{time.time_ns()}.json")
        job = self._write_job("generate", prompts, prefix, out,
                              {"max_tokens": max_tokens})
        self._cmd(f"GENERATE {job}")
        try:
            responses = json.loads(out.read_text())["responses"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            raise HarnessError(
                f"GENERATE {job}: bad result file {out}: {e}") from e
        out.unlink(missing_ok=True)
        job.unlink(missing_ok=True)
        if len(responses) != len(prompts):
            raise HarnessError(
                f"GENERATE {job}: {out}: got {len(responses)} responses for "
                f"{len(prompts)} prompts")
        return responses

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
        try:
            self._cmd("QUIT")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
