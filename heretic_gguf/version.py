# SPDX-License-Identifier: AGPL-3.0-or-later
# Provenance helpers: exact git commit hashes of the two trees this project
# depends on (llama.cpp for the harness, this repo for the Python side),
# recorded in study user attributes and adapter GGUF metadata.

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit(repo_dir: str | Path) -> str | None:
    """Full HEAD commit hash of the git repo at `repo_dir`, or None."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _find_repo_root(path: Path) -> Path | None:
    """Walk up from `path` to the first directory containing a .git entry."""
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def llama_cpp_commit(harness_binary: str | Path) -> str | None:
    """Commit of the llama.cpp tree containing the heretic-eval harness.

    Derived by walking up from the harness binary (build/bin/heretic-eval)
    to the enclosing git repo. Returns None if the binary is not inside a
    git checkout.
    """
    root = _find_repo_root(Path(harness_binary).resolve().parent)
    return git_commit(root) if root else None


def heretic_gguf_commit() -> str | None:
    """Commit of the heretic-gguf repository (the package's parent dir)."""
    root = _find_repo_root(Path(__file__).resolve().parent)
    return git_commit(root) if root else None
