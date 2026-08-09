# SPDX-License-Identifier: AGPL-3.0-or-later
# GGUF I/O for heretic-gguf: multi-shard read-only model access,
# LoRA adapter writing, and merged-GGUF export (copy-and-patch).
#
# Conventions: ggml stores 2-D weights as [in, out]; the pip `gguf` package
# dequantizes to numpy arrays shaped (out, in) — i.e. PyTorch convention,
# out = W @ x. All Python-side code in this project uses (out, in).
# 3-D expert tensors dequantize per expert slice to (out, in) as well.

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import gguf
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType as QType
from gguf.quants import dequantize, quantize


@dataclass
class TensorRef:
    name: str
    shard: int
    reader: GGUFReader
    tensor_index: int
    qtype: QType
    ggml_shape: tuple[int, ...]  # [ne0, ne1, ...] = [in, out, ...]

    @property
    def n_experts(self) -> int:
        return self.ggml_shape[2] if len(self.ggml_shape) == 3 else 0

    @property
    def out_features(self) -> int:
        return self.ggml_shape[1]

    @property
    def in_features(self) -> int:
        return self.ggml_shape[0]


class GGUFModel:
    """Read-only multi-shard GGUF index with lazy, mmap-backed tensor access."""

    def __init__(self, first_shard_path: str | Path):
        self.first_shard_path = Path(first_shard_path)
        self.readers: list[GGUFReader] = []
        self.shard_paths: list[Path] = []
        self.tensors: dict[str, TensorRef] = {}

        r0 = GGUFReader(str(self.first_shard_path))
        self.readers.append(r0)
        self.shard_paths.append(self.first_shard_path)
        self.metadata = self._read_metadata(r0)

        # Discover sibling shards (llama.cpp split naming: -0000N-of-0000M.gguf)
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", self.first_shard_path.name)
        if m:
            if int(m.group(2)) != 1:
                raise ValueError(
                    f"multi-shard model: pass the FIRST shard "
                    f"({m.group(1)}-00001-of-{m.group(3)}.gguf), "
                    f"not {self.first_shard_path.name}")
            count = int(m.group(3))
            for i in range(2, count + 1):
                p = self.first_shard_path.with_name(
                    f"{m.group(1)}-{i:05d}-of-{count:05d}.gguf"
                )
                if not p.exists():
                    raise FileNotFoundError(f"missing shard: {p}")
                self.readers.append(GGUFReader(str(p)))
                self.shard_paths.append(p)

        for shard_idx, reader in enumerate(self.readers):
            for ti, t in enumerate(reader.tensors):
                self.tensors[t.name] = TensorRef(
                    name=t.name,
                    shard=shard_idx,
                    reader=reader,
                    tensor_index=ti,
                    qtype=t.tensor_type,
                    ggml_shape=tuple(int(d) for d in t.shape),
                )

        self.arch = self._meta_str("general.architecture")
        self.n_layers = int(self._meta_val(f"{self.arch}.block_count"))
        self.n_embd = int(self._meta_val(f"{self.arch}.embedding_length"))
        # Dense models have no expert_count key -> 0 (all 3-D expert logic
        # keys off TensorRef.n_experts / this value and no-ops).
        self.n_experts = int(self._meta_val(f"{self.arch}.expert_count")
                             if f"{self.arch}.expert_count" in self.metadata else 0)

    # ------------------------------------------------------------------ meta
    @staticmethod
    def _read_metadata(reader: GGUFReader) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in reader.fields.items():
            parts = v.parts
            if len(v.data) == 1:
                out[k] = parts[v.data[0]]
            else:
                out[k] = [parts[d] for d in v.data]
        return out

    def _meta_val(self, key: str):
        v = self.metadata[key]
        if isinstance(v, list) and len(v) == 1:
            return v[0]
        if isinstance(v, np.ndarray):
            return v.flat[0]
        return v

    def _meta_str(self, key: str) -> str:
        v = self.metadata[key]
        if isinstance(v, (bytes, bytearray)):
            return bytes(v).decode("utf-8")
        if isinstance(v, np.ndarray):  # string fields come back as uint8 arrays
            return bytes(v.tolist()).decode("utf-8")
        if isinstance(v, list):
            return "".join(
                bytes(x.tolist()).decode("utf-8") if isinstance(x, np.ndarray) else str(x)
                for x in v
            )
        return str(v)

    # --------------------------------------------------------------- tensors
    def info(self, name: str) -> TensorRef:
        return self.tensors[name]

    def dequant(self, name: str, expert: int | None = None) -> np.ndarray:
        """Dequantize to float32 numpy (out, in). For 3-D expert tensors, pass
        `expert` to get that expert's 2-D slice (mmap-lazy: only that expert's
        blocks are paged in)."""
        ref = self.tensors[name]
        t = ref.reader.tensors[ref.tensor_index]
        raw = t.data  # mmap-backed uint8 view for quantized, typed array otherwise
        if ref.n_experts:
            if expert is None:
                raise ValueError(f"{name} is a fused expert tensor; pass expert=<idx>")
            raw = raw[expert]
        return dequantize(raw, ref.qtype).astype(np.float32)

    def dequant_torch(self, name: str, expert: int | None = None,
                      device: str | torch.device = "cpu") -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(self.dequant(name, expert))).to(device)

    # --------------------------------------------------------- adapter write
    @staticmethod
    def write_adapter(path: str | Path, arch: str, rank: int,
                      tensors: dict[str, np.ndarray],
                      metadata: dict[str, str | int | float | None] | None = None
                      ) -> None:
        """Write a llama.cpp-compatible LoRA adapter GGUF.

        `tensors` maps adapter tensor names ("<model tensor>.lora_a"/".lora_b")
        to numpy arrays in ggml-reversed dims:
          2-D: lora_a (rank, in), lora_b (out, rank)
          3-D: lora_a (n_experts, rank, in), lora_b (n_experts, out, rank)
        llama.cpp applies scale = alpha / rank, so alpha = rank means the
        adapter applies at exactly strength 1.

        `metadata` (optional) is embedded as `adapter.heretic.<key>` KV pairs
        (str/int/float values; None values are skipped). llama.cpp ignores
        unknown keys; they exist for downstream humans and tooling
        (`strings adapter.gguf | grep adapter.heretic`).
        """
        writer = GGUFWriter(str(path), arch=arch)
        writer.add_key_value("general.type", "adapter", gguf.GGUFValueType.STRING)
        writer.add_key_value("adapter.type", "lora", gguf.GGUFValueType.STRING)
        writer.add_float32("adapter.lora.alpha", float(rank))
        for key, value in (metadata or {}).items():
            if value is None:
                continue
            k = f"adapter.heretic.{key}"
            if isinstance(value, bool) or isinstance(value, int):
                writer.add_int64(k, int(value))
            elif isinstance(value, float):
                writer.add_float64(k, float(value))
            else:
                writer.add_key_value(k, str(value), gguf.GGUFValueType.STRING)
        for name, arr in tensors.items():
            writer.add_tensor(name, np.ascontiguousarray(arr))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()

    # ---------------------------------------------------------- merged export
    def export_merged(self, out_dir: str | Path,
                      deltas: dict[str, dict[int | None, np.ndarray]],
                      requant: str = "orig") -> list[Path]:
        """Write a full merged model by copy-and-patch.

        Each shard is copied verbatim (metadata, tokenizer, and all untouched
        tensors stay byte-identical — zero quantization loss), then every
        ablated tensor is patched in place: dequantize -> add delta ->
        requantize to the SAME type. Requantized blocks have exactly the
        original byte size (verified for MXFP4 and Q8_0), so tensor offsets
        never move and no metadata needs rewriting.

        `deltas`: model tensor name -> {expert (or None for 2-D): delta}.
        A delta is either a full (out, in) f32 array (row_normalization="full")
        or a (B, A) tuple of f32 vectors with B (out,), A (in,) — the rank-1
        factorization, materialized per expert slice at patch time so peak
        memory stays at a few hundred MB regardless of model size.
        Only `requant="orig"` is supported by design. To get edited tensors in
        a different dtype, use --mode adapter instead (lossless, tiny, and
        needs no ~155 GB of free disk).

        NOTE: needs roughly the model's size in free disk space at `out_dir`.
        """
        if requant != "orig":
            raise ValueError("patch-in-place export only supports requant='orig'")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        delta_by_shard: dict[int, dict[str, dict[int | None, np.ndarray]]] = {}
        for name, per_expert in deltas.items():
            ref = self.tensors[name]
            delta_by_shard.setdefault(ref.shard, {})[name] = per_expert

        for shard_idx, reader in enumerate(self.readers):
            src = self.shard_paths[shard_idx]
            out_path = out_dir / src.name
            print(f"copying {src.name} ({src.stat().st_size / 1e9:.1f} GB)...")
            shutil.copyfile(src, out_path)
            shard_deltas = delta_by_shard.get(shard_idx, {})
            if not shard_deltas:
                written.append(out_path)
                continue

            with open(out_path, "r+b") as f:
                for t in reader.tensors:
                    if t.name not in shard_deltas:
                        continue
                    ref = self.tensors[t.name]
                    per_expert = shard_deltas[t.name]
                    if ref.n_experts:
                        expert_nbytes = t.n_bytes // ref.n_experts
                        for e, delta in per_expert.items():
                            if e is None or not 0 <= e < ref.n_experts:
                                raise ValueError(
                                    f"{t.name}: bad expert index {e} "
                                    f"(n_experts={ref.n_experts})")
                            w = self.dequant(t.name, expert=e) + _materialize(delta)
                            q = quantize(np.ascontiguousarray(w, dtype=np.float32),
                                         ref.qtype)
                            raw = np.ascontiguousarray(q).view(np.uint8).reshape(-1)
                            if raw.nbytes != expert_nbytes:
                                raise ValueError(
                                    f"{t.name} expert {e}: requant size changed "
                                    f"({raw.nbytes} != {expert_nbytes})")
                            f.seek(t.data_offset + e * expert_nbytes)
                            f.write(raw.tobytes())
                    else:
                        if set(per_expert) != {None}:
                            raise ValueError(
                                f"{t.name}: 2-D tensor deltas must be keyed by "
                                f"None, got {list(per_expert)}")
                        w = self.dequant(t.name) + _materialize(per_expert[None])
                        q = quantize(np.ascontiguousarray(w, dtype=np.float32),
                                     ref.qtype)
                        raw = np.ascontiguousarray(q).view(np.uint8).reshape(-1)
                        if raw.nbytes != t.n_bytes:
                            raise ValueError(
                                f"{t.name}: requant size changed "
                                f"({raw.nbytes} != {t.n_bytes})")
                        f.seek(t.data_offset)
                        f.write(raw.tobytes())
                    print(f"  patched {t.name}")
            written.append(out_path)
        return written


# ---------------------------------------------------------------------------
# Component -> tensor mapping now lives in heretic_gguf/archs.py (per-arch
# ArchSpec registry). adapter_tensor_names stays here next to write_adapter.

def adapter_tensor_names(model_tensor: str):
    """llama.cpp adapter naming: '<model tensor name>.lora_a' / '.lora_b'."""
    return f"{model_tensor}.lora_a", f"{model_tensor}.lora_b"


def _materialize(delta) -> np.ndarray:
    """Merged-export deltas are stored either as a full (out, in) array or as
    rank-1 factors (B, A) with B (out,), A (in,) — expand the latter."""
    if isinstance(delta, tuple):
        return np.outer(delta[0], delta[1])
    return delta
