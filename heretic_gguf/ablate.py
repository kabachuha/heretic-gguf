# SPDX-License-Identifier: AGPL-3.0-or-later
# Ablation math — faithful port of Heretic's parameterized ablation
# (src/heretic/model.py:461-619) to GGUF tensors, with per-expert extensions.
#
# Heretic expresses ablation as LoRA:  ΔW = lora_B @ lora_A
#   lora_A = v^T Ŵ            (Ŵ = row-normalized W unless mode == "none")
#   lora_B = -w · v           (× row norms when mode == "pre")
# so that (W + ΔW) x = W x - w · v (v^T Ŵ x): the v-component of the output
# is subtracted, scaled by the layer weight kernel w.
#
# PERFORMANCE DESIGN (constraint 1 — no full-GGUF writes, no full model in RAM):
#   * The base GGUF is only ever mmap-read; ablations are LoRA adapters in /dev/shm.
#   * `precompute()` makes ONE streaming pass over all ablatable tensors
#     (fast torch MXFP4/Q8_0 dequant, GPU if available), caching per layer:
#       A_l   = v_l^T Ŵ      (per-layer-scope trials are then FREE)
#       norms = row norms of W (needed for lora_B in "pre" mode)
#     Cache size ≈ 300 MB.
#   * Global-scope trials (interpolated direction v' applied to every layer's
#     weights) re-project only the layers inside the trial's weight-kernel
#     window: v'^T Ŵ_L is computed on the fly from mmap (page-cached) raw
#     bytes, chunked per expert group. ~0.1–0.3 s per MoE layer on GPU.
#
# Mode "full" (norm-preserving biprojected abliteration) is nonlinear in the
# layer weight w and needs the full W per trial: it uses the same streaming
# path plus svd_lowrank — exact but slower. The merged exporter never needs
# SVD: it writes the exact full-rank delta.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.linalg as LA

from .archs import ArchSpec, get_arch_spec
from .gguf_io import (
    GGUFModel,
    adapter_tensor_names,
)
from .quants import dequant_mxfp4, dequant_q8_0
from gguf.constants import GGMLQuantizationType as QType


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


def layer_weight(layer_index: int, params: AbliterationParameters) -> float:
    """Heretic's linear weight kernel (model.py:489-506)."""
    distance = abs(layer_index - params.max_weight_position)
    if distance > params.min_weight_distance:
        return 0.0
    if params.min_weight_distance == 0:
        # Only the exact center survives the distance check above.
        return params.max_weight
    return params.max_weight + (distance / params.min_weight_distance) * (
        params.min_weight - params.max_weight
    )


class Ablater:
    def __init__(
        self,
        model: GGUFModel,
        directions: torch.Tensor,  # (n_layers, n_embd) f32, unit norm
        row_normalization: str = "pre",  # "none" | "pre" | "full"
        full_rank: int = 3,
        components: list[str] | None = None,
        device: str | None = None,     # dequant/projection device; None = auto
        expert_chunk: int = 8,         # experts per dequant chunk (VRAM transient
                                       # ≈ 40 MB/expert; keep small — the harness
                                       # occupies most VRAM during optimize)
        seed: int = 42,
    ):
        assert row_normalization in ("none", "pre", "full")
        self.model = model
        self.spec: ArchSpec = get_arch_spec(model.arch, model)
        self.directions = directions.to(torch.float32)
        self.row_normalization = row_normalization
        self.full_rank = full_rank
        self.components = components or list(self.spec.all_components)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.expert_chunk = expert_chunk
        self.seed = seed

        self.n_layers = model.n_layers
        self.n_experts = model.n_experts
        assert directions.shape[0] == self.n_layers

        # Per-layer caches (filled by precompute()):
        #   exps: A (n_experts, in), norms (n_experts, out)
        #   2-D:  A (in,),         norms (out,)
        self._A: dict[str, list[torch.Tensor | None]] = {}
        self._norms: dict[str, list[torch.Tensor | None]] = {}
        self._precomputed = False

    def _layer_direction(self, layer: int) -> torch.Tensor | None:
        """Direction paired with `layer` in per-layer scope (Heretic pairs
        layer L with its OUTPUT direction, residual_directions[L+1]).

        None when no such direction exists: on block-input-capture archs
        (deepseek4) directions[L+1] runs past the end for the last layer —
        its output direction was never captured, so the layer is skipped in
        per-layer scope (see ArchSpec.per_layer_direction_shift)."""
        k = layer + self.spec.per_layer_direction_shift
        return None if k >= self.directions.shape[0] else self.directions[k]

    # ------------------------------------------------------ low-level access
    def _dequant_tensor(self, name: str, e0: int, e1: int) -> torch.Tensor:
        """Dequant expert slice [e0, e1) (or whole 2-D tensor when e1 is None)
        to (rows, in) float32 on self.device, where rows = (e1-e0)*out for
        expert tensors. Uses the fast torch path for MXFP4/Q8_0."""
        ref = self.model.info(name)
        t = ref.reader.tensors[ref.tensor_index]
        raw_np = t.data
        if ref.qtype in (QType.MXFP4, QType.Q8_0):
            if ref.n_experts:
                rows_per_expert = raw_np.shape[1]
                raw_np = raw_np[e0:e1].reshape(-1, raw_np.shape[-1])
            raw = torch.from_numpy(np.ascontiguousarray(raw_np)).to(self.device)
            if ref.qtype == QType.MXFP4:
                return dequant_mxfp4(raw)
            return dequant_q8_0(raw)
        # Fallback: pip gguf lib (CPU).
        if ref.n_experts:
            parts = [
                torch.from_numpy(np.ascontiguousarray(
                    self.model.dequant(name, expert=e))) for e in range(e0, e1)
            ]
            return torch.cat(parts, dim=0).to(self.device)
        return torch.from_numpy(
            np.ascontiguousarray(self.model.dequant(name))).to(self.device)

    @torch.no_grad()
    def _project(self, comp: str, layer: int, v: torch.Tensor):
        """A = v^T Ŵ for one layer/component, streamed in expert chunks.
        Returns (in,) or (n_experts, in) on CPU; None when the layer has no
        such component (hybrid archs)."""
        name = self.spec.tensor_name(comp, layer)
        if name is None:
            return None
        v = v.to(self.device)
        ref = self.model.info(name)
        if not ref.n_experts:
            W = self._dequant_tensor(name, 0, None)
            if self.row_normalization != "none":
                W = F.normalize(W, p=2, dim=1)
            return (v @ W).cpu()
        outs = []
        for e0 in range(0, self.n_experts, self.expert_chunk):
            e1 = min(e0 + self.expert_chunk, self.n_experts)
            W = self._dequant_tensor(name, e0, e1)  # ((e1-e0)*out, in)
            W = W.reshape(e1 - e0, ref.out_features, ref.in_features)
            if self.row_normalization != "none":
                W = F.normalize(W, p=2, dim=2)
            outs.append(torch.einsum("o,eoi->ei", v, W).cpu())
            del W
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _row_norms(self, comp: str, layer: int):
        """Row (output) norms of W: (out,) or (n_experts, out) on CPU;
        None when the layer has no such component."""
        name = self.spec.tensor_name(comp, layer)
        if name is None:
            return None
        ref = self.model.info(name)
        if not ref.n_experts:
            W = self._dequant_tensor(name, 0, None)
            return LA.vector_norm(W, dim=1).cpu()
        outs = []
        for e0 in range(0, self.n_experts, self.expert_chunk):
            e1 = min(e0 + self.expert_chunk, self.n_experts)
            W = self._dequant_tensor(name, e0, e1)
            W = W.reshape(e1 - e0, ref.out_features, ref.in_features)
            outs.append(LA.vector_norm(W, dim=2).cpu())
            del W
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------ pass
    @torch.no_grad()
    def precompute(self, progress=lambda msg: None) -> None:
        """One streaming pass caching per-layer A_l = v_l^T Ŵ and row norms."""
        for comp in self.components:
            self._A[comp] = []
            self._norms[comp] = []
        for layer in range(self.n_layers):
            v = self._layer_direction(layer)
            for comp in self.components:
                # A is pairing-dependent: None for layers with no paired
                # direction (last layer on block-input archs), which
                # per-layer scope then skips. Row norms are a property of W,
                # not of v: GLOBAL scope can still ablate such layers, so
                # norms must be cached for EVERY layer.
                self._A[comp].append(
                    None if v is None else self._project(comp, layer, v))
                if self.row_normalization == "pre":
                    self._norms[comp].append(self._row_norms(comp, layer))
                else:
                    self._norms[comp].append(None)
            if layer % 5 == 0:
                progress(f"precompute layer {layer}/{self.n_layers}")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._precomputed = True

    # ------------------------------------------------------------- assembly
    def _global_direction(self, direction_index: float) -> torch.Tensor:
        weight, index = math.modf(direction_index)
        i = int(index)
        j = min(i + 1, self.n_layers - 1)
        return F.normalize(
            self.directions[i].lerp(self.directions[j], weight), p=2, dim=0
        )

    @torch.no_grad()
    def build_adapter(
        self,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
        expert_scale: np.ndarray | None = None,  # (n_layers, n_experts) f32
        adapter_dtype: np.dtype = np.float32,
        merged: bool = False,
    ):
        """Build one trial's ablation.

        merged=False -> adapter-GGUF tensors (numpy, ggml-reversed dims):
            2-D: lora_a (rank, in), lora_b (out, rank)
            3-D: lora_a (n_experts, rank, in), lora_b (n_experts, out, rank)
        merged=True  -> deltas dict: name -> {expert|None: (B, A)} with
            B (out,), A (in,) f32 rank-1 factors (delta = outer(B, A)).
            Kept factorized so the merged exporter can materialize one
            expert slice at a time instead of holding ~8 GB of f32 deltas
            per MoE layer in RAM (see gguf_io.export_merged).
        """
        if self.row_normalization == "full":
            return self._build_full(direction_index, parameters, expert_scale,
                                    adapter_dtype, merged)
        assert self._precomputed, "call precompute() first"

        v_global = (
            None if direction_index is None else self._global_direction(direction_index)
        )
        out: dict = {}
        for layer in range(self.n_layers):
            # Skip the layer early if no component ablates it.
            weights = {
                comp: layer_weight(layer, parameters[comp]) for comp in self.components
            }
            if all(w == 0.0 for w in weights.values()):
                continue
            for comp, w in weights.items():
                if w == 0.0:
                    continue
                name = self.spec.tensor_name(comp, layer)
                if name is None:
                    continue  # hybrid arch: this layer has no such component
                if v_global is None:
                    v = self._layer_direction(layer)
                    if v is None:
                        continue  # no paired direction for this layer
                    A = self._A[comp][layer]
                else:
                    v = v_global
                    A = self._project(comp, layer, v)
                ref = self.model.info(name)

                # lora_B = -w · v (× row norms in "pre" mode × per-expert s_e)
                B = (-w) * v
                if ref.n_experts:
                    B = B.unsqueeze(0).expand(self.n_experts, -1).contiguous()
                    if self.row_normalization == "pre":
                        B = self._norms[comp][layer] * B
                    if expert_scale is not None:
                        s = torch.from_numpy(expert_scale[layer]).to(torch.float32)
                        B = s.unsqueeze(1) * B
                elif self.row_normalization == "pre":
                    B = self._norms[comp][layer] * B

                if merged:
                    if ref.n_experts:
                        d = out.setdefault(name, {})
                        for e in range(self.n_experts):
                            d[e] = (B[e].numpy(), A[e].numpy())
                    else:
                        out.setdefault(name, {})[None] = (B.numpy(), A.numpy())
                else:
                    la, lb = adapter_tensor_names(name)
                    if ref.n_experts:
                        out[la] = A.unsqueeze(1).numpy().astype(adapter_dtype)
                        out[lb] = B.unsqueeze(2).numpy().astype(adapter_dtype)
                    else:
                        out[la] = A.unsqueeze(0).numpy().astype(adapter_dtype)
                        out[lb] = B.unsqueeze(1).numpy().astype(adapter_dtype)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    # ---------------------------------------------------------- full (slow)
    @torch.no_grad()
    def _build_full(self, direction_index, parameters, expert_scale,
                    adapter_dtype, merged):
        """Exact port of Heretic's RowNormalization.FULL path (model.py:558-611):
        rank-1 step on row-normalized W, renormalize, restore original row
        norms, subtract original; SVD to LoRA rank for adapters (exact full-rank
        delta for merged export).

        Note: with merged=True the full-rank deltas cannot stay factorized,
        so they are materialized eagerly (RAM-hungry on MoE models — prefer
        row_normalization="pre" for merged exports, or --mode adapter)."""
        r = self.full_rank
        v_global = (
            None if direction_index is None else self._global_direction(direction_index)
        )
        out: dict = {}
        for layer in range(self.n_layers):
            weights = {
                comp: layer_weight(layer, parameters[comp]) for comp in self.components
            }
            if all(w == 0.0 for w in weights.values()):
                continue
            if v_global is None:
                v = self._layer_direction(layer)
                if v is None:
                    continue  # no paired direction for this layer
            else:
                v = v_global
            for comp, w in weights.items():
                if w == 0.0:
                    continue
                name = self.spec.tensor_name(comp, layer)
                if name is None:
                    continue  # hybrid arch: this layer has no such component
                ref = self.model.info(name)
                experts = range(self.n_experts) if ref.n_experts else [None]
                A_slices, B_slices = [], []
                for e in experts:
                    if e is None:
                        W = self._dequant_tensor(name, 0, None)
                    else:
                        W = self._dequant_tensor(name, e, e + 1)
                    W_org = W
                    W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
                    W = F.normalize(W, p=2, dim=1)
                    vv = v.to(W.device)
                    lora_A = (vv @ W).view(1, -1)
                    lora_B = (-w * vv).view(-1, 1)
                    if expert_scale is not None and e is not None:
                        lora_B = lora_B * float(expert_scale[layer][e])
                    W = W + lora_B @ lora_A
                    W = F.normalize(W, p=2, dim=1)
                    W = W * W_row_norms
                    delta = W - W_org
                    if merged:
                        out.setdefault(name, {})[e] = delta.cpu().numpy()
                    else:
                        torch.manual_seed(self.seed)
                        U, S, Vh = torch.svd_lowrank(delta, q=2 * r + 4, niter=6)
                        U, S, Vh = U[:, :r], S[:r], Vh[:, :r].T
                        sqrt_S = torch.sqrt(S)
                        Bc = (U @ torch.diag(sqrt_S)).cpu().numpy()
                        Ac = (torch.diag(sqrt_S) @ Vh).cpu().numpy()
                        if e is None:
                            la, lb = adapter_tensor_names(name)
                            out[la] = Ac.astype(adapter_dtype)
                            out[lb] = Bc.astype(adapter_dtype)
                        else:
                            A_slices.append(Ac.astype(adapter_dtype))
                            B_slices.append(Bc.astype(adapter_dtype))
                    del W, W_org, delta
                if not merged and ref.n_experts:
                    la, lb = adapter_tensor_names(name)
                    out[la] = np.stack(A_slices, axis=0)  # (n_experts, r, in)
                    out[lb] = np.stack(B_slices, axis=0)  # (n_experts, out, r)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return out

    # ------------------------------------------------- expert strength maps
    def expert_scale_from_groups(self, group_mults: np.ndarray, n_groups: int) -> np.ndarray:
        """(n_layers, n_experts) scale map from per-group multipliers shared
        across layers. Groups are contiguous expert ranges."""
        if self.n_experts % n_groups != 0:
            raise ValueError(
                f"n_experts ({self.n_experts}) not divisible by n_groups "
                f"({n_groups})")
        g = np.asarray(group_mults, dtype=np.float32)
        per_layer = np.repeat(g, self.n_experts // n_groups).astype(np.float32)
        return np.tile(per_layer, (self.n_layers, 1))

    def expert_scale_from_routing(self, routing_stats: np.ndarray,
                                  strength: float = 1.0) -> np.ndarray:
        """s_e = normalized harmful/harmless routing frequency ratio, mapped to
        [0, strength]. routing_stats: (n_layers, n_experts) f32 ratios (see
        routing_ratio)."""
        r = np.asarray(routing_stats, dtype=np.float32)
        if r.shape != (self.n_layers, self.n_experts):
            raise ValueError(
                f"routing stats shape {r.shape} != "
                f"(n_layers, n_experts) = {(self.n_layers, self.n_experts)}")
        r = np.clip(r, 0, None)
        mx = r.max()
        return (r / mx * strength).astype(np.float32) if mx > 0 else np.zeros_like(r)


def routing_ratio(good_counts: np.ndarray, bad_counts: np.ndarray,
                  eps: float = 1.0) -> np.ndarray:
    """Per-(layer, expert) harmful/harmless routing-frequency ratio from raw
    top-k count histograms (n_layers, n_experts), as dumped by the harness
    during residual collection. Frequencies are normalized per layer (each
    layer routes the same number of tokens); eps is Laplace smoothing so
    never-observed experts get ratio ~1 instead of inf/0."""
    g = np.asarray(good_counts, dtype=np.float64) + eps
    b = np.asarray(bad_counts, dtype=np.float64) + eps
    g /= g.sum(axis=1, keepdims=True)
    b /= b.sum(axis=1, keepdims=True)
    return (b / g).astype(np.float32)
