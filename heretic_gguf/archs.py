# SPDX-License-Identifier: AGPL-3.0-or-later
# Per-architecture registry — the entire arch-specific surface of heretic-gguf
# in one place. Each supported GGUF architecture registers an
# ArchSpec with:
#
#   * the component -> GGUF tensor mapping (Heretic ablates output projections
#     only: attention out-proj and MLP down-proj);
#   * the residual capture node name (the cb() node whose "<name>-<layer>"
#     instances the heretic-eval harness captures; passed to the harness via
#     the HERETIC_CAPTURE_NODE env var, no rebuild needed per arch);
#   * per-arch defaults: CoT-skip response prefix and the residual capture
#     semantics (capture_point): whether the captured node is a layer OUTPUT
#     (Heretic's hidden_states[il+1]) or a block INPUT (Heretic's
#     residual_directions[il], embedding-equivalent at index 0) — this decides
#     the per-layer direction pairing shift.
#
# Adding a new arch should be ~20 lines: one ArchSpec + one register() call.

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

# Canonical component names (shared by all archs; same names as Heretic's
# ablatable components plus the shared-expert extension).
COMPONENT_ATTN_OUT = "attn.o_proj"
COMPONENT_MLP_DOWN = "mlp.down_proj"
COMPONENT_MLP_DOWN_SHARED = "mlp.down_proj.shared"

# A tensor mapping entry is either a template ("blk.{layer}.ffn_down.weight")
# or a callable(layer) -> template | None (None = component absent on this
# layer, e.g. hybrid stacks where only some layers have full attention).
TensorMapping = Mapping[str, "str | Callable[[int], str | None]"]


@dataclass(frozen=True)
class ArchSpec:
    arch: str                          # GGUF general.architecture string
    capture_node: str                  # cb() node name ("<capture_node>-<il>")
    tensors: TensorMapping = field(default_factory=dict)
    response_prefix: str = ""          # default CoT-skip prefix
    capture_point: str = "layer_output"  # "layer_output" | "block_input":
                                       # what the capture node holds per layer
    notes: str = ""

    def __post_init__(self) -> None:
        if self.capture_point not in ("layer_output", "block_input"):
            raise ValueError(
                f"capture_point must be 'layer_output' or 'block_input', "
                f"got {self.capture_point!r}")

    @property
    def per_layer_direction_shift(self) -> int:
        """Per-layer scope pairs layer L with directions[L + shift].

        Heretic pairs layer L with residual_directions[L+1] — the layer's
        OUTPUT direction (model.py:511). On "layer_output" archs (qwen35moe,
        l_out-<il>) directions[L] IS that output direction: shift 0, exact.
        On "block_input" archs (deepseek4, hc_attn_pre-<il>) directions[L] is
        the layer's INPUT ≡ Heretic's residual_directions[L], so pairing
        shifts to L+1 — and the last layer then has no direction (its output
        was never captured) and is skipped in per-layer scope."""
        return 1 if self.capture_point == "block_input" else 0

    @property
    def all_components(self) -> tuple[str, ...]:
        return tuple(self.tensors)

    def tensor_name(self, component: str, layer: int) -> str | None:
        """GGUF tensor name for (component, layer), or None when the layer
        does not have that component (callers skip such layers)."""
        try:
            entry = self.tensors[component]
        except KeyError:
            raise ValueError(
                f"unknown component {component!r} for arch {self.arch!r} "
                f"(known: {sorted(self.tensors)})") from None
        if callable(entry):
            entry = entry(layer)
        return None if entry is None else entry.format(layer=layer)


# model is duck-typed here (anything with a .metadata dict) to avoid a
# circular import with gguf_io.
_ArchFactory = Callable[[object | None], ArchSpec]
_REGISTRY: dict[str, _ArchFactory] = {}


def register(arch: str, factory: _ArchFactory) -> None:
    _REGISTRY[arch] = factory


def supported_archs() -> list[str]:
    return sorted(_REGISTRY)


def get_arch_spec(arch: str, model: object | None = None) -> ArchSpec:
    try:
        factory = _REGISTRY[arch]
    except KeyError:
        raise ValueError(
            f"unsupported architecture {arch!r} "
            f"(registered: {supported_archs()}) — add an ArchSpec to "
            f"heretic_gguf/archs.py") from None
    return factory(model)


# ---------------------------------------------------------------------------
# deepseek4 (DeepSeek-V4-Flash-0731) — verified against
# src/models/deepseek4.cpp and the UD-Q4_K_XL GGUF tensor list.

_DEEPSEEK4 = ArchSpec(
    arch="deepseek4",
    # Hyper-connections x4: the residual is 4 parallel streams; we capture the
    # collapsed block INPUT "hc_attn_pre-<il>" (single n_embd vector, pre-norm,
    # deepseek4.cpp:1113-1118). That equals Heretic's residual_directions[il]
    # — index 0 IS the embedding-equivalent row (NOT dropped); what is missing
    # is the LAST layer's output direction. Per-layer scope therefore pairs
    # layer L with directions[L+1] (per_layer_direction_shift = 1) and skips
    # the last layer.
    capture_node="hc_attn_pre",
    capture_point="block_input",
    response_prefix="<think></think>",  # reasoning model: skip the CoT block
    tensors={
        # attn_output_a is grouped-low-rank and NOT LoRA-targetable in
        # llama.cpp (raw ggml_mul_mat); attn_output_b is the tensor whose
        # output lives in residual space.
        COMPONENT_ATTN_OUT: "blk.{layer}.attn_output_b.weight",
        COMPONENT_MLP_DOWN: "blk.{layer}.ffn_down_exps.weight",
        COMPONENT_MLP_DOWN_SHARED: "blk.{layer}.ffn_down_shexp.weight",
    },
)
register("deepseek4", lambda model: _DEEPSEEK4)


# ---------------------------------------------------------------------------
# qwen35moe (Qwen3.5/3.6 MoE, e.g. Qwen3.6-35B-A3B) — hybrid gated-delta-net /
# full-attention stack with fused experts. Verified against
# src/models/qwen35moe.cpp @ c5a4a0bb:
#   * end-of-layer residual is cb(cur, "l_out", il) — a plain residual stream
#     (no hyper-connections), so capture is Heretic-exact modulo the dropped
#     embedding row;
#   * res->t_layer_inp[il] is wired (llama_get_embeddings_layer_inp fallback);
#   * build_moe_ffn names top-k ids "ffn_moe_topk-<il>" (routing stats work);
#   * attn_output (full-attn layers), ssm_out (delta-net layers), ffn_down_exps
#     and ffn_down_shexp all go through build_lora_mm/build_moe_ffn -> all
#     LoRA-targetable.

def _qwen35moe(model: object | None) -> ArchSpec:
    interval = 4  # llama.cpp default when no attention.recurrent_layers KV
    if model is not None:
        v = getattr(model, "metadata", {}).get("qwen35moe.full_attention_interval")
        if v is not None:
            try:
                interval = int(v)
            except (TypeError, ValueError):
                pass

    def attn_out(layer: int) -> str:
        # Hybrid stack: full attention every `interval`-th layer
        # ((il+1) % interval == 0), gated delta net (linear attention)
        # elsewhere. ssm_out is the delta-net output projection into residual
        # space — the same ablation target as attn_output.
        if (layer + 1) % interval == 0:
            return "blk.{layer}.attn_output.weight"
        return "blk.{layer}.ssm_out.weight"

    return ArchSpec(
        arch="qwen35moe",
        capture_node="l_out",
        capture_point="layer_output",
        response_prefix="<think></think>",  # thinking model: skip the CoT block
        tensors={
            COMPONENT_ATTN_OUT: attn_out,
            COMPONENT_MLP_DOWN: "blk.{layer}.ffn_down_exps.weight",
            COMPONENT_MLP_DOWN_SHARED: "blk.{layer}.ffn_down_shexp.weight",
        },
        notes=("Plain residual stream (l_out-<il> == Heretic hidden_states[il+1]): "
               "directions[L] is layer L's output direction, so per-layer "
               "pairing is exact (shift 0); the embedding row is absent but "
               "unused by the optimizer ranges in practice. "
               "attn.o_proj covers attn_output on full-attention layers and "
               "ssm_out on gated-delta-net layers (both LoRA-able)."),
    )


register("qwen35moe", _qwen35moe)


# ---------------------------------------------------------------------------
# gigachat35 (GigaChat 3.5, e.g. GigaChat3.5-432B-A28B[-Reasoning]) —
# DeepSeek-V3-style MoE (MLA attention + routed experts + shared expert) with
# a Qwen3-Next-style Gated-Delta-Net / full-MLA hybrid attention stack, gated
# RMSNorm "four-norm sandwich", and optional 2-layer MTP head. Verified
# against src/models/gigachat35.cpp (llama.cpp PR #25342) and its converter
# (conversion/gigachat.py):
#   * end-of-layer residual is cb(cur, "l_out", il) — a plain residual stream
#     (no hyper-connections), so capture is Heretic-exact modulo the dropped
#     embedding row (same as qwen35moe);
#   * block_count INCLUDES the MTP blocks (the converter appends them); GGUFModel
#     strips them into n_layers_nextn, so layer indices below are main-layer
#     indices (0..n_layers-1); the MTP blocks (blk.<n>..blk.<n+mtp-1>) are left
#     untouched by ablation;
#   * attention.recurrent_layers (bool array, one per block) marks the
#     Gated-Delta-Net (linear-attention) layers explicitly; when absent the
#     converter's default applies: recurrent iff (il+1) % full_attention_interval
#     != 0 (interval from full_attention_interval, default 4);
#   * leading_dense_block_count leading layers are dense (ffn_gate/up/down),
#     the rest are MoE (ffn_*_exps 3-D fused + ffn_*_shexp shared expert);
#   * all output projections (attn_output, ssm_out, ffn_down, ffn_down_exps,
#     ffn_down_shexp) go through build_lora_mm/build_moe_ffn -> LoRA-targetable.
#
# The Reasoning chat template (GigaChat3.5-Reasoning.jinja) forces the
# generation prompt to end with the think-open token, and an empty-think
# answer is the think-close token + "\n\n" + text. So the CoT-skip prefix
# appended after the prompt is the think-close token followed by two newlines.

def _gigachat35(model: object | None) -> ArchSpec:
    n_layers = getattr(model, "n_layers", 0) if model is not None else 0
    meta = getattr(model, "metadata", {}) or {} if model is not None else {}

    # explicit per-block recurrent mask (converter writes one per block,
    # MTP blocks included — MTP blocks are never recurrent)
    mask = None
    v = meta.get("gigachat35.attention.recurrent_layers")
    if v is not None:
        try:
            if isinstance(v, np.ndarray):
                v = v.tolist()
            mask = [bool(x) for x in v]
        except (TypeError, ValueError):
            mask = None

    interval = 4
    v = meta.get("gigachat35.full_attention_interval")
    if v is not None:
        try:
            interval = int(v)
        except (TypeError, ValueError):
            pass

    n_dense = 0
    v = meta.get("gigachat35.leading_dense_block_count")
    if v is not None:
        try:
            n_dense = int(v)
        except (TypeError, ValueError):
            pass

    def is_recurrent(layer: int) -> bool:
        if mask is not None and layer < len(mask):
            return mask[layer]
        return (layer + 1) % interval != 0

    def attn_out(layer: int) -> str:
        # full-MLA layers project through attn_output; Gated-Delta-Net layers
        # through ssm_out (both are the block's output projection into
        # residual space — the same ablation target).
        if is_recurrent(layer):
            return "blk.{layer}.ssm_out.weight"
        return "blk.{layer}.attn_output.weight"

    def mlp_down(layer: int) -> str:
        if layer < n_dense:
            return "blk.{layer}.ffn_down.weight"
        return "blk.{layer}.ffn_down_exps.weight"

    def mlp_down_shared(layer: int) -> str | None:
        if layer < n_dense:
            return None  # leading dense layers have no shared expert
        return "blk.{layer}.ffn_down_shexp.weight"

    return ArchSpec(
        arch="gigachat35",
        capture_node="l_out",
        capture_point="layer_output",
        response_prefix="</think>\n\n",  # Reasoning model: skip the CoT block
        tensors={
            COMPONENT_ATTN_OUT: attn_out,
            COMPONENT_MLP_DOWN: mlp_down,
            COMPONENT_MLP_DOWN_SHARED: mlp_down_shared,
        },
        notes=("Plain residual stream (l_out-<il> == Heretic hidden_states[il+1]): "
               "directions[L] is layer L's output direction, so per-layer "
               "pairing is exact (shift 0); the embedding row is absent but "
               "unused by the optimizer ranges in practice. "
               "attn.o_proj covers attn_output on full-MLA layers and ssm_out on "
               "Gated-Delta-Net layers; mlp.down_proj covers ffn_down (leading "
               "dense layers) and ffn_down_exps (MoE layers). block_count "
               "includes the MTP blocks; they are excluded from ablation."),
    )


register("gigachat35", _gigachat35)
