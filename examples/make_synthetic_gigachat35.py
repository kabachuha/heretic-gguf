#!/usr/bin/env python3
"""Generate a small synthetic GigaChat3.5 (gigachat35) GGUF for smoke testing.

The geometry mirrors the shape of the real 432B-A28B model (Gated-Delta-Net /
full-MLA hybrid attention, gated-RMSNorm sandwiches, DeepSeek-style MoE with
routed + shared experts, optional MTP block) but at a size that fits on a single
consumer GPU / a few GB of RAM.  Weights are random (fixed seed) -- the model is
NOT meant to produce meaningful output, only to exercise the full heretic-gguf
pipeline (load, ablate, optimize, export) and the llama.cpp harness
(expert offload + moe-cache, GENERATE).

The tokenizer is a minimal GPT-2-style byte-level BPE:
  * tokens 0..255  = the 256 GPT-2 byte-level tokens (byte i -> its encoded char)
  * tokens 256..   = the model's special tokens (role_sep, message_sep,
                     file_start, file_end, think-open, think-close), marked CONTROL
  * merges         = empty (every pre-token chunk tokenizes to its bytes)
This tokenizes and detokenizes any ASCII text, and renders the embedded
GigaChat3.5-Reasoning chat template with the special tokens as single tokens.

All special-token byte strings are built from hex (never typed literally) so the
script is self-contained and does not depend on the template file's location.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf

# ---------------------------------------------------------------------------
# Special tokens (raw, WITHOUT the trailing newlines the chat template appends).
# Hex extracted from GigaChat3.5-Reasoning.jinja (add_tokens namespace + the
# forced think-open at the generation-prompt tail).
# ---------------------------------------------------------------------------
SPECIAL_TOKENS: dict[str, str] = {
    "role_sep":    bytes.fromhex("3c7c726f6c655f7365707c3e").decode("utf-8"),
    "message_sep": bytes.fromhex("3c7c6d6573736167655f7365707c3e").decode("utf-8"),
    "file_start":  bytes.fromhex("3c7c66696c657c3e").decode("utf-8"),
    "file_end":    bytes.fromhex("3c7c2f66696c657c3e").decode("utf-8"),
    "think_open":  bytes.fromhex("3c7468696e6b3e").decode("utf-8"),
    "think_close": bytes.fromhex("3c2f7468696e6b3e").decode("utf-8"),
}


def gpt2_byte_tokens() -> list[str]:
    """Reproduce llama.cpp's GPT-2 byte encoder (unicode_byte_to_utf8_map).

    byte 0x21..0x7E, 0xA1..0xAC, 0xAE..0xFF map to themselves; the remaining
    bytes map, in ascending order, to U+0100 (chr(256)), U+0101, ...
    """
    m: dict[int, str] = {}
    for ch in range(0x21, 0x7F):   # '!' .. '~'
        m[ch] = chr(ch)
    for ch in range(0xA1, 0xAD):   # U+00A1 .. U+00AC
        m[ch] = chr(ch)
    for ch in range(0xAE, 0x100):  # U+00AE .. U+00FF
        m[ch] = chr(ch)
    n = 0
    for ch in range(256):
        if ch not in m:
            m[ch] = chr(256 + n)
            n += 1
    return [m[i] for i in range(256)]


# ---------------------------------------------------------------------------
# Geometry (mirrors the PR test fixture + real-model structure, K-quant-friendly
# expert dims = multiples of 256).
# ---------------------------------------------------------------------------
class Geo:
    n_embd   = 256
    n_head   = 8
    n_head_kv = 1
    n_layer  = 4          # main blocks
    n_ff     = 256        # dense FFN intermediate
    n_ff_exp = 256        # routed-expert intermediate (multiple of 256 -> Q2_K)
    n_ff_shexp = 256      # shared-expert intermediate  (multiple of 256 -> Q6_K)
    n_expert = 8
    n_expert_used = 2
    n_expert_shared = 1

    q_lora_rank   = 512
    kv_lora_rank  = 512
    key_mla       = 192   # n_embd_head_k_mla
    value_mla     = 128   # n_embd_head_v_mla
    n_rot         = 64    # rope dim
    key_length    = 576   # kv_lora_rank + n_rot
    value_length  = 512   # kv_lora_rank

    ssm_conv_kernel  = 4
    ssm_inner        = 1024   # = head_v_dim * n_v_heads
    ssm_state        = 128    # head_k_dim == head_v_dim
    ssm_dt_rank      = 8      # n_v_heads
    ssm_group        = 4      # n_k_heads
    ssm_dt_b_c_rms   = True

    full_attn_interval   = 2
    leading_dense_blocks = 1
    norm_eps             = 1e-6
    gating_func          = 2  # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
    ctx_len              = 2048

    # derived SSM geometry
    head_k = ssm_state
    head_v = ssm_state
    n_k_heads = ssm_group
    n_v_heads = ssm_dt_rank
    key_dim   = head_k * n_k_heads          # 512
    value_dim = head_v * n_v_heads          # 1024
    conv_dim  = 2 * key_dim + value_dim     # 2048

    n_embd_head_qk_nope = key_mla - n_rot   # 128
    norm_gate_hidden    = 16


def is_recurrent(il: int, n_mtp: int) -> bool:
    n_layer_all = Geo.n_layer + n_mtp
    if il >= Geo.n_layer:            # MTP block -> full MLA
        return False
    return (il + 1) % Geo.full_attn_interval != 0


def add_kv(w: gguf.GGUFWriter, geo: Geo, n_mtp: int, n_vocab: int) -> None:
    a = "gigachat35"  # (general.architecture is set by the constructor)
    w.add_string("general.basename", "synthetic-gigachat35")
    w.add_string("general.description",
                 "Synthetic GigaChat3.5-Reasoning (gigachat35) for heretic-gguf smoke tests. "
                 "Random weights; not a real model.")
    w.add_uint32(f"{a}.block_count",              geo.n_layer + n_mtp)
    w.add_uint32(f"{a}.embedding_length",         geo.n_embd)
    w.add_uint32(f"{a}.vocab_size",               n_vocab)
    w.add_uint32(f"{a}.context_length",           geo.ctx_len)
    w.add_uint32(f"{a}.feed_forward_length",      geo.n_ff)
    w.add_uint32(f"{a}.attention.head_count",     geo.n_head)
    w.add_uint32(f"{a}.attention.head_count_kv",  geo.n_head_kv)
    w.add_float32(f"{a}.attention.layer_norm_rms_epsilon", geo.norm_eps)
    w.add_uint32(f"{a}.attention.q_lora_rank",    geo.q_lora_rank)
    w.add_uint32(f"{a}.attention.kv_lora_rank",   geo.kv_lora_rank)
    w.add_uint32(f"{a}.attention.key_length_mla", geo.key_mla)
    w.add_uint32(f"{a}.attention.value_length_mla", geo.value_mla)
    w.add_uint32(f"{a}.attention.key_length",     geo.key_length)
    w.add_uint32(f"{a}.attention.value_length",   geo.value_length)
    w.add_uint32(f"{a}.rope.dimension_count",     geo.n_rot)
    w.add_uint32(f"{a}.ssm.conv_kernel",          geo.ssm_conv_kernel)
    w.add_uint32(f"{a}.ssm.inner_size",           geo.ssm_inner)
    w.add_uint32(f"{a}.ssm.state_size",           geo.ssm_state)
    w.add_uint32(f"{a}.ssm.time_step_rank",       geo.ssm_dt_rank)
    w.add_uint32(f"{a}.ssm.group_count",          geo.ssm_group)
    w.add_bool(f"{a}.ssm.dt_b_c_rms",             geo.ssm_dt_b_c_rms)
    w.add_uint32(f"{a}.expert_count",             geo.n_expert)
    w.add_uint32(f"{a}.expert_used_count",        geo.n_expert_used)
    w.add_uint32(f"{a}.expert_shared_count",      geo.n_expert_shared)
    w.add_uint32(f"{a}.expert_feed_forward_length", geo.n_ff_exp)
    w.add_uint32(f"{a}.expert_shared_feed_forward_length", geo.n_ff_shexp)
    w.add_uint32(f"{a}.expert_gating_func",       geo.gating_func)
    w.add_uint32(f"{a}.full_attention_interval",  geo.full_attn_interval)
    w.add_uint32(f"{a}.leading_dense_block_count", geo.leading_dense_blocks)
    # explicit per-layer mask (like the real converter writes); main layers not
    # in the full-attention set are recurrent, MTP blocks are always full MLA
    w.add_array(f"{a}.attention.recurrent_layers",
                [is_recurrent(il, n_mtp) for il in range(geo.n_layer + n_mtp)])
    if n_mtp:
        w.add_uint32(f"{a}.nextn_predict_layers", n_mtp)


def add_tensors(w: gguf.GGUFWriter, geo: Geo, n_mtp: int, n_vocab: int, seed: int) -> None:
    # NOTE on shapes: the pip gguf writer stores GGUF ne[] = reversed(array shape)
    # and writes the array's C-order bytes as-is, so every array is passed with
    # shape = reversed(GGUF ne).  1-D tensors are unaffected.
    rng = np.random.default_rng(seed)

    def r16(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float16)

    def r32(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    def norm1(*shape):
        return np.ones(shape, dtype=np.float32)

    g = geo
    n_layer_all = g.n_layer + n_mtp

    # ---- global ----------------------------------------------------------
    # token_embd / output: ne = (n_embd, n_vocab)
    w.add_tensor("token_embd.weight", r16(n_vocab, g.n_embd))
    w.add_tensor("output.weight",     r16(n_vocab, g.n_embd))
    # output_norm: ne = (n_embd,), gate_up ne = (n_embd, H), gate_down ne = (H, n_embd)
    w.add_tensor("output_norm.weight", norm1(g.n_embd))
    w.add_tensor("output_norm_gate_up.weight",   r16(g.norm_gate_hidden, g.n_embd))
    w.add_tensor("output_norm_gate_down.weight", r16(g.n_embd, g.norm_gate_hidden))

    def common_norms(il: int) -> None:
        p = f"blk.{il}."
        for name in ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"):
            w.add_tensor(f"{p}{name}.weight", norm1(g.n_embd))
            w.add_tensor(f"{p}{name}_gate_up.weight",   r16(g.norm_gate_hidden, g.n_embd))
            w.add_tensor(f"{p}{name}_gate_down.weight", r16(g.n_embd, g.norm_gate_hidden))

    def mla_attention(il: int) -> None:
        p = f"blk.{il}."
        w.add_tensor(f"{p}attn_q_a.weight", r16(g.q_lora_rank, g.n_embd))              # ne (n_embd, q_lora)
        w.add_tensor(f"{p}attn_q_a_norm.weight", norm1(g.q_lora_rank))
        w.add_tensor(f"{p}attn_q_a_norm_gate_up.weight",   r16(g.norm_gate_hidden, g.q_lora_rank))
        w.add_tensor(f"{p}attn_q_a_norm_gate_down.weight", r16(g.q_lora_rank, g.norm_gate_hidden))
        w.add_tensor(f"{p}attn_q_b.weight", r16(g.n_head * g.key_mla, g.q_lora_rank))  # ne (q_lora, n_head*k_mla)
        w.add_tensor(f"{p}attn_kv_a_mqa.weight", r16(g.kv_lora_rank + g.n_rot, g.n_embd))  # ne (n_embd, kv_lora+rope)
        w.add_tensor(f"{p}attn_kv_a_norm.weight", norm1(g.kv_lora_rank))
        w.add_tensor(f"{p}attn_kv_a_norm_gate_up.weight",   r16(g.norm_gate_hidden, g.kv_lora_rank))
        w.add_tensor(f"{p}attn_kv_a_norm_gate_down.weight", r16(g.kv_lora_rank, g.norm_gate_hidden))
        w.add_tensor(f"{p}attn_k_b.weight", r16(g.n_head, g.kv_lora_rank, g.n_embd_head_qk_nope))  # ne (qk_nope, kv_lora, n_head)
        w.add_tensor(f"{p}attn_v_b.weight", r16(g.n_head, g.value_mla, g.kv_lora_rank))            # ne (kv_lora, v_mla, n_head)
        w.add_tensor(f"{p}attn_output.weight", r16(g.n_embd, g.n_head * g.value_mla))              # ne (n_head*v_mla, n_embd)
        # wqkv_gate: ne = (n_embd, n_head * value_mla)  [shape from tensor meta]
        w.add_tensor(f"{p}attn_gate.weight", r16(g.n_head * g.value_mla, g.n_embd))

    def linear_attention(il: int) -> None:
        p = f"blk.{il}."
        w.add_tensor(f"{p}attn_qkv.weight", r16(g.conv_dim, g.n_embd))                   # ne (n_embd, conv_dim)
        w.add_tensor(f"{p}attn_gate.weight", r16(g.value_dim, g.n_embd))                 # ne (n_embd, value_dim)
        # ssm_conv (CPU and CUDA kernels) requires F32 conv weights
        w.add_tensor(f"{p}ssm_conv1d.weight", r32(g.conv_dim, g.ssm_conv_kernel))        # ne (kernel, conv_dim)
        w.add_tensor(f"{p}ssm_dt.bias", r32(g.ssm_dt_rank))
        w.add_tensor(f"{p}ssm_a", r32(g.ssm_dt_rank))
        w.add_tensor(f"{p}ssm_beta.weight",  r16(g.n_v_heads, g.n_embd))                 # ne (n_embd, n_v_heads)
        w.add_tensor(f"{p}ssm_alpha.weight", r16(g.n_v_heads, g.n_embd))
        w.add_tensor(f"{p}ssm_norm.weight", norm1(g.head_v))
        w.add_tensor(f"{p}ssm_out.weight", r16(g.n_embd, g.value_dim))                   # ne (value_dim, n_embd)

    def dense_ffn(il: int) -> None:
        p = f"blk.{il}."
        w.add_tensor(f"{p}ffn_gate.weight", r16(g.n_ff, g.n_embd))
        w.add_tensor(f"{p}ffn_up.weight",   r16(g.n_ff, g.n_embd))
        w.add_tensor(f"{p}ffn_down.weight", r16(g.n_embd, g.n_ff))

    def moe_ffn(il: int) -> None:
        p = f"blk.{il}."
        w.add_tensor(f"{p}ffn_gate_inp.weight", r16(g.n_expert, g.n_embd))               # ne (n_embd, n_expert)
        w.add_tensor(f"{p}exp_probs_b.bias", r32(g.n_expert))
        w.add_tensor(f"{p}ffn_down_exps.weight",   r16(g.n_expert, g.n_embd, g.n_ff_exp))     # ne (ff_exp, n_embd, n_expert)
        w.add_tensor(f"{p}ffn_gate_up_exps.weight", r16(g.n_expert, 2 * g.n_ff_exp, g.n_embd))  # ne (n_embd, 2*ff_exp, n_expert)
        w.add_tensor(f"{p}ffn_gate_shexp.weight", r16(g.n_ff_shexp, g.n_embd))
        w.add_tensor(f"{p}ffn_up_shexp.weight",   r16(g.n_ff_shexp, g.n_embd))
        w.add_tensor(f"{p}ffn_down_shexp.weight", r16(g.n_embd, g.n_ff_shexp))

    for il in range(n_layer_all):
        common_norms(il)
        is_mtp = il >= g.n_layer
        if is_mtp or not is_recurrent(il, n_mtp):
            mla_attention(il)
        else:
            linear_attention(il)
        if il < g.leading_dense_blocks or is_mtp:
            dense_ffn(il)
        else:
            moe_ffn(il)

    if n_mtp:
        il = g.n_layer  # first (and only) MTP block
        p = f"blk.{il}."
        w.add_tensor(f"{p}nextn.eh_proj.weight", r16(g.n_embd, 2 * g.n_embd))            # ne (2*n_embd, n_embd)
        for name in ("nextn.enorm", "nextn.hnorm"):
            w.add_tensor(f"{p}{name}.weight", norm1(g.n_embd))
            w.add_tensor(f"{p}{name}_gate_up.weight",   r16(g.norm_gate_hidden, g.n_embd))
            w.add_tensor(f"{p}{name}_gate_down.weight", r16(g.n_embd, g.norm_gate_hidden))
        w.add_tensor(f"{p}nextn.embed_tokens.weight",     r16(n_vocab, g.n_embd))        # ne (n_embd, n_vocab)
        w.add_tensor(f"{p}nextn.shared_head_head.weight", r16(n_vocab, g.n_embd))
        w.add_tensor(f"{p}nextn.shared_head_norm.weight", norm1(g.n_embd))
        w.add_tensor(f"{p}nextn.shared_head_norm_gate_up.weight",   r16(g.norm_gate_hidden, g.n_embd))
        w.add_tensor(f"{p}nextn.shared_head_norm_gate_down.weight", r16(g.n_embd, g.norm_gate_hidden))


def add_tokenizer(w: gguf.GGUFWriter, n_vocab: int, chat_template: str | None) -> None:
    byte_tokens = gpt2_byte_tokens()
    specials = list(SPECIAL_TOKENS.values())
    assert len(byte_tokens) + len(specials) == n_vocab, (
        f"vocab mismatch: {len(byte_tokens)} + {len(specials)} != {n_vocab}")

    tokens  = byte_tokens + specials
    scores  = [0.0] * n_vocab
    # GGUF token types: 1=NORMAL, 3=CONTROL (control tokens are parsed as whole
    # units by llama.cpp's tokenizer_st_partition when parse_special=true).
    types   = [1] * len(byte_tokens) + [3] * len(specials)

    # "gpt2" selects the BPE tokenizer (a merges key is required; empty is fine).
    # "llama" would select SPM, which is wrong for this byte-level vocab.
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "gigachat")   # -> GPT-2 byte-level BPE (pinned by name)
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.scores", scores)
    w.add_array("tokenizer.ggml.token_type", types)
    # The BPE tokenizer requires a merges key.  The pip gguf writer cannot store
    # an empty array, so use one dummy merge that can never match: the byte
    # tokens for 0x00 and 0x01 (which never occur in the smoke-test text).
    w.add_array("tokenizer.ggml.merges", [chr(256) + " " + chr(257)])
    # GPT-2 style: no forced BOS/EOS; the chat template supplies any BOS.
    w.add_bool("tokenizer.ggml.add_bos_token", False)
    w.add_bool("tokenizer.ggml.add_eos_token", False)
    if chat_template is not None:
        w.add_string("tokenizer.chat_template", chat_template)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--mtp", type=int, default=1, help="number of MTP blocks (0 to disable)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chat-template", default=None,
                    help="path to the GigaChat3.5-Reasoning.jinja template to embed")
    args = ap.parse_args()

    geo = Geo
    n_mtp = max(0, args.mtp)
    n_vocab = 256 + len(SPECIAL_TOKENS)

    chat_template = None
    if args.chat_template:
        chat_template = Path(args.chat_template).read_text(encoding="utf-8")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(out, "gigachat35")

    add_kv(w, geo, n_mtp, n_vocab)
    add_tensors(w, geo, n_mtp, n_vocab, args.seed)
    add_tokenizer(w, n_vocab, chat_template)

    # the pip gguf writer is a state machine; drive it explicitly
    w.open_output_file()
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()  # internally writes the tensor-info section first
    w.flush()
    w.close()
    size = out.stat().st_size
    n_blocks = geo.n_layer + n_mtp
    print(f"wrote {out}  ({size / 1e6:.2f} MB)")
    print(f"  arch=gigachat35  blocks={n_blocks} (main={geo.n_layer}, mtp={n_mtp})")
    print(f"  n_embd={geo.n_embd} n_head={geo.n_head} n_vocab={n_vocab}")
    print(f"  experts={geo.n_expert} (used {geo.n_expert_used}, shared {geo.n_expert_shared}) "
          f"ff_exp={geo.n_ff_exp} ff_shexp={geo.n_ff_shexp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
