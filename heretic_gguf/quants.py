# SPDX-License-Identifier: AGPL-3.0-or-later
# Fast vectorized dequantization for the quant types used by DeepSeek-V4's
# ablatable tensors. Layouts verified against ggml (ggml-common.h /
# ggml-quants.c) and cross-checked numerically against the pip `gguf` package.
#
# MXFP4 (ggml block_mxfp4): 17 bytes per 32 values = 1 E8M0 scale byte +
# 16 nibble bytes. Byte j holds value j in the LOW nibble and value j+16 in
# the HIGH nibble. Nibbles index the E2M1 LUT; value = E2M1 * 2^(e-127).

from __future__ import annotations

import numpy as np
import torch

# E2M1 values (ggml stores 2x E2M1 in kvalues_mxfp4 and compensates with the
# E8M0-to-float halving; net is exactly the OCP MXFP4 spec).
_E2M1 = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.float32,
) * 0.5


def dequant_mxfp4(raw: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """raw: uint8 tensor (rows, 17*B) on CPU or CUDA -> (rows, 32*B) float.

    All ops are vectorized; on CUDA this is orders of magnitude faster than
    the reference numpy path (~0.1 s vs ~12 s per DeepSeek-V4 expert layer).
    """
    rows, nbytes = raw.shape
    assert nbytes % 17 == 0
    nb = nbytes // 17
    lut = _E2M1.to(raw.device)
    blocks = raw.reshape(rows, nb, 17).to(torch.int32)
    e = blocks[..., 0]
    qs = blocks[..., 1:]
    lo = lut[qs & 0x0F]
    hi = lut[qs >> 4]
    vals = torch.cat([lo, hi], dim=-1)  # (rows, nb, 32)
    scales = torch.exp2((e - 127).to(torch.float32))
    return (vals * scales[..., None]).reshape(rows, -1).to(out_dtype)


def dequant_q8_0(raw: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """raw: uint8 tensor (rows, 34*B) -> (rows, 32*B) float.
    Q8_0 block: fp16 scale + 32 int8 values."""
    rows, nbytes = raw.shape
    assert nbytes % 34 == 0
    nb = nbytes // 34
    blocks = raw.reshape(rows, nb, 34)
    d = blocks[..., :2].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    q = blocks[..., 2:].to(torch.int8).to(torch.float32)
    return (q * d[..., None]).reshape(rows, -1).to(out_dtype)


# ---------------------------------------------------------------------------
# K-quants (256-value super-blocks) — layouts from ggml-common.h, bit
# extraction from ggml-quants.c dequantize_row_q2_K / dequantize_row_q6_K.
# The per-value byte/bit indices are precomputed once and reused via gather,
# which keeps the whole dequant to a handful of vectorized ops.

_K_INDEX_CACHE: dict[str, torch.Tensor] = {}


def _k_index_tensors(name: str, device) -> torch.Tensor:
    if name not in _K_INDEX_CACHE:
        if name == "q2_k":
            # v = g*16 + l, g = group (0..15), l = local (0..15);
            # c = g//8 (128-chunk), p = g%2, shift = 2*((g%4)//2)
            v = np.arange(256, dtype=np.int32)
            g, l = v // 16, v % 16
            idx_qs = (g // 8) * 32 + (g % 2) * 16 + l
            sh_qs = 2 * ((g % 8) // 2)
            _K_INDEX_CACHE[name] = torch.stack([
                torch.from_numpy(idx_qs.astype(np.int64)),
                torch.from_numpy(sh_qs.astype(np.int64)),
                torch.from_numpy(g.astype(np.int64)),
            ], dim=1)
        elif name == "q6_k":
            # v = g*16 + l (g = 16-value group 0..15). The 256 values are
            # laid out so that ql2/qh2 row i = v//32, col jj = v%32:
            #   ql (128 B): byte 64*(i//4) + 32*(i%2) + jj, shifted by
            #     4*((i//2)%2) (low vs high nibble of the byte)
            #   qh (64 B):  byte 32*(i//4) + jj, shifted by 2*(i%4)
            #   scales:     int8 scale g = v//16
            v = np.arange(256, dtype=np.int32)
            i = v // 32
            jj = v % 32
            g = v // 16
            idx_ql = 64 * (i // 4) + 32 * (i % 2) + jj  # low-4-bit byte
            sh_ql = 4 * ((i // 2) % 2)                  # low vs high nibble
            idx_qh = 32 * (i // 4) + jj                 # high-2-bit byte
            sh_qh = 2 * (i % 4)                         # 2-bit slot
            idx_sc = g                                  # int8 scale index
            _K_INDEX_CACHE[name] = torch.stack([
                torch.from_numpy(idx_ql.astype(np.int64)),
                torch.from_numpy(sh_ql.astype(np.int64)),
                torch.from_numpy(idx_qh.astype(np.int64)),
                torch.from_numpy(sh_qh.astype(np.int64)),
                torch.from_numpy(idx_sc.astype(np.int64)),
            ], dim=1)
        else:
            raise ValueError(name)
    t = _K_INDEX_CACHE[name]
    return t if t.device == torch.device(device) else t.to(device)


def dequant_q2_k(raw: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """raw: uint8 tensor (rows, 84*B) -> (rows, 256*B) float.
    block_q2_K (84 B per 256 values): scales[16] (4-bit scale | 4-bit min
    per 16-value group), qs[64] (2-bit quants), d, dmin (fp16).
    value = d*(sc&0xF)*q - dmin*(sc>>4)."""
    rows, nbytes = raw.shape
    assert nbytes % 84 == 0
    nb = nbytes // 84
    b = raw.reshape(rows, nb, 84)
    idx = _k_index_tensors("q2_k", raw.device)
    sc = b[..., 0:16].to(torch.int32)
    qs = b[..., 16:80].to(torch.int32)
    d = b[..., 80:82].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    dmin = b[..., 82:84].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    i0 = idx[:, 0].unsqueeze(0).expand(rows, nb, -1)
    i1 = idx[:, 1].unsqueeze(0).expand(rows, nb, -1)
    i2 = idx[:, 2].unsqueeze(0).expand(rows, nb, -1)
    q = (qs.gather(2, i0) >> i1) & 3
    s = sc.gather(2, i2)
    dl = d[:, :, None] * (s & 15)
    ml = dmin[:, :, None] * (s >> 4)
    return (dl * q - ml).reshape(rows, -1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """raw: uint8 tensor (rows, 210*B) -> (rows, 256*B) float.
    block_q6_K (210 B per 256 values): ql[128] (low 4 bits), qh[64] (high
    2 bits), scales[16] (int8 per 16-value group), d (fp16).
    value = d * scale * ((low | high<<4) - 32)."""
    rows, nbytes = raw.shape
    assert nbytes % 210 == 0
    nb = nbytes // 210
    b = raw.reshape(rows, nb, 210)
    idx = _k_index_tensors("q6_k", raw.device)
    ql = b[..., 0:128].to(torch.int32)
    qh = b[..., 128:192].to(torch.int32)
    # Scales are int8 in the K-quant layout: wrap 0..255 to -128..127.
    sc = b[..., 192:208].to(torch.int8).to(torch.int32)
    d = b[..., 208:210].contiguous().view(torch.float16).squeeze(-1).to(torch.float32)
    i0 = idx[:, 0].unsqueeze(0).expand(rows, nb, -1)
    i1 = idx[:, 1].unsqueeze(0).expand(rows, nb, -1)
    i2 = idx[:, 2].unsqueeze(0).expand(rows, nb, -1)
    i3 = idx[:, 3].unsqueeze(0).expand(rows, nb, -1)
    i4 = idx[:, 4].unsqueeze(0).expand(rows, nb, -1)
    low = (ql.gather(2, i0) >> i1) & 0xF
    high = (qh.gather(2, i2) >> i3) & 0x3
    q = (low | (high << 4)) - 32
    s = sc.gather(2, i4)
    return (d[:, :, None] * s * q).reshape(rows, -1).to(out_dtype)


def raw_tensor_bytes(np_raw: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """Move an mmap-backed uint8 numpy view to a torch tensor (copying if
    needed for alignment/device)."""
    arr = np.ascontiguousarray(np_raw)
    return torch.from_numpy(arr).to(device)
