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


def raw_tensor_bytes(np_raw: np.ndarray, device: str | torch.device) -> torch.Tensor:
    """Move an mmap-backed uint8 numpy view to a torch tensor (copying if
    needed for alignment/device)."""
    arr = np.ascontiguousarray(np_raw)
    return torch.from_numpy(arr).to(device)
