# SPDX-License-Identifier: AGPL-3.0-or-later
# Residual direction computation — a faithful port of Heretic's logic
# (src/heretic/main.py:538-579 and model.py:703-785):
#
#   * Heretic captures hidden_states (layer outputs, d_model wide), including
#     the embedding row at index 0, so its residual_directions[k] is the
#     output of layer k-1 (== the input of layer k), and ablation pairs layer
#     L with residual_directions[L+1]. Where our per-layer capture lands in
#     that indexing is the ArchSpec's `capture_point`:
#       - "layer_output" archs (qwen35moe, l_out-<il>): directions[L] = output
#         of layer L ≡ Heretic's residual_directions[L+1]. The embedding row
#         is absent; per-layer pairing is exact (shift 0).
#       - "block_input" archs (deepseek4, hc_attn_pre-<il>): directions[k] =
#         block input of layer k ≡ Heretic's residual_directions[k] — index 0
#         IS the embedding-equivalent row (not dropped); what is missing is
#         the LAST layer's output direction. Per-layer pairing shifts by +1
#         (layer L uses directions[L+1]; the last layer is skipped).
#   * Everything else (winsorization, difference of means, orthogonalization
#     against the good direction, L2 normalization) is identical.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def winsorize(residuals: torch.Tensor, quantile: float) -> torch.Tensor:
    """Symmetric per-(prompt, layer) winsorization, as in Heretic
    (model.py:737-747). Disabled for quantile >= 1."""
    if not (0 <= quantile < 1):
        return residuals
    abs_r = torch.abs(residuals)
    thresholds = torch.quantile(abs_r, quantile, dim=2, keepdim=True)
    return torch.clamp(residuals, -thresholds, thresholds)


def compute_directions(
    good_residuals: torch.Tensor,
    bad_residuals: torch.Tensor,
    orthogonalize: bool = True,
) -> torch.Tensor:
    """(n_good, n_layers, d) x (n_bad, n_layers, d) -> (n_layers, d) unit directions.

    Port of main.py:567-579.
    """
    good_means = good_residuals.mean(dim=0)
    bad_means = bad_residuals.mean(dim=0)

    directions = F.normalize(bad_means - good_means, p=2, dim=1)

    if orthogonalize:
        # https://huggingface.co/blog/grimjim/projected-abliteration
        good_directions = F.normalize(good_means, p=2, dim=1)
        projection = torch.sum(directions * good_directions, dim=1)
        directions = directions - projection.unsqueeze(1) * good_directions
        directions = F.normalize(directions, p=2, dim=1)

    return directions


def direction_at(directions: torch.Tensor, direction_index: float) -> torch.Tensor:
    """Fractional-index direction with linear interpolation between adjacent
    layer directions, then renormalization (model.py:472-480, without the +1
    embedding shift)."""
    weight, index = math.modf(direction_index)
    i = int(index)
    if i >= directions.shape[0] - 1:
        return F.normalize(directions[-1], p=2, dim=0)
    return F.normalize(directions[i].lerp(directions[i + 1], weight), p=2, dim=0)


def direction_lerp_coeffs(direction_index: float) -> tuple[int, int, float, float]:
    """Return (i, j, a, b) such that the interpolated unit direction is
    normalize(a * v_i + b * v_j). Used for exact linear-combination caching of
    lora_A = v^T W: A = (a * A_i + b * A_j) / ||a * v_i + b * v_j||."""
    weight, index = math.modf(direction_index)
    i = int(index)
    j = min(i + 1, 10_000)  # clamped by caller against n_layers-1
    return i, j, 1.0 - weight, weight


def compute_som_directions(
    good_residuals: torch.Tensor,
    bad_residuals: torch.Tensor,
    k: int = 4,
    som_x: int = 4,
    som_y: int = 4,
    iterations: int = 10000,
    lr: float = 0.01,
    sigma: float = 0.5,
    orthogonalize: bool = True,
    progress=lambda msg: None,
) -> torch.Tensor:
    """(n_good, n_layers, d) x (n_bad, n_layers, d) -> (n_layers, k, d)
    unit directions, one per SOM neuron.

    Port of p-e-w/heretic PR #196 (main.py, multidirectional_som branch)
    implementing arXiv:2511.08379: per layer, train a hexagonal SOM on the
    harmful residuals, take the top-k most representative neuron weights as
    the per-layer "bad means", and form each direction as the normalized
    difference from the harmless mean. The optional orthogonalization
    projects each direction out of the (good) mean direction — the same
    projected-abliteration step as compute_directions, applied per
    direction.
    """
    from .som import SOMCalculator

    good_means = good_residuals.mean(dim=0)  # (n_layers, d)
    n_layers = bad_residuals.shape[1]

    bad_means: list[torch.Tensor] = []
    for layer_idx in range(n_layers):
        progress(f"SOM layer {layer_idx + 1}/{n_layers}")
        # (n_bad, d) for this layer
        layer_residuals = bad_residuals[:, layer_idx, :].cpu().float().numpy()
        som_calc = SOMCalculator(som_x, som_y, iterations, lr, sigma)
        som_calc.fit(layer_residuals)

        # The top-k neuron weights are this layer's "bad means". SOM de-
        # duplicates neurons that collapsed to the same weight, so fewer
        # than k may come back — repeat them back, as in the PR.
        t = torch.tensor(
            som_calc.get_top_k_neuron_weights(k=k),
            dtype=good_means.dtype, device=good_means.device)
        num_found = t.shape[0]
        if num_found < k:
            t = t.repeat(k // num_found, 1)
            if t.shape[0] < k:
                t = torch.cat([t, t[0:k - t.shape[0]]], dim=0)
        bad_means.append(t)

    # (n_layers, k, d)
    bad_means = torch.stack(bad_means, dim=0).permute(1, 0, 2)

    directions = [F.normalize(bad_mean - good_means, p=2, dim=1)
                  for bad_mean in bad_means]

    if orthogonalize:
        # https://huggingface.co/blog/grimjim/projected-abliteration
        good_directions = F.normalize(good_means, p=2, dim=1)
        directions = [
            F.normalize(
                d - torch.sum(d * good_directions, dim=1).unsqueeze(1)
                * good_directions,
                p=2, dim=1)
            for d in directions
        ]

    directions = torch.stack(directions, dim=0).permute(1, 0, 2)
    return directions
