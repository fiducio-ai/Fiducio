"""Calibration metrics for segmentation probabilities.

All metrics accept probabilities of shape ``(B, C, *spatial)`` (class axis at
dimension 1) and integer labels of shape ``(B, *spatial)``, with optional
``mask`` and ``ignore_index``. They return Python floats.
"""

from __future__ import annotations

from typing import Any

import torch

from ..utils import class_last_flatten, safe_log, to_tensor

_EPS = 1e-12


def _flatten_valid(
    probs: Any,
    targets: Any,
    mask: Any | None,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = to_tensor(probs, dtype=torch.float32)
    y = to_tensor(targets).long()
    if p.ndim < 2:
        raise ValueError("probs must have shape (B, C, *spatial)")
    flat, _ = class_last_flatten(p)
    y_flat = y.reshape(-1)
    if y_flat.shape[0] != flat.shape[0]:
        raise ValueError("probs and targets have incompatible shapes")
    valid = y_flat != ignore_index
    if mask is not None:
        valid = valid & to_tensor(mask).reshape(-1).to(torch.bool)
    return flat[valid], y_flat[valid]


def negative_log_likelihood(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
) -> float:
    """Mean negative log-likelihood (cross-entropy) over valid voxels."""
    p, y = _flatten_valid(probs, targets, mask, ignore_index)
    if p.shape[0] == 0:
        return float("nan")
    true_p = p.gather(1, y.unsqueeze(1)).squeeze(1)
    return float((-safe_log(true_p)).mean().item())


def brier_score(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
) -> float:
    """Mean multiclass Brier score over valid voxels."""
    p, y = _flatten_valid(probs, targets, mask, ignore_index)
    if p.shape[0] == 0:
        return float("nan")
    one_hot = torch.zeros_like(p)
    one_hot.scatter_(1, y.unsqueeze(1), 1.0)
    return float((p - one_hot).square().sum(dim=1).mean().item())


def expected_calibration_error(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
    n_bins: int = 15,
) -> float:
    """Top-1 expected calibration error (ECE) with uniform binning.

    The confidence is the maximum predicted probability and the accuracy is
    whether the argmax matches the label. Bins partition ``[0, 1]`` uniformly.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _flatten_valid(probs, targets, mask, ignore_index)
    if p.shape[0] == 0:
        return float("nan")
    confidence, prediction = p.max(dim=1)
    correct = (prediction == y).to(torch.float32)
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    # bin index in [0, n_bins-1]
    idx = torch.bucketize(confidence, edges[1:-1].contiguous(), right=False)
    total = confidence.shape[0]
    ece = torch.zeros((), device=p.device)
    for b in range(n_bins):
        in_bin = idx == b
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        avg_conf = confidence[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece = ece + (count / total) * (avg_conf - avg_acc).abs()
    return float(ece.item())
