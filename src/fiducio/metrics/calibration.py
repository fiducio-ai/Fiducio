"""Calibration metrics for segmentation probabilities.

All metrics accept probabilities of shape ``(B, C, *spatial)`` (class axis at
dimension 1) and integer labels of shape ``(B, *spatial)``, with optional
``mask`` and ``ignore_index``. They return Python floats (or, for
:func:`reliability_curve`, a :class:`ReliabilityCurve`).
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class ReliabilityCurve:
    """Per-bin reliability statistics from top-1 confidence binning.

    Attributes
    ----------
    bin_edges:
        ``(n_bins + 1,)`` bin boundaries in ``[0, 1]``.
    bin_confidence:
        ``(n_bins,)`` mean predicted confidence per bin (0 for empty bins).
    bin_accuracy:
        ``(n_bins,)`` mean accuracy per bin (0 for empty bins).
    bin_counts:
        ``(n_bins,)`` number of voxels per bin.
    ece:
        Expected calibration error (count-weighted mean ``|confidence -
        accuracy|`` over bins).
    ace:
        Average calibration error (unweighted mean ``|confidence - accuracy|``
        over non-empty bins). Unlike ``ece``, a sparsely populated bin counts
        as much as a densely populated one.
    """

    bin_edges: torch.Tensor
    bin_confidence: torch.Tensor
    bin_accuracy: torch.Tensor
    bin_counts: torch.Tensor
    ece: float
    ace: float


def reliability_curve(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
    n_bins: int = 15,
) -> ReliabilityCurve:
    """Compute top-1 reliability statistics with uniform binning.

    The confidence is the maximum predicted probability and the accuracy is
    whether the argmax matches the label. Useful both for reporting ECE/ACE and
    for drawing reliability diagrams (see
    :func:`fiducio.plots.reliability_diagram`).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _flatten_valid(probs, targets, mask, ignore_index)
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    if p.shape[0] == 0:
        zeros = torch.zeros(n_bins)
        return ReliabilityCurve(
            edges, zeros, zeros.clone(), zeros.clone(), float("nan"), float("nan")
        )

    confidence, prediction = p.max(dim=1)
    correct = (prediction == y).to(torch.float32)
    # Bin index in [0, n_bins - 1].
    idx = torch.bucketize(confidence, edges[1:-1].contiguous(), right=False)

    counts = torch.zeros(n_bins).scatter_add_(0, idx, torch.ones_like(confidence))
    sum_conf = torch.zeros(n_bins).scatter_add_(0, idx, confidence)
    sum_acc = torch.zeros(n_bins).scatter_add_(0, idx, correct)
    safe_counts = counts.clamp_min(1.0)
    bin_conf = sum_conf / safe_counts
    bin_acc = sum_acc / safe_counts
    total = float(confidence.shape[0])
    gap = (bin_conf - bin_acc).abs()
    ece = float(((counts / total) * gap).sum().item())
    nonempty = counts > 0
    ace = float(gap[nonempty].mean().item()) if bool(nonempty.any()) else float("nan")
    return ReliabilityCurve(edges, bin_conf, bin_acc, counts, ece, ace)


def expected_calibration_error(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
    n_bins: int = 15,
) -> float:
    """Top-1 expected calibration error (ECE) with uniform binning.

    The confidence is the maximum predicted probability and the accuracy is
    whether the argmax matches the label. Bins partition ``[0, 1]`` uniformly
    and each bin's gap is weighted by its share of voxels — see
    :func:`average_calibration_error` for the unweighted variant.

    ``n_bins`` defaults to 15. The paper reports ECE with ``n_bins=50`` (ACE with
    ``n_bins=15``) and averages metrics per image before pooling; pass
    ``n_bins=50`` and compute per case to follow that convention. Boundary-aware
    ECE is not included.
    """
    return reliability_curve(probs, targets, mask, ignore_index, n_bins).ece


def average_calibration_error(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
    n_bins: int = 15,
) -> float:
    """Top-1 average calibration error (ACE) with uniform binning.

    Uses the same uniform confidence bins as :func:`expected_calibration_error`,
    but averages the per-bin ``|confidence - accuracy|`` gap **unweighted**
    over non-empty bins instead of weighting each bin by its share of voxels.
    A confidence region visited by only a handful of voxels therefore counts as
    much as a densely populated one, which ECE would otherwise drown out.
    ``n_bins=15`` matches the paper's ACE; the paper computes ACE on the pooled
    test voxels rather than per image.
    """
    return reliability_curve(probs, targets, mask, ignore_index, n_bins).ace
