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

from ..utils import flatten_valid, safe_log, to_tensor, validate_predictions, validate_targets
from ..utils.tensors import integer_targets

_EPS = 1e-12


def _flatten_valid(
    probs: Any,
    targets: Any,
    mask: Any | None,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = to_tensor(probs, dtype=torch.float32).detach()
    c = validate_predictions(p, input_type="probs")
    y = integer_targets(targets, device=p.device)
    m = None if mask is None else to_tensor(mask, device=p.device).bool()
    validate_targets(p, y, m, num_classes=c, ignore_index=ignore_index)
    return flatten_valid(p, y, m, ignore_index)


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
        ``(n_bins,)`` int64 number of voxels per bin. Other statistics are
        float64; all tensors reside on the predictions' device.
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
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be an integer >= 1")
    p, y = _flatten_valid(probs, targets, mask, ignore_index)
    # Construct the same float32 boundaries as before, then accumulate in double.
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device).double()
    if p.shape[0] == 0:
        zeros = torch.zeros(n_bins, dtype=torch.float64, device=p.device)
        return ReliabilityCurve(
            edges, zeros, zeros.clone(), zeros.long(), float("nan"), float("nan")
        )

    confidence, prediction = p.max(dim=1)
    confidence = confidence.double()
    correct = (prediction == y).double()
    # Bin index in [0, n_bins - 1].
    idx = torch.bucketize(confidence, edges[1:-1].contiguous(), right=False)

    counts = torch.zeros(n_bins, dtype=torch.int64, device=p.device).scatter_add_(
        0, idx, torch.ones_like(idx)
    )
    sum_conf = torch.zeros(n_bins, dtype=torch.float64, device=p.device).scatter_add_(0, idx, confidence)
    sum_acc = torch.zeros_like(sum_conf).scatter_add_(0, idx, correct)
    safe_counts = counts.clamp_min(1.0)
    bin_conf = sum_conf / safe_counts
    bin_acc = sum_acc / safe_counts
    total = float(confidence.shape[0])
    gap = (bin_conf - bin_acc).abs()
    ece = float(((counts.double() / total) * gap).sum().item())
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
