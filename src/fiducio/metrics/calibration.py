"""Calibration metrics for segmentation probabilities.

All metrics accept probabilities of shape ``(B, C, *spatial)`` (class axis at
dimension 1) and integer labels ``(B, *spatial)`` or floating class distributions
``(B, C, *spatial)``. Soft targets use ``mask``; ``ignore_index`` applies to hard
labels. They return Python floats (or, for
:func:`reliability_curve`, a :class:`ReliabilityCurve`).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import torch

from ..utils import safe_log, to_tensor, validate_predictions, validate_targets
from ..utils.tensors import prepare_targets


def _voxels(
    probs: Any,
    targets: Any,
    mask: Any | None,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Validate inputs and return ``(probs, labels, valid)`` in their spatial layout.

    Metrics reduce over the class axis before selecting valid voxels, so large
    volumes are never copied into ``(N, C)`` rows. Labels at invalid voxels are
    replaced by 0 so they can be used as gather indices; ``valid`` is ``None``
    when every voxel is valid.
    """
    p = to_tensor(probs).detach()
    if not p.is_floating_point():
        p = p.float()
    m = None if mask is None else to_tensor(mask, device=p.device).bool()
    c = validate_predictions(p, input_type="probs", mask=m, context="metrics")
    y = prepare_targets(p, targets)
    validate_targets(p, y, m, num_classes=c, ignore_index=ignore_index)
    if p.dtype not in (torch.float32, torch.float64):
        p = p.float()
    soft = y.shape == p.shape
    valid = torch.ones_like(p[:, 0], dtype=torch.bool) if soft else y != ignore_index
    if m is not None:
        valid &= m
    if bool(valid.all()):
        return p, y, None
    return p, y.masked_fill(~valid.unsqueeze(1) if soft else ~valid, 0), valid


def _select(x: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    return x.reshape(-1) if valid is None else x[valid]


def _true_class_probs(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return p.gather(1, y.unsqueeze(1)).squeeze(1)


def negative_log_likelihood(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
) -> float:
    """Mean negative log-likelihood (cross-entropy) over valid voxels.

    Zero probabilities are floored at ``1e-12`` by :func:`fiducio.utils.safe_log`,
    capping the NLL at ~27.63 instead of ``inf``.
    """
    p, y, valid = _voxels(probs, targets, mask, ignore_index)
    if y.shape == p.shape:
        loss = torch.zeros_like(_select(p[:, 0], valid), dtype=torch.float64)
        for c in range(p.shape[1]):
            loss -= _select(y[:, c], valid).double() * safe_log(_select(p[:, c], valid).double())
        return float(loss.mean().item())
    true_p = _select(_true_class_probs(p, y), valid).double()
    if true_p.numel() == 0:
        return float("nan")
    return float((-safe_log(true_p)).mean().item())


def brier_score(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
) -> float:
    """Mean sum of squared class residuals against labels or soft distributions.

    Against an MHR, this differs from the mean per-rater Brier by the rater
    disagreement term. Binary two-channel Brier is twice positive-channel MSE.
    Residuals accumulate in float64 to retain errors near perfect predictions.
    """
    p, y, valid = _voxels(probs, targets, mask, ignore_index)
    if valid is not None and not bool(valid.any()):
        return float("nan")
    # Direct residuals avoid cancellation near perfect predictions, one channel at a time.
    squared = torch.zeros_like(_select(p[:, 0], valid), dtype=torch.float64)
    for c in range(p.shape[1]):
        target = y[:, c] if y.shape == p.shape else y == c
        residual = _select(p[:, c], valid).double() - _select(target, valid).double()
        squared += residual.square()
    return float(squared.mean().item())


@dataclass
class ReliabilityCurve:
    """Per-bin top-1 or class-specific reliability statistics.

    Attributes
    ----------
    bin_edges:
        ``(n_bins + 1,)`` bin boundaries in ``[0, 1]``.
    bin_confidence:
        ``(n_bins,)`` mean predicted confidence per bin (0 for empty bins).
    bin_accuracy:
        ``(n_bins,)`` mean agreement or class proportion per bin (0 for empty bins).
    bin_counts:
        ``(n_bins,)`` int64 number of voxels per bin. Other statistics are
        float64; all tensors reside on the predictions' device.
    ece:
        Expected calibration error (count-weighted mean ``|confidence -
        accuracy|`` over bins).
    ace:
        Unweighted calibration error (unweighted mean ``|confidence - accuracy|``
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
    *,
    class_index: int | None = None,
) -> ReliabilityCurve:
    """Compute top-1 or class-specific reliability with uniform binning.

    The confidence is the maximum predicted probability and the accuracy is
    whether the argmax matches the label, or ``q[argmax(p)]`` for soft targets.
    With ``class_index=c``, compare ``p_c`` with ``1[y=c]`` or ``q_c``.
    ECE against MHR differs from mean per-rater ECE; retain the individual votes
    to compute the latter. Useful both for reporting ECE/ACE and
    for drawing reliability diagrams (see
    :func:`fiducio.plots.reliability_diagram`).
    """
    try:  # operator.index accepts numpy integers
        n_bins = -1 if isinstance(n_bins, bool) else operator.index(n_bins)
    except TypeError:
        n_bins = -1
    if n_bins < 1:
        raise ValueError("n_bins must be an integer >= 1")
    p, y, valid = _voxels(probs, targets, mask, ignore_index)
    # Construct the same float32 boundaries as before, then accumulate in double.
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device).double()
    if class_index is None:
        confidence, prediction = p.max(dim=1)
        observed = _true_class_probs(y, prediction) if y.shape == p.shape else prediction == y
    else:
        try:
            c = -1 if isinstance(class_index, bool) else operator.index(class_index)
        except TypeError:
            c = -1
        if not 0 <= c < p.shape[1]:
            raise ValueError(f"class_index must be an integer in [0, {p.shape[1] - 1}]")
        confidence = p[:, c]
        observed = y[:, c] if y.shape == p.shape else y == c
    correct = _select(observed, valid).double()
    confidence = _select(confidence, valid).double().contiguous()
    if confidence.numel() == 0:
        zeros = torch.zeros(n_bins, dtype=torch.float64, device=p.device)
        return ReliabilityCurve(
            edges, zeros, zeros.clone(), zeros.long(), float("nan"), float("nan")
        )
    # Bin index in [0, n_bins - 1].
    idx = torch.bucketize(confidence, edges[1:-1].contiguous(), out_int32=True)

    counts = torch.bincount(idx, minlength=n_bins)
    # Weighted sums need scatter_add_: deterministic on CPU, on CUDA only with
    # torch.use_deterministic_algorithms(True).
    sum_conf = torch.zeros(n_bins, dtype=torch.float64, device=p.device).scatter_add_(
        0, idx.long(), confidence
    )
    sum_acc = torch.zeros_like(sum_conf).scatter_add_(0, idx.long(), correct)
    safe_counts = counts.clamp_min(1)
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
    *,
    class_index: int | None = None,
) -> float:
    """Top-1 expected calibration error (ECE) with uniform binning.

    Pass ``class_index`` for class-specific calibration against hard or soft
    targets. This measures ECE against the supplied targets, not mean rater ECE.

    The confidence is the maximum predicted probability and the accuracy is
    whether the argmax matches the label. Bins partition ``[0, 1]`` uniformly
    and each bin's gap is weighted by its share of voxels — see
    :func:`unweighted_calibration_error` for the unweighted variant.

    ``n_bins`` defaults to 15. The paper reports ECE with ``n_bins=50`` (ACE with
    ``n_bins=15``) and averages metrics per image before pooling; pass
    ``n_bins=50`` and compute per case to follow that convention. Boundary-aware
    ECE is not included.
    """
    return reliability_curve(
        probs, targets, mask, ignore_index, n_bins, class_index=class_index
    ).ece


def unweighted_calibration_error(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
    n_bins: int = 15,
    *,
    class_index: int | None = None,
) -> float:
    """Top-1 unweighted calibration error (ACE) with uniform binning.

    ``class_index`` selects a class instead of the default top-1 confidence.

    The unweighted mean of the per-bin ``|confidence - accuracy|`` gap over
    non-empty uniform bins. This is **not** the adaptive calibration error of
    other work, which uses data-dependent (equal-mass) bins.

    Uses the same uniform confidence bins as :func:`expected_calibration_error`,
    but weights each non-empty bin equally instead of by its share of voxels; a
    confidence region visited by only a handful of voxels therefore counts as
    much as a densely populated one. ``n_bins=15`` matches the paper's ACE; the
    paper computes ACE on the pooled test voxels rather than per image. Kept as
    ``average_calibration_error`` for backwards compatibility.
    """
    return reliability_curve(
        probs, targets, mask, ignore_index, n_bins, class_index=class_index
    ).ace


average_calibration_error = unweighted_calibration_error
