"""Reliability diagnostics of a segmenter on a calibration set, before calibrating.

:func:`diagnose` turns logits and labels into a :class:`ReliabilityReport`:
class balance, global and per-class calibration, and images or classes where the
model fails (Dice close to zero). The report describes the calibration set with
its labels; it does not predict failures on new images.

Everything is computed from per-image sufficient statistics, so confidence
intervals come from a bootstrap over images (voxels of one image are strongly
correlated, so a per-voxel interval would be far too narrow).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .metrics.calibration import (
    ReliabilityCurve,
    brier_score,
    negative_log_likelihood,
    reliability_curve,
)
from .plots import _draw_reliability, _require_matplotlib
from .utils import to_tensor, validate_predictions, validate_targets
from .utils.tensors import prepare_targets, validation_positions

__all__ = ["BoundaryStats", "ClassStats", "FailureStats", "ReliabilityReport", "diagnose"]

_FAIL_RATE_ALERT = 0.1  # a class failing on this share of its images is flagged
_SATURATION = 0.999  # confidence above which a voxel counts as saturated
NAN_CI = (float("nan"), float("nan"))


@dataclass
class ClassStats:
    """Per-class balance, calibration and Dice summary (NaN when undefined).

    The ``*_ci`` fields are 95% percentile bootstrap intervals over images (not
    voxels). They are NaN when fewer than two resamples or too few images are
    available for that quantity.
    """

    name: str
    label_voxels: int | float
    label_fraction: float
    pred_voxels: int
    pred_fraction: float
    dice: float  # pooled over all images
    class_ece: float  # p_k against q_k (or 1[label == k])
    routed_ece: float  # top-1 ECE on voxels whose argmax is this class
    n_present: int  # images where the class is in the labels
    n_failed: int  # of those, images with Dice below the threshold
    n_spurious: int  # images where the class is predicted but absent from the labels
    dice_median: float  # Dice over the images where the class is present
    dice_p10: float
    dice_min: float
    class_ece_ci: tuple[float, float] = NAN_CI
    routed_ece_ci: tuple[float, float] = NAN_CI
    dice_ci: tuple[float, float] = NAN_CI


@dataclass
class BoundaryStats:
    """Top-1 calibration in the first and last uniform confidence bins.

    ``low_bin_mass``/``high_bin_mass`` are the shares of valid voxels whose
    top-1 confidence falls in ``[0, 1/n_bins]`` and in
    ``[(n_bins-1)/n_bins, 1]`` (the first bin includes 0, the last includes 1).
    ``*_bin_gap`` is their mean confidence minus mean accuracy, so a positive
    value is over-confidence. ``ece_contribution`` is exactly the part of the
    report's ``ece`` that these two bins contribute, in ``[0, 1]``; it is
    computed once when ``n_bins == 1``.
    """

    low_bin_mass: float
    low_bin_gap: float
    high_bin_mass: float
    high_bin_gap: float
    ece_contribution: float


@dataclass
class FailureStats:
    """Images whose mean Dice falls below the threshold, and their effect on calibration."""

    threshold: float
    n_failed: int
    ece_all: float
    ece_without_failed: float  # NaN when every image failed
    confidence_failed: float  # mean confidence on failed images (NaN if none)
    confidence_rest: float
    # (image id, mean Dice, class with the lowest Dice, that Dice), lowest first
    worst: list[tuple[str, float, str, float]] = field(default_factory=list)


@dataclass(repr=False)
class ReliabilityReport:
    """Result of :func:`diagnose`. ``print(report)`` shows :meth:`summary`.

    ``metrics`` maps ``ece``, ``nll``, ``brier`` and ``gap`` (mean confidence minus
    accuracy) to ``(value, ci_low, ci_high)``; the interval is NaN when it could not
    be computed. ``image_dice`` is the ``(n_images, n_classes)`` Dice matrix, NaN where
    the class is absent from both labels and prediction, or everywhere for soft
    targets (hard-label Dice is undefined there). ``boundary`` reports the first and
    last confidence bins. ``soft_targets`` records which target kind was diagnosed.
    """

    n_images: int
    n_valid: int
    n_bins: int
    saturated_fraction: float
    boundary: BoundaryStats
    metrics: dict[str, tuple[float, float, float]]
    classes: list[ClassStats]
    failures: FailureStats
    curve: ReliabilityCurve
    image_dice: torch.Tensor
    warnings: list[str]
    soft_targets: bool = False

    @property
    def bias(self) -> str:
        """``overconfident``/``underconfident`` if the interval of the gap excludes 0."""
        _, lo, hi = self.metrics["gap"]
        if math.isnan(lo):
            return "unknown (no confidence interval)"
        if lo > 0:
            return "overconfident"
        if hi < 0:
            return "underconfident"
        return "no significant bias"

    def summary(self) -> str:
        """Plain-text report for a terminal or a notebook cell."""
        m = self.metrics

        def est(key: str) -> str:
            return _fmt_interval(*m[key])

        lines = [
            f"Reliability report: {self.n_images} images, {self.n_valid} valid voxels, "
            f"{len(self.classes)} classes",
            f"  ECE   {est('ece')}",
            f"  NLL   {est('nll')}",
            f"  Brier {est('brier')}",
            f"  Bias  {self.bias} (confidence - accuracy = {est('gap')})",
            f"  Saturated voxels (confidence > {_SATURATION}): {self.saturated_fraction:.1%}",
            f"  First/last bin mass {self.boundary.low_bin_mass:.1%}/"
            f"{self.boundary.high_bin_mass:.1%}, gap "
            f"{_f(self.boundary.low_bin_gap)}/{_f(self.boundary.high_bin_gap)}, "
            f"ECE contribution {_f(self.boundary.ece_contribution)}",
            "",
        ]
        rows = [["class", "label %", "pred %", "Dice", "class ECE", "routed vox", "routed ECE"]] + [
            [
                c.name,
                f"{100 * c.label_fraction:.2f}",
                f"{100 * c.pred_fraction:.2f}",
                _f(c.dice),
                _f(c.class_ece),
                str(c.pred_voxels),
                _f(c.routed_ece),
            ]
            for c in self.classes
        ]
        lines += _table(rows) + [""]
        lines.append("  Per-class 95% intervals (bootstrap over images):")
        rows = [["class", "class ECE", "routed ECE", "Dice"]] + [
            [
                c.name,
                _fmt_interval(c.class_ece, *c.class_ece_ci),
                _fmt_interval(c.routed_ece, *c.routed_ece_ci),
                _fmt_interval(c.dice, *c.dice_ci),
            ]
            for c in self.classes
        ]
        lines += _table(rows) + [""]
        f = self.failures
        lines.append(f"Failed images (mean Dice < {f.threshold:g}): {f.n_failed}/{self.n_images}")
        lines.append(
            f"  ECE with all images {_f(f.ece_all)}, without the failed ones "
            f"{_f(f.ece_without_failed)}; mean confidence {_f(f.confidence_failed)} "
            f"on failed vs {_f(f.confidence_rest)} on the rest"
        )
        lines.append(
            f"  Per class, a failure is Dice < {f.threshold:g} on an image where it is present:"
        )
        rows = [["class", "present", "failed", "spurious", "Dice median", "p10", "min"]] + [
            [
                c.name,
                str(c.n_present),
                str(c.n_failed),
                str(c.n_spurious),
                _f(c.dice_median),
                _f(c.dice_p10),
                _f(c.dice_min),
            ]
            for c in self.classes
        ]
        lines += _table(rows)
        if f.worst:
            lines.append("  Lowest-Dice images:")
            lines += [
                f"    {name}: mean Dice {_f(d)}, worst class {cls} ({_f(cd)})"
                for name, d, cls, cd in f.worst
            ]
        lines += ["", "Warnings" if self.warnings else "No warnings"]
        lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)

    __str__ = summary
    __repr__ = summary

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dictionary (NaN becomes ``None``)."""
        out: dict[str, Any] = _clean(dataclasses.asdict(self) | {"bias": self.bias})
        return out

    def plot(self) -> Any:
        """Reliability, confidence histogram, per-class ECE and per-image Dice (2x2)."""
        plt = _require_matplotlib()
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        _draw_reliability(
            axes[0, 0], self.curve, class_index=None, title="Reliability (top-1)", metric="ece"
        )

        edges = self.curve.bin_edges.cpu().numpy()
        counts = self.curve.bin_counts.cpu().numpy()
        ax = axes[0, 1]
        ax.bar(
            (edges[:-1] + edges[1:]) / 2, counts / max(counts.sum(), 1), width=edges[1] - edges[0]
        )
        ax.set_yscale("log")
        ax.set_xlabel("confidence")
        ax.set_ylabel("share of voxels")
        ax.set_title("Confidence histogram")

        names = [f"{c.name}\n{100 * c.label_fraction:.1f}%" for c in self.classes]
        ax = axes[1, 0]
        ax.bar(
            range(len(names)),
            [0.0 if math.isnan(c.class_ece) else c.class_ece for c in self.classes],
        )
        ax.set_xticks(range(len(names)), names)
        ax.set_ylabel("class ECE")
        ax.set_title("Per-class ECE (share of labels below the name)")

        ax = axes[1, 1]
        dice = self.image_dice.cpu().numpy()
        for c in range(dice.shape[1]):
            # Deterministic jitter so that points of one class do not overlap.
            jitter = ((torch.arange(dice.shape[0]) * 0.618) % 1 - 0.5).numpy() * 0.4
            ax.scatter(c + jitter, dice[:, c], s=8, alpha=0.6)
        ax.axhline(self.failures.threshold, color="red", linestyle="--", label="failure threshold")
        ax.set_xticks(range(len(names)), [c.name for c in self.classes])
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Dice per image")
        ax.set_title("Per-image Dice")
        ax.legend(loc="lower right", fontsize="small")
        fig.tight_layout()
        return fig


def _f(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.4f}"


def _fmt_interval(value: float, lo: float, hi: float) -> str:
    if math.isnan(lo):
        return _f(value)
    return f"{_f(value)} [{_f(lo)}, {_f(hi)}]"


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return [
        "  "
        + "  ".join(
            cell.ljust(w) if i == 0 else cell.rjust(w)
            for i, (cell, w) in enumerate(zip(r, widths, strict=True))
        )
        for r in rows
    ]


def _clean(obj: Any) -> Any:
    """Make a ``dataclasses.asdict`` result strict-JSON: tensors to lists, NaN to None."""
    if torch.is_tensor(obj):
        return _clean(obj.tolist())
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _bin_stats(curve: ReliabilityCurve) -> torch.Tensor:
    """``(n_bins, 3)`` sufficient statistics: voxel count, sum of confidence, sum of agreement."""
    n = curve.bin_counts.double()
    return torch.stack([n, curve.bin_confidence * n, curve.bin_accuracy * n], dim=-1)


def _ece(stats: torch.Tensor) -> torch.Tensor:
    """ECE from ``(..., n_bins, 3)`` statistics: ``sum_b |sum_conf - sum_acc| / N``."""
    return (stats[..., 1] - stats[..., 2]).abs().sum(-1) / stats[..., 0].sum(-1)


def _gap(stats: torch.Tensor) -> torch.Tensor:
    return (stats[..., 1].sum(-1) - stats[..., 2].sum(-1)) / stats[..., 0].sum(-1)


def _ci(values: torch.Tensor) -> tuple[float, float]:
    """95% percentile interval of the bootstrap values (NaN with fewer than two)."""
    values = values[~torch.isnan(values)]
    if values.numel() < 2:
        return float("nan"), float("nan")
    q = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=values.dtype))
    return float(q[0]), float(q[1])


def _dice_summary(values: torch.Tensor) -> tuple[float, float, float]:
    """Median, 10th percentile and minimum of ``values`` (NaN if empty)."""
    if values.numel() == 0:
        return (float("nan"),) * 3
    q = torch.quantile(values, torch.tensor([0.5, 0.1], dtype=values.dtype))
    return float(q[0]), float(q[1]), float(values.min())


def _boundary_stats(pooled: torch.Tensor, total: float) -> BoundaryStats:
    """First/last-bin statistics from ``(n_bins, 3)`` pooled top-1 statistics."""
    counts, conf_sum, acc_sum = pooled[:, 0], pooled[:, 1], pooled[:, 2]

    def gap(i: int) -> float:
        return float((conf_sum[i] - acc_sum[i]) / counts[i]) if counts[i] > 0 else float("nan")

    indices = [0] if pooled.shape[0] == 1 else [0, -1]
    contribution = float(sum((conf_sum[i] - acc_sum[i]).abs() for i in indices) / total)
    return BoundaryStats(
        low_bin_mass=float(counts[0] / total),
        low_bin_gap=gap(0),
        high_bin_mass=float(counts[-1] / total),
        high_bin_gap=gap(-1),
        ece_contribution=contribution,
    )


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors in a model input (tensor, list, tuple or dict) to ``device``."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move_to_device(v, device) for v in value)
    return value


@dataclass
class _Accumulator:
    """Per-image sufficient statistics, filled batch by batch.

    Holds only ``O(n_images)`` summaries (per-bin counts and sums, confusion
    matrices), never the images themselves, so a DataLoader path can run on data
    that does not fit in memory. The bootstrap resamples these summaries.
    """

    num_classes: int
    n_bins: int
    input_type: str
    ignore_index: int
    top: list[torch.Tensor] = field(default_factory=list)
    per_class: list[torch.Tensor] = field(default_factory=list)
    routed: list[torch.Tensor] = field(default_factory=list)
    confusion: list[torch.Tensor] = field(default_factory=list)
    nll_sum: list[float] = field(default_factory=list)
    brier_sum: list[float] = field(default_factory=list)
    n_valid: list[int] = field(default_factory=list)
    n_saturated: list[int] = field(default_factory=list)
    label_counts: list[torch.Tensor] = field(default_factory=list)
    soft: bool | None = None

    def add(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None) -> None:
        """Accumulate one validated batch ``(B, C, *spatial)`` / ``(B, *spatial)``."""
        soft = y.shape == x.shape
        if self.soft is None:
            self.soft = soft
        elif self.soft != soft:
            raise ValueError("hard and soft targets must not be mixed across batches")
        for i in range(x.shape[0]):
            self._add_image(
                x[i : i + 1],
                y[i : i + 1],
                None if mask is None else mask[i : i + 1],
                soft,
            )

    def _add_image(
        self, xi: torch.Tensor, yi: torch.Tensor, mi: torch.Tensor | None, soft: bool
    ) -> None:
        if self.input_type == "probs":
            p = xi
        else:
            p = torch.softmax(xi.double() if xi.dtype == torch.float64 else xi.float(), dim=1)
        valid = validation_positions(xi, yi, mi, self.ignore_index)
        if valid is None:
            valid = torch.ones(xi.shape[0], *xi.shape[2:], dtype=torch.bool, device=xi.device)
        pred = p.argmax(dim=1)
        self.top.append(_bin_stats(reliability_curve(p, yi, mi, self.ignore_index, self.n_bins)))
        self.per_class.append(
            torch.stack(
                [
                    _bin_stats(
                        reliability_curve(
                            p, yi, mi, self.ignore_index, self.n_bins, class_index=c
                        )
                    )
                    for c in range(self.num_classes)
                ]
            )
        )
        self.routed.append(
            torch.stack(
                [
                    _bin_stats(
                        reliability_curve(
                            p, yi, valid & (pred == c), self.ignore_index, self.n_bins
                        )
                    )
                    for c in range(self.num_classes)
                ]
            )
        )
        n = int(valid.sum())
        self.n_valid.append(n)
        self.nll_sum.append(
            n * negative_log_likelihood(p, yi, mi, self.ignore_index) if n > 0 else 0.0
        )
        self.brier_sum.append(n * brier_score(p, yi, mi, self.ignore_index) if n > 0 else 0.0)
        self.n_saturated.append(int((p.max(dim=1).values[valid] > _SATURATION).sum()))
        if soft:
            # Expected class counts: a soft target may split a voxel across classes.
            self.label_counts.append(yi.movedim(1, -1)[valid].sum(0))
        else:
            joint = yi[valid] * self.num_classes + pred[valid]
            self.confusion.append(
                torch.bincount(joint, minlength=self.num_classes**2).view(self.num_classes, -1)
            )

    @property
    def n_images(self) -> int:
        return len(self.n_valid)

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Stacked float64 CPU statistics; the confusion is ``None`` for soft targets."""
        top = torch.stack(self.top).cpu()
        per_class = torch.stack(self.per_class).cpu()
        routed = torch.stack(self.routed).cpu()
        confusion = None if self.soft else torch.stack(self.confusion).cpu().double()
        return top, per_class, routed, confusion


def _string_sequence(value: Any, name: str) -> list[str]:
    """Materialise a sequence of strings without exploding a bare string."""
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of strings, not a single string")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if not all(isinstance(item, str) for item in items):
        raise ValueError(f"{name} must contain only strings")
    return items


def diagnose(
    logits: Any,
    labels: Any,
    *,
    input_type: str = "logits",
    mask: Any | None = None,
    ignore_index: int = -100,
    device: Any | None = None,
    n_bins: int = 15,
    class_names: list[str] | None = None,
    image_ids: list[str] | None = None,
    background_class: int | None = None,
    dice_failure_threshold: float = 0.1,
    n_worst: int = 5,
    n_boot: int = 200,
    seed: int = 0,
    min_images: int = 20,
    min_class_fraction: float = 0.01,
    min_class_voxels: int = 1000,
) -> ReliabilityReport:
    """Diagnose the reliability of a segmenter on a calibration set.

    Parameters
    ----------
    logits:
        ``(B, C, *spatial)`` logits (or probabilities with ``input_type="probs"``);
        ``B`` is the number of images or cases. Alternatively a ``torch.nn.Module``:
        ``labels`` is then a ``DataLoader`` and each batch is ``(inputs, labels)`` or
        ``(inputs, labels, mask)``. ``inputs`` are whatever the model accepts (moved
        to ``device``); the model output must have shape ``(B, C, *spatial)``. The
        model runs under ``torch.no_grad()`` and its ``training`` flag is unchanged
        (call ``model.eval()`` yourself if you want inference-mode dropout/batchnorm).
        Only per-image statistics are retained, so the loader's images are never
        materialised together.
    labels:
        ``(B, *spatial)`` integer labels, or ``(B, C, *spatial)`` floating class
        distributions (soft targets / MHR, same convention as the metrics). With a
        model, the DataLoader instead of labels.
    mask, ignore_index:
        Voxel selection, as in the metrics. For soft targets only ``mask`` applies.
        With the DataLoader path, pass per-batch masks in the batch instead.
    device:
        Device for the DataLoader path's batches (default: the device of the model's
        first parameter, otherwise CPU). Ignored for tensor inputs.
    class_names, image_ids:
        Optional display names (default: indices).
    background_class:
        Class excluded from the per-image mean Dice, whose value near 1 would hide a
        failure on the classes that matter. ``None`` excludes nothing.
    dice_failure_threshold:
        A class fails on an image when it is in the labels and its Dice is below this.
        An image fails when its mean Dice over the classes present is below it.
    n_worst:
        Number of lowest-Dice images listed.
    n_boot, seed:
        Bootstrap resamples over images for the 95% intervals (``0`` disables them;
        an interval needs at least two resamples, so ``1`` yields NaN) and the seed
        of the resampling. Per-class ECE, routed ECE and Dice intervals reuse the same
        resampling.
    min_images, min_class_fraction, min_class_voxels:
        Thresholds of the warnings. They are heuristics, not statistical guarantees.

    Notes
    -----
    Soft targets. Per-class ECE compares ``p_c`` with the target mass ``q_c`` and the
    top-1/routed ECE compares the top-1 confidence with ``q[argmax(p)]``; NLL and
    Brier follow :mod:`fiducio.metrics.calibration`. Hard-label Dice and the failure
    statistics are undefined: every Dice field, the image Dice matrix and the
    per-image failure counts are NaN/0, and ``image_dice`` is all NaN. Class balance
    still counts the expected label mass ``sum(q_c)``.

    Boundary bins. ``report.boundary`` describes the first ``[0, 1/n_bins]`` and last
    ``[(n_bins-1)/n_bins, 1]`` top-1 confidence bins: their voxel mass, their mean
    confidence minus accuracy, and their exact contribution to ``ece``. It is
    additive reporting only; ``ece`` keeps its definition.

    The ECE interval is a percentile bootstrap. ECE has a finite-sample noise floor that
    resampling raises further, so for a well-calibrated model the interval can sit
    slightly above the estimate (it missed it in 15 of 100 simulated calibrated models
    of 24 images). Read it as an order of magnitude. The *bias* line uses the
    voxel-weighted confidence-accuracy gap, which has no such noise floor, but its
    intervals are slightly narrow: it flagged 9 of those 100 calibrated models, against
    5 expected (``reproducibility/diagnose_interval_check.py`` reruns the simulation).

    Cost is ``2C + 1`` reliability curves per image (global, per-class and routed),
    plus a few passes for NLL, Brier and the confusion matrix. With a DataLoader the
    model forward pass dominates; the accumulator keeps ``O(n_images)`` summaries
    (``O(n_images * C * n_bins)``), not the voxels. Classes absent from both labels and
    prediction have an undefined Dice and are never counted as failures.
    """
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, numbers.Integral):
        raise TypeError(f"ignore_index must be an int, got {ignore_index!r}")
    ignore_index = int(ignore_index)
    if input_type not in ("logits", "probs"):
        raise ValueError(f"input_type must be 'logits' or 'probs', got {input_type!r}")
    for name, value in (
        ("n_boot", n_boot),
        ("n_worst", n_worst),
        ("min_images", min_images),
        ("min_class_voxels", min_class_voxels),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if (
        isinstance(min_class_fraction, bool)
        or not isinstance(min_class_fraction, numbers.Real)
        or not 0.0 <= min_class_fraction <= 1.0
    ):
        raise ValueError("min_class_fraction must be a number in [0, 1]")
    if (
        isinstance(dice_failure_threshold, bool)
        or not isinstance(dice_failure_threshold, numbers.Real)
        or not 0.0 <= dice_failure_threshold <= 1.0
    ):
        raise ValueError("dice_failure_threshold must be a number in [0, 1]")

    acc: _Accumulator | None = None
    if isinstance(logits, torch.nn.Module):
        if labels is None:
            raise ValueError("labels must be a DataLoader when logits is a torch.nn.Module")
        if isinstance(labels, (torch.Tensor, np.ndarray)):
            raise ValueError(
                "labels must be a DataLoader when logits is a torch.nn.Module, not an array"
            )
        if mask is not None:
            raise ValueError("with a model and DataLoader, pass each mask in the batch")
        model = logits
        if device is None:
            first_param = next(iter(model.parameters()), None)
            device = first_param.device if first_param is not None else torch.device("cpu")
        batch_device = torch.device(device)
        for batch in labels:
            try:
                items = list(batch)
            except TypeError as exc:
                raise ValueError(
                    "each loader batch must be (inputs, labels) or (inputs, labels, mask)"
                ) from exc
            if not 2 <= len(items) <= 3:
                raise ValueError(
                    "each loader batch must be (inputs, labels) or (inputs, labels, mask)"
                )
            inputs = _move_to_device(items[0], batch_device)
            target = _move_to_device(items[1], batch_device)
            batch_mask = _move_to_device(items[2], batch_device) if len(items) == 3 else None
            with torch.no_grad():
                output = model(inputs)
            if not torch.is_tensor(output):
                raise TypeError("the model must return a tensor of shape (B, C, *spatial)")
            xb = output.detach()
            if not xb.is_floating_point():
                xb = xb.float()
            if xb.ndim < 3:
                raise ValueError(
                    "the model output must have shape (B, C, *spatial) with at least one "
                    f"spatial axis; got {tuple(xb.shape)}"
                )
            mb = None if batch_mask is None else to_tensor(batch_mask, device=xb.device).bool()
            yb = prepare_targets(xb, target)
            positions = validation_positions(xb, yb, mb, ignore_index)
            if acc is None:
                num_classes = validate_predictions(xb, input_type=input_type, mask=positions)
                acc = _Accumulator(num_classes, n_bins, input_type, ignore_index)
            else:
                validate_predictions(
                    xb,
                    input_type=input_type,
                    mask=positions,
                    expected_num_classes=acc.num_classes,
                )
            validate_targets(xb, yb, mb, num_classes=acc.num_classes, ignore_index=ignore_index)
            acc.add(xb, yb, mb)
        if acc is None or acc.n_images == 0:
            raise ValueError("at least one image is required")
    else:
        x = to_tensor(logits).detach()
        if not x.is_floating_point():
            x = x.float()
        m = None if mask is None else to_tensor(mask, device=x.device).bool()
        if x.ndim < 3:
            raise ValueError(
                f"logits must have shape (B, C, *spatial) with at least one spatial axis; got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            raise ValueError("at least one image is required")
        y = prepare_targets(x, labels)
        num_classes = validate_predictions(
            x, input_type=input_type, mask=validation_positions(x, y, m, ignore_index)
        )
        validate_targets(x, y, m, num_classes=num_classes, ignore_index=ignore_index)
        acc = _Accumulator(num_classes, n_bins, input_type, ignore_index)
        acc.add(x, y, m)

    num_classes, n_images = acc.num_classes, acc.n_images
    names = (
        _string_sequence(class_names, "class_names")
        if class_names is not None
        else [str(c) for c in range(num_classes)]
    )
    ids = (
        _string_sequence(image_ids, "image_ids")
        if image_ids is not None
        else [str(i) for i in range(n_images)]
    )
    if len(names) != num_classes or len(ids) != n_images:
        raise ValueError("class_names must have one name per class and image_ids one per image")
    if background_class is not None:
        if isinstance(background_class, bool) or not isinstance(background_class, numbers.Integral):
            raise ValueError(f"background_class must be an integer in [0, {num_classes - 1}]")
        background_class = int(background_class)
        if not 0 <= background_class < num_classes:
            raise ValueError(f"background_class must lie in [0, {num_classes - 1}]")

    # Everything below is small: the accumulator holds O(n_images) summaries.
    top_s, class_s, routed_s, conf = acc.stacked()
    nll_t = torch.tensor(acc.nll_sum, dtype=torch.float64)
    brier_t = torch.tensor(acc.brier_sum, dtype=torch.float64)
    nvox = torch.tensor(acc.n_valid, dtype=torch.float64)
    soft = bool(acc.soft)
    total_valid = float(nvox.sum())
    if total_valid == 0:
        raise ValueError("no valid voxels: every label is ignored or masked out")

    # --- global metrics and bootstrap over images -----------------------------------------
    pooled = top_s.sum(0)
    point = {
        "ece": float(_ece(pooled)),
        "nll": float(nll_t.sum() / nvox.sum()),
        "brier": float(brier_t.sum() / nvox.sum()),
        "gap": float(_gap(pooled)),
    }
    intervals = {k: NAN_CI for k in point}
    class_ece_ci: list[tuple[float, float]] = [NAN_CI] * num_classes
    routed_ece_ci: list[tuple[float, float]] = [NAN_CI] * num_classes
    boot_w: torch.Tensor | None = None
    if n_boot > 0 and n_images >= 2:
        g = torch.Generator().manual_seed(seed)
        draws = torch.randint(n_images, (n_boot, n_images), generator=g)
        w = torch.zeros(n_boot, n_images, dtype=torch.float64).scatter_add_(
            1, draws, torch.ones_like(draws, dtype=torch.float64)
        )
        boot_top = torch.einsum("rb,bkj->rkj", w, top_s)
        boot_n = w @ nvox
        intervals = {
            "ece": _ci(_ece(boot_top)),
            "nll": _ci((w @ nll_t) / boot_n),
            "brier": _ci((w @ brier_t) / boot_n),
            "gap": _ci(_gap(boot_top)),
        }
        boot_class = torch.einsum("rb,bckj->rckj", w, class_s)
        boot_routed = torch.einsum("rb,bckj->rckj", w, routed_s)
        class_ece_ci = [_ci(_ece(boot_class[:, c])) for c in range(num_classes)]
        routed_images = (routed_s[..., 0] > 0).sum(dim=(0, 2))  # (C,) images with routed voxels
        routed_ece_ci = [
            _ci(_ece(boot_routed[:, c])) if int(routed_images[c]) >= 2 else NAN_CI
            for c in range(num_classes)
        ]
        boot_w = w
    metrics = {k: (point[k], *intervals[k]) for k in point}

    # --- Dice, failures -------------------------------------------------------------------
    if soft:
        # Hard-label Dice is undefined for class distributions; do not invent a soft Dice.
        dice = torch.full((n_images, num_classes), float("nan"))
        present = torch.zeros((n_images, num_classes), dtype=torch.bool)
        failed_cls, spurious, considered = present.clone(), present.clone(), present.clone()
        mean_dice = torch.full((n_images,), float("nan"))
        failed_img = torch.zeros(n_images, dtype=torch.bool)
        pooled_dice = torch.full((num_classes,), float("nan"))
        label_totals = torch.stack(acc.label_counts).sum(0)  # (C,) expected mass
        pred_totals = routed_s[..., 0].sum(dim=(0, 2))  # (C,) argmax counts
        tp = lab = prd = None
    else:
        assert conf is not None
        tp = conf.diagonal(dim1=1, dim2=2)  # (B, C)
        lab, prd = conf.sum(2), conf.sum(1)
        dice = 2 * tp / (lab + prd)  # NaN where the class is in neither labels nor prediction
        present = lab > 0
        failed_cls = present & (dice < dice_failure_threshold)
        spurious = (lab == 0) & (prd > 0)
        considered = present.clone()
        if background_class is not None:
            considered[:, background_class] = False
        mean_dice = dice.masked_fill(~considered, float("nan")).nanmean(dim=1)  # (B,)
        failed_img = mean_dice < dice_failure_threshold
        pooled_conf = conf.sum(0)
        pooled_dice = 2 * pooled_conf.diagonal() / (pooled_conf.sum(1) + pooled_conf.sum(0))

    dice_ci: list[tuple[float, float]] = [NAN_CI] * num_classes
    if boot_w is not None and tp is not None and lab is not None and prd is not None:
        for c in range(num_classes):
            if int(present[:, c].sum()) < 2:
                continue
            boot_dice = 2 * (boot_w @ tp[:, c]) / (boot_w @ (lab[:, c] + prd[:, c]))
            dice_ci[c] = _ci(boot_dice)

    routed_pool, class_pool = routed_s.sum(0), class_s.sum(0)
    classes = []
    for c in range(num_classes):
        class_ece = float(_ece(class_pool[c]))
        routed_ece = float(_ece(routed_pool[c]))
        if soft:
            label_vox: int | float = float(label_totals[c])
            pred_vox = int(pred_totals[c])
            dice_c = median = p10 = lowest = float("nan")
            n_present = n_failed = n_spurious = 0
        else:
            assert conf is not None
            d_c = dice[:, c][present[:, c]]
            median, p10, lowest = _dice_summary(d_c)
            label_vox, pred_vox = int(pooled_conf[c].sum()), int(pooled_conf[:, c].sum())
            dice_c = float(pooled_dice[c])
            n_present = int(present[:, c].sum())
            n_failed = int(failed_cls[:, c].sum())
            n_spurious = int(spurious[:, c].sum())
        classes.append(
            ClassStats(
                name=names[c],
                label_voxels=label_vox,
                label_fraction=label_vox / total_valid,
                pred_voxels=pred_vox,
                pred_fraction=pred_vox / total_valid,
                dice=dice_c,
                class_ece=class_ece,
                routed_ece=routed_ece,
                n_present=n_present,
                n_failed=n_failed,
                n_spurious=n_spurious,
                dice_median=median,
                dice_p10=p10,
                dice_min=lowest,
                class_ece_ci=class_ece_ci[c],
                routed_ece_ci=routed_ece_ci[c],
                dice_ci=dice_ci[c],
            )
        )

    if soft:
        failures = FailureStats(
            threshold=dice_failure_threshold,
            n_failed=0,
            ece_all=point["ece"],
            ece_without_failed=float("nan"),
            confidence_failed=float("nan"),
            confidence_rest=float("nan"),
        )
    else:
        assert conf is not None

        def mean_conf(sel: torch.Tensor) -> float:
            s = top_s[sel].sum(0)
            return float(s[:, 1].sum() / s[:, 0].sum())

        rest = ~failed_img
        ranked = torch.argsort(mean_dice.nan_to_num(float("inf")))[: min(n_worst, n_images)]
        worst = []
        for i in (
            int(j) for j in ranked if float(mean_dice[j]) < 1.0
        ):  # NaN and perfect images excluded
            row = dice[i].masked_fill(~considered[i], float("inf"))
            c = int(row.argmin())
            worst.append((ids[i], float(mean_dice[i]), names[c], float(dice[i, c])))
        failures = FailureStats(
            threshold=dice_failure_threshold,
            n_failed=int(failed_img.sum()),
            ece_all=point["ece"],
            ece_without_failed=float(_ece(top_s[rest].sum(0)))
            if rest.any()
            else float("nan"),
            confidence_failed=mean_conf(failed_img) if failed_img.any() else float("nan"),
            confidence_rest=mean_conf(rest) if rest.any() else float("nan"),
            worst=worst,
        )

    # --- curve of the pooled top-1 statistics ---------------------------------------------
    counts = pooled[:, 0]
    bin_conf, bin_acc = pooled[:, 1] / counts.clamp_min(1), pooled[:, 2] / counts.clamp_min(1)
    nonempty = counts > 0
    curve = ReliabilityCurve(
        torch.linspace(0.0, 1.0, n_bins + 1).double(),
        bin_conf,
        bin_acc,
        counts.long(),
        point["ece"],
        float((bin_conf - bin_acc).abs()[nonempty].mean()),
    )

    report = ReliabilityReport(
        n_images=n_images,
        n_valid=int(total_valid),
        n_bins=n_bins,
        saturated_fraction=sum(acc.n_saturated) / total_valid,
        boundary=_boundary_stats(pooled, total_valid),
        metrics=metrics,
        classes=classes,
        failures=failures,
        curve=curve,
        image_dice=dice,
        warnings=[],
        soft_targets=soft,
    )
    report.warnings = _warnings(report, min_images, min_class_fraction, min_class_voxels)
    if background_class is not None and not soft and bool(mean_dice.isnan().all()):
        report.warnings.insert(
            0,
            "background_class leaves no class in the per-image Dice; "
            "image failure statistics are undefined",
        )
    return report


def _warnings(
    r: ReliabilityReport,
    min_images: int,
    min_class_fraction: float,
    min_class_voxels: int,
) -> list[str]:
    out = []
    if r.soft_targets:
        out.append(
            "soft targets: hard-label Dice and failure statistics are undefined "
            "(Dice fields and image_dice are NaN)"
        )
    if r.n_images < min_images:
        out.append(
            f"only {r.n_images} image{'s' * (r.n_images != 1)} (< {min_images}): the intervals are wide because the "
            "uncertainty comes from the number of images, not voxels"
        )
    for c in r.classes:
        if c.label_voxels == 0:
            out.append(
                f"class '{c.name}' is absent from the labels: its Dice and per-image summaries are undefined; its class ECE is computed against empty labels"
            )
        elif c.pred_voxels == 0:
            out.append(f"class '{c.name}' is never predicted")
        else:
            if c.label_fraction < min_class_fraction:
                out.append(
                    f"class '{c.name}' is rare ({c.label_voxels} voxels, "
                    f"{100 * c.label_fraction:.2f}% of labels): its estimates are noisy"
                )
            elif c.label_voxels < min_class_voxels:
                out.append(
                    f"class '{c.name}' has only {c.label_voxels} voxels in the labels "
                    f"({100 * c.label_fraction:.2f}%): its estimates are noisy"
                )
            if c.pred_voxels < min_class_voxels:
                out.append(
                    f"class '{c.name}' is predicted on only {c.pred_voxels} voxels: "
                    "class-conditional calibrators (CDC, CMSap, CMSop) fit one map per "
                    "predicted class, so this map would rest on very few voxels"
                )
        if c.n_present and c.n_failed / c.n_present >= _FAIL_RATE_ALERT:
            out.append(
                f"class '{c.name}' fails (Dice < {r.failures.threshold:g}) on "
                f"{c.n_failed}/{c.n_present} of the images where it is present"
            )
        if c.n_spurious:
            out.append(
                f"class '{c.name}' is predicted on {c.n_spurious} images where it is absent "
                "from the labels"
            )
    f = r.failures
    if f.n_failed:
        out.append(
            f"{f.n_failed} image(s) have a mean Dice below {f.threshold:g} and enter the "
            f"fit; ECE is {_f(f.ece_all)} with them and {_f(f.ece_without_failed)} without"
        )
    if r.bias in ("overconfident", "underconfident"):
        out.append(
            f"the model is {r.bias} on this set (confidence - accuracy = {_f(r.metrics['gap'][0])})"
        )
    return out
