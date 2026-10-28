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
from dataclasses import dataclass, field
from typing import Any

import torch

from .metrics.calibration import (
    ReliabilityCurve,
    brier_score,
    negative_log_likelihood,
    reliability_curve,
)
from .plots import _draw_reliability, _require_matplotlib
from .utils import to_tensor, validate_predictions, validate_targets
from .utils.tensors import prepare_targets

__all__ = ["ClassStats", "FailureStats", "ReliabilityReport", "diagnose"]

_FAIL_RATE_ALERT = 0.1  # a class failing on this share of its images is flagged
_SATURATION = 0.999  # confidence above which a voxel counts as saturated


@dataclass
class ClassStats:
    """Per-class balance, calibration and Dice summary (NaN when undefined)."""

    name: str
    label_voxels: int
    label_fraction: float
    pred_voxels: int
    pred_fraction: float
    dice: float  # pooled over all images
    class_ece: float  # p_k against 1[label == k]
    routed_ece: float  # top-1 ECE on voxels whose argmax is this class
    n_present: int  # images where the class is in the labels
    n_failed: int  # of those, images with Dice below the threshold
    n_spurious: int  # images where the class is predicted but absent from the labels
    dice_median: float  # Dice over the images where the class is present
    dice_p10: float
    dice_min: float


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
    the class is absent from both labels and prediction.
    """

    n_images: int
    n_valid: int
    n_bins: int
    saturated_fraction: float
    metrics: dict[str, tuple[float, float, float]]
    classes: list[ClassStats]
    failures: FailureStats
    curve: ReliabilityCurve
    image_dice: torch.Tensor
    warnings: list[str]

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


def diagnose(
    logits: Any,
    labels: Any,
    *,
    input_type: str = "logits",
    mask: Any | None = None,
    ignore_index: int = -100,
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
        ``B`` is the number of images or cases.
    labels:
        ``(B, *spatial)`` integer labels. Soft targets are not supported.
    mask, ignore_index:
        Voxel selection, as in the metrics.
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
        Bootstrap resamples over images for the 95% intervals (``0`` disables them)
        and the seed of the resampling.
    min_images, min_class_fraction, min_class_voxels:
        Thresholds of the warnings. They are heuristics, not statistical guarantees.

    Notes
    -----
    The ECE interval is a percentile bootstrap. ECE has a finite-sample noise floor that
    resampling raises further, so for a well-calibrated model the interval can sit
    slightly above the estimate (it missed it in 15 of 100 simulated calibrated models
    of 24 images). Read it as an order of magnitude. The *bias* line uses the
    voxel-weighted confidence-accuracy gap, which has no such noise floor, but its
    intervals are slightly narrow: it flagged 9 of those 100 calibrated models, against
    5 expected (``reproducibility/diagnose_interval_check.py`` reruns the simulation).

    Cost is ``2C + 1`` reliability curves per image (global, per-class and routed),
    plus a few passes for NLL, Brier and the confusion matrix. Classes absent from both
    labels and prediction have an undefined Dice and are never counted as failures.
    """
    if not isinstance(ignore_index, int):
        raise TypeError(f"ignore_index must be an int, got {ignore_index!r}")
    if input_type not in ("logits", "probs"):
        raise ValueError(f"input_type must be 'logits' or 'probs', got {input_type!r}")
    if n_boot < 0 or n_worst < 0:
        raise ValueError("n_boot and n_worst must be >= 0")
    if not 0.0 <= dice_failure_threshold <= 1.0:
        raise ValueError("dice_failure_threshold must lie in [0, 1]")

    x = to_tensor(logits).detach()
    if not x.is_floating_point():
        x = x.float()
    m = None if mask is None else to_tensor(mask, device=x.device).bool()
    if x.ndim < 3:
        raise ValueError(
            f"logits must have shape (B, C, *spatial) with at least one spatial axis; got {tuple(x.shape)}"
        )
    num_classes = validate_predictions(x, input_type=input_type, mask=m)
    y = prepare_targets(x, labels)
    if y.shape == x.shape:
        raise ValueError("diagnose supports hard integer labels only, not class distributions")
    validate_targets(x, y, m, num_classes=num_classes, ignore_index=ignore_index)
    n_images = int(x.shape[0])
    if n_images == 0:
        raise ValueError("at least one image is required")
    names = list(class_names) if class_names is not None else [str(c) for c in range(num_classes)]
    ids = list(image_ids) if image_ids is not None else [str(i) for i in range(n_images)]
    if len(names) != num_classes or len(ids) != n_images:
        raise ValueError("class_names must have one name per class and image_ids one per image")
    if background_class is not None and not 0 <= background_class < num_classes:
        raise ValueError(f"background_class must lie in [0, {num_classes - 1}]")

    # --- per-image sufficient statistics --------------------------------------------------
    top, per_class, routed, confusion = [], [], [], []
    nll_sum, brier_sum, n_valid, n_saturated = [], [], [], []
    for i in range(n_images):
        xi, yi = x[i : i + 1], y[i : i + 1]
        mi = None if m is None else m[i : i + 1]
        if input_type == "probs":
            p = xi
        else:
            p = torch.softmax(xi.double() if xi.dtype == torch.float64 else xi.float(), dim=1)
        valid = yi != ignore_index
        if mi is not None:
            valid = valid & mi
        pred = p.argmax(dim=1)
        curve = reliability_curve(p, yi, mi, ignore_index, n_bins)
        top.append(_bin_stats(curve))
        per_class.append(
            torch.stack(
                [
                    _bin_stats(reliability_curve(p, yi, mi, ignore_index, n_bins, class_index=c))
                    for c in range(num_classes)
                ]
            )
        )
        routed.append(
            torch.stack(
                [
                    _bin_stats(reliability_curve(p, yi, valid & (pred == c), ignore_index, n_bins))
                    for c in range(num_classes)
                ]
            )
        )
        n = int(valid.sum())
        n_valid.append(n)
        has_voxels = n > 0
        nll_sum.append(n * negative_log_likelihood(p, yi, mi, ignore_index) if has_voxels else 0.0)
        brier_sum.append(n * brier_score(p, yi, mi, ignore_index) if has_voxels else 0.0)
        n_saturated.append(int((p.max(dim=1).values[valid] > _SATURATION).sum()))
        joint = yi[valid] * num_classes + pred[valid]
        confusion.append(torch.bincount(joint, minlength=num_classes**2).view(num_classes, -1))

    # Everything below is small: move it to the CPU once.
    total_valid = float(sum(n_valid))
    if total_valid == 0:
        raise ValueError("no valid voxels: every label is ignored or masked out")
    top_s = torch.stack(top).cpu()  # (B, bins, 3)
    class_s = torch.stack(per_class).cpu()  # (B, C, bins, 3)
    routed_s = torch.stack(routed).cpu()
    conf = torch.stack(confusion).cpu().double()  # (B, C, C): rows labels, columns predictions
    nll_t = torch.tensor(nll_sum, dtype=torch.float64)
    brier_t = torch.tensor(brier_sum, dtype=torch.float64)
    nvox = torch.tensor(n_valid, dtype=torch.float64)

    # --- global metrics and bootstrap over images -----------------------------------------
    pooled = top_s.sum(0)
    point = {
        "ece": float(_ece(pooled)),
        "nll": float(nll_t.sum() / nvox.sum()),
        "brier": float(brier_t.sum() / nvox.sum()),
        "gap": float(_gap(pooled)),
    }
    intervals = {k: (float("nan"), float("nan")) for k in point}
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
    metrics = {k: (point[k], *intervals[k]) for k in point}

    # --- Dice, failures -------------------------------------------------------------------
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

    routed_pool, class_pool = routed_s.sum(0), class_s.sum(0)
    classes = []
    for c in range(num_classes):
        d_c = dice[:, c][present[:, c]]
        median, p10, lowest = _dice_summary(d_c)
        label_vox, pred_vox = int(pooled_conf[c].sum()), int(pooled_conf[:, c].sum())
        classes.append(
            ClassStats(
                name=names[c],
                label_voxels=label_vox,
                label_fraction=label_vox / total_valid,
                pred_voxels=pred_vox,
                pred_fraction=pred_vox / total_valid,
                dice=float(pooled_dice[c]),
                class_ece=float(_ece(class_pool[c])),
                routed_ece=float(_ece(routed_pool[c])),
                n_present=int(present[:, c].sum()),
                n_failed=int(failed_cls[:, c].sum()),
                n_spurious=int(spurious[:, c].sum()),
                dice_median=median,
                dice_p10=p10,
                dice_min=lowest,
            )
        )

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
        ece_without_failed=float(_ece(top_s[rest].sum(0))) if rest.any() else float("nan"),
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
        saturated_fraction=sum(n_saturated) / total_valid,
        metrics=metrics,
        classes=classes,
        failures=failures,
        curve=curve,
        image_dice=dice,
        warnings=[],
    )
    report.warnings = _warnings(report, min_images, min_class_fraction, min_class_voxels)
    return report


def _warnings(
    r: ReliabilityReport,
    min_images: int,
    min_class_fraction: float,
    min_class_voxels: int,
) -> list[str]:
    out = []
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
