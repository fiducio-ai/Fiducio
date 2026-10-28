"""Optional plotting helpers.

These require the ``plots`` extra (``pip install "fiducio[plots]"``). Matplotlib
is imported lazily inside each function, so importing :mod:`fiducio` never pulls
in matplotlib.
"""

from __future__ import annotations

from typing import Any

from ..metrics.calibration import ReliabilityCurve, reliability_curve

__all__ = ["reliability_diagram"]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            'plotting requires matplotlib; install it with `pip install "fiducio[plots]"`'
        ) from exc
    return plt


def reliability_diagram(
    probs: Any,
    targets: Any,
    mask: Any | None = None,
    ignore_index: int = -100,
    n_bins: int = 15,
    *,
    ax: Any | None = None,
    title: str | None = "Reliability diagram",
    metric: str | None = "ece",
    class_index: int | None = None,
) -> Any:
    """Draw top-1 or class reliability and return the matplotlib ``Axes``.

    Parameters
    ----------
    probs:
        ``(B, C, *spatial)`` probabilities.
    targets:
        ``(B, *spatial)`` integer labels or ``(B, C, *spatial)`` floating class
        distributions, including mean human responses.
    mask, ignore_index:
        Optional voxel selection (see :class:`fiducio.Calibrator`).
    n_bins:
        Number of confidence bins.
    ax:
        Existing axes to draw on. A new figure/axes is created when ``None``.
    title:
        Plot title (``None`` to omit).
    metric:
        Which calibration error to annotate the panel with: ``"ece"``
        (population-weighted, the default), ``"ace"`` (unweighted mean over
        non-empty bins — see :func:`fiducio.unweighted_calibration_error`), or
        ``None`` to omit the annotation.
    class_index:
        Optional class whose probability is compared with the observed class
        proportion (MHR for soft targets). The annotated ECE is against these
        targets, not the mean ECE against individual raters. See TwinTrack,
        Kirscher et al., MIDL 2026, https://arxiv.org/abs/2604.15950v2.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn on.
    """
    if metric not in (None, "ece", "ace"):
        raise ValueError(f"metric must be 'ece', 'ace' or None, got {metric!r}")
    plt = _require_matplotlib()
    curve = reliability_curve(probs, targets, mask, ignore_index, n_bins, class_index=class_index)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    return _draw_reliability(ax, curve, class_index=class_index, title=title, metric=metric)


def _draw_reliability(
    ax: Any,
    curve: ReliabilityCurve,
    *,
    class_index: int | None,
    title: str | None,
    metric: str | None,
) -> Any:
    """Draw an already computed reliability curve on ``ax``."""
    edges = curve.bin_edges.detach().cpu().numpy()
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]
    accuracy = curve.bin_accuracy.detach().cpu().numpy()
    confidence = curve.bin_confidence.detach().cpu().numpy()
    counts = curve.bin_counts.detach().cpu().numpy()
    populated = counts > 0

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.bar(
        centers[populated],
        accuracy[populated],
        width=width * 0.95,
        edgecolor="black",
        alpha=0.75,
        label="accuracy" if class_index is None else "observed proportion",
    )
    ax.plot(
        confidence[populated],
        accuracy[populated],
        marker="o",
        linestyle="none",
        color="black",
        label="confidence",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence" if class_index is None else f"class {class_index} probability")
    ax.set_ylabel(
        "agreement" if class_index is None else "observed proportion / mean human response"
    )
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if metric is not None:
        if metric == "ece":
            label, value = "ECE (against targets)", curve.ece
        elif metric == "ace":
            label, value = "ACE", curve.ace
        ax.text(
            0.05,
            0.95,
            f"{label} = {value:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    ax.legend(loc="lower right", fontsize="small")
    return ax
