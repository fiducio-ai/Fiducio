"""Optional plotting helpers.

These require the ``plots`` extra (``pip install "fiducio[plots]"``). Matplotlib
is imported lazily inside each function, so importing :mod:`fiducio` never pulls
in matplotlib.
"""

from __future__ import annotations

from typing import Any

from ..metrics.calibration import reliability_curve

__all__ = ["reliability_diagram"]


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "plotting requires matplotlib; install it with "
            '`pip install "fiducio[plots]"`'
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
) -> Any:
    """Draw a top-1 reliability diagram and return the matplotlib ``Axes``.

    Parameters
    ----------
    probs:
        ``(B, C, *spatial)`` probabilities.
    targets:
        ``(B, *spatial)`` integer labels.
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
        non-empty bins — see :func:`fiducio.average_calibration_error`), or
        ``None`` to omit the annotation.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the diagram was drawn on.
    """
    plt = _require_matplotlib()
    curve = reliability_curve(probs, targets, mask, ignore_index, n_bins)

    edges = curve.bin_edges.detach().cpu().numpy()
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]
    accuracy = curve.bin_accuracy.detach().cpu().numpy()
    confidence = curve.bin_confidence.detach().cpu().numpy()
    counts = curve.bin_counts.detach().cpu().numpy()
    populated = counts > 0

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.bar(
        centers[populated],
        accuracy[populated],
        width=width * 0.95,
        edgecolor="black",
        alpha=0.75,
        label="accuracy",
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
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    if metric is not None:
        if metric == "ece":
            label, value = "ECE", curve.ece
        elif metric == "ace":
            label, value = "ACE", curve.ace
        else:
            raise ValueError(f"metric must be 'ece', 'ace' or None, got {metric!r}")
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
