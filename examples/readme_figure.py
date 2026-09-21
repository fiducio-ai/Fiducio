"""Regenerate the before/after reliability figure shown in the README.

Needs the plotting extra (``pip install "fiducio[plots]"``). Run from the
repository root::

    python examples/readme_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fiducio import CMSop, TemperatureScaling
from fiducio.plots import reliability_diagram


def synthetic_split(n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Over-confident 4-class logits whose over-confidence depends on the predicted class."""
    labels = torch.randint(0, 4, (n, 32, 32), generator=generator)
    noise = torch.randn(n, 4, 32, 32, generator=generator) * 2.0
    logits = noise.clone()
    logits.scatter_add_(1, labels.unsqueeze(1), torch.full((n, 1, 32, 32), 2.0))
    scale = torch.tensor([1.0, 1.5, 2.5, 4.0]).view(1, 4, 1, 1)
    return logits * scale, labels


def main() -> None:
    g = torch.Generator().manual_seed(0)
    fit_logits, fit_labels = synthetic_split(16, g)
    test_logits, test_labels = synthetic_split(16, g)

    panels = {
        "Uncalibrated": torch.softmax(test_logits, dim=1),
        "TemperatureScaling": TemperatureScaling(max_iter=2000).fit(fit_logits, fit_labels).transform(test_logits),
        "CMSop": CMSop(max_iter=2000).fit(fit_logits, fit_labels).transform(test_logits),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (title, probs) in zip(axes, panels.items(), strict=True):
        reliability_diagram(probs, test_labels, ax=ax, title=title)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "reliability_before_after.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
