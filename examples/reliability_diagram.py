"""Plot before/after reliability diagrams for a calibrated model.

Needs the plotting extra (``pip install "fiducio[plots]"``). It writes a PNG to a
temporary directory and prints the path; it never opens a window. Run with::

    python examples/reliability_diagram.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

try:
    import matplotlib

    matplotlib.use("Agg")  # headless backend, no display needed
    import matplotlib.pyplot as plt
except ImportError:
    print('This example needs matplotlib. Install it with: pip install "fiducio[plots]"')
    sys.exit(0)

from fiducio import TemperatureScaling
from fiducio.plots import reliability_diagram


def main() -> None:
    torch.manual_seed(0)
    # Over-confident synthetic logits and matching labels.
    signal = torch.randn(16, 4, 32, 32)
    labels = signal.argmax(dim=1)
    logits = signal * 4.0 + torch.randn(16, 4, 32, 32) * 0.5

    calibrator = TemperatureScaling().fit(logits, labels)

    raw_probs = torch.softmax(logits, dim=1)
    cal_probs = calibrator.transform(logits)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    reliability_diagram(raw_probs, labels, ax=ax1, title="Before calibration")
    reliability_diagram(cal_probs, labels, ax=ax2, title="After calibration")
    fig.tight_layout()

    out = Path(tempfile.gettempdir()) / "fiducio_reliability.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to {out}")


if __name__ == "__main__":
    main()
