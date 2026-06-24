"""Tests for the optional plotting helpers."""

from __future__ import annotations

import pytest

from conftest import synthetic_logits, to_probs

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def test_import_fiducio_does_not_load_matplotlib():
    import subprocess
    import sys

    code = (
        "import sys, fiducio; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib loaded eagerly'; "
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_reliability_diagram_returns_axes():
    import matplotlib.pyplot as plt

    from fiducio.plots import reliability_diagram

    logits, labels = synthetic_logits((4, 3, 10, 10), seed=70)
    probs = to_probs(logits)
    ax = reliability_diagram(probs, labels, n_bins=10)
    assert ax.__class__.__name__ == "Axes"
    plt.close("all")


def test_reliability_diagram_accepts_existing_axes():
    import matplotlib.pyplot as plt

    from fiducio.plots import reliability_diagram

    logits, labels = synthetic_logits((3, 4, 8, 8), seed=71)
    fig, ax = plt.subplots()
    out = reliability_diagram(to_probs(logits), labels, ax=ax, title=None, show_ece=False)
    assert out is ax
    plt.close(fig)
