"""Calibrate synthetic 2D segmentation logits with temperature scaling.

Runs in a couple of seconds on CPU and needs no external data. It:

1. creates overconfident synthetic 2D logits and labels;
2. splits them into a calibration set and a test set;
3. fits :class:`TemperatureScaling` on the calibration set;
4. reports NLL and ECE before and after calibration on the test set;
5. saves the calibrator, reloads it and checks the output is identical.

Run with::

    python examples/basic_segmentation_calibration.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from fiducio import (
    TemperatureScaling,
    expected_calibration_error,
    load_calibrator,
    negative_log_likelihood,
)


def make_overconfident_logits(
    n: int, num_classes: int, size: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(B, C, H, W)`` logits and ``(B, H, W)`` labels.

    The logits are deliberately over-confident (scaled up) so calibration has
    something to correct.
    """
    g = torch.Generator().manual_seed(seed)
    signal = torch.randn(n, num_classes, size, size, generator=g)
    labels = signal.argmax(dim=1)
    logits = signal * 4.0 + torch.randn(n, num_classes, size, size, generator=g) * 0.5
    return logits, labels


def main() -> None:
    num_classes = 4
    cal_logits, cal_labels = make_overconfident_logits(16, num_classes, 24, seed=0)
    test_logits, test_labels = make_overconfident_logits(16, num_classes, 24, seed=1)

    raw_probs = F.softmax(test_logits, dim=1)
    print("Before calibration (test set):")
    print(f"  NLL = {negative_log_likelihood(raw_probs, test_labels):.4f}")
    print(f"  ECE = {expected_calibration_error(raw_probs, test_labels):.4f}")

    # Fit on the calibration set only, never on the test set.
    calibrator = TemperatureScaling(input_type="logits", device="cpu")
    calibrator.fit(cal_logits, cal_labels)
    print(f"\nFitted temperature T = {calibrator.temperature:.4f}")

    cal_probs = calibrator.transform(test_logits)
    print("\nAfter calibration (test set):")
    print(f"  NLL = {negative_log_likelihood(cal_probs, test_labels):.4f}")
    print(f"  ECE = {expected_calibration_error(cal_probs, test_labels):.4f}")

    # Persist and reload.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "temperature.pt"
        calibrator.save(path)
        reloaded = load_calibrator(path)
        reloaded_probs = reloaded.transform(test_logits)

    assert torch.allclose(cal_probs, reloaded_probs, atol=1e-6)
    print("\nSaved, reloaded and verified identical output. Done.")


if __name__ == "__main__":
    main()
