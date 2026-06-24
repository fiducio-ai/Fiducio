"""Calibrate a generic PyTorch segmentation model on CPU.

This example uses a tiny ``nn.Module`` standing in for a real segmenter (U-Net,
nnU-Net, SegFormer, ...). The procedure is identical for any model: collect the
logits and labels for a held-out calibration set, fit a calibrator on them, then
apply the calibrator to the logits produced at inference time.

No weights or datasets are downloaded. Run with::

    python examples/pytorch_model_calibration.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fiducio import (
    MatrixScaling,
    expected_calibration_error,
    negative_log_likelihood,
)


class TinySegmenter(nn.Module):
    """A minimal stand-in for a real 2D segmentation network."""

    def __init__(self, in_channels: int = 1, num_classes: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)  # (B, C, H, W) logits


def make_loader(n: int, *, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 1, 32, 32, generator=g)
    # A label that correlates with the input so the model can learn something.
    labels = (images.squeeze(1) * 2).round().clamp(0, 2).long()
    return DataLoader(TensorDataset(images, labels), batch_size=4)


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over a loader and stack the logits and labels.

    Returns ``(B, C, H, W)`` logits and ``(B, H, W)`` labels, exactly the layout
    Fiducio expects.
    """
    model.eval()
    logit_batches, label_batches = [], []
    for images, labels in loader:
        logit_batches.append(model(images))
        label_batches.append(labels)
    return torch.cat(logit_batches), torch.cat(label_batches)


def main() -> None:
    torch.manual_seed(0)
    model = TinySegmenter(num_classes=3)

    # (In practice the model is already trained; we skip training here.)
    calib_loader = make_loader(24, seed=1)
    test_loader = make_loader(24, seed=2)

    calib_logits, calib_labels = collect_logits(model, calib_loader)
    test_logits, test_labels = collect_logits(model, test_loader)

    raw_probs = F.softmax(test_logits, dim=1)
    print("Before calibration:")
    print(f"  NLL = {negative_log_likelihood(raw_probs, test_labels):.4f}")
    print(f"  ECE = {expected_calibration_error(raw_probs, test_labels):.4f}")

    # Fit a calibrator on the calibration logits and labels.
    calibrator = MatrixScaling(input_type="logits", lambda_reg=1e-2, device="cpu")
    calibrator.fit(calib_logits, calib_labels)

    # Apply calibration at inference time.
    calibrated_probs = calibrator.transform(test_logits)
    print("\nAfter calibration:")
    print(f"  NLL = {negative_log_likelihood(calibrated_probs, test_labels):.4f}")
    print(f"  ECE = {expected_calibration_error(calibrated_probs, test_labels):.4f}")
    print("\nThe same recipe works for U-Net, nnU-Net or SegFormer logits.")


if __name__ == "__main__":
    main()
