# Fiducio

**Fiducio** is a model-agnostic Python library for **post-hoc calibration of 2D
and 3D semantic segmentation** models. It turns the raw logits or probabilities
of any segmentation model into better-calibrated probabilities, using a small
labelled calibration set.

Fiducio does not depend on any particular architecture. If your model produces
logits, Fiducio can calibrate them — whether it is a PyTorch U-Net, nnU-Net,
SegFormer or anything else.

## Why calibrate?

Modern segmentation networks are usually **over-confident**: a voxel predicted
with probability 0.99 is correct far less than 99% of the time. Post-hoc
calibration learns a lightweight transform on a held-out calibration set that
makes predicted probabilities match observed frequencies, without retraining or
changing the segmentation itself.

## What you get

- A single, consistent API for every calibrator:
  `fit` / `transform` / `predict_proba` / `fit_transform` / `save`, plus
  `fiducio.load_calibrator`.
- Works on 2D `(B, C, H, W)`, 3D `(B, C, D, H, W)` and general `(B, C, *spatial)`
  inputs, with logits **or** probabilities, masks and `ignore_index`.
- A family of calibrators from simple temperature scaling to class-conditional
  matrix scaling with argmax- and order-preserving guarantees.
- Calibration metrics (NLL, ECE, Brier) and safe, registry-backed persistence.

## Install

```bash
pip install fiducio
```

See [Installation](installation.md) for extras and the GitHub install.

## A 30-second example

```python
import torch
from fiducio import TemperatureScaling

logits = torch.randn(8, 4, 64, 64)          # (B, C, H, W) from your model
labels = torch.randint(0, 4, (8, 64, 64))   # (B, H, W) calibration labels

calibrator = TemperatureScaling().fit(logits, labels)
probs = calibrator.transform(logits)        # calibrated probabilities
```

Continue with the [Quickstart](quickstart.md).

## Project status

Fiducio is **beta** (`0.x`). The API is usable and tested, but may still change
before `1.0`. It is not certified for safety-critical clinical use.
