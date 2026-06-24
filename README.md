# Fiducio

[![Tests](https://github.com/fiducio-ai/Fiducio/actions/workflows/tests.yml/badge.svg)](https://github.com/fiducio-ai/Fiducio/actions/workflows/tests.yml)
[![Docs](https://github.com/fiducio-ai/Fiducio/actions/workflows/docs.yml/badge.svg)](https://fiducio-ai.github.io/Fiducio/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Fiducio** is a model-agnostic Python library for **post-hoc calibration of 2D
and 3D semantic segmentation** models. Give it the logits or probabilities of any
segmentation model and a labelled calibration set, and it returns
better-calibrated probabilities — without retraining and without changing the
segmentation.

It works with PyTorch U-Net, nnU-Net, SegFormer or any model that produces
logits: Fiducio only ever sees `(B, C, *spatial)` tensors and never needs to know
the architecture.

## Install

```bash
pip install fiducio
```

Until the first PyPI release, install from GitHub:

```bash
pip install "git+https://github.com/fiducio-ai/Fiducio.git"
```

The core install depends only on `numpy` and `torch`. Optional extras:
`fiducio[plots]`, `fiducio[medical]`, `fiducio[docs]`, `fiducio[dev]`,
`fiducio[all]`.

## Quick start

```python
import torch
from fiducio import TemperatureScaling, load_calibrator

# From a held-out, labelled calibration set:
logits = torch.randn(8, 4, 64, 64)          # (B, C, H, W) model logits
labels = torch.randint(0, 4, (8, 64, 64))   # (B, H, W) labels

calibrator = TemperatureScaling(input_type="logits").fit(logits, labels)

# Apply to new predictions (returns calibrated probabilities):
probs = calibrator.transform(torch.randn(2, 4, 64, 64))

# Save and reload (loads on CPU by default):
calibrator.save("calibrator.pt")
calibrator = load_calibrator("calibrator.pt")
```

Every calibrator exposes the same API:
`fit(predictions, targets, mask=None)`, `transform`, `predict_proba`,
`fit_transform`, `save`, and the top-level `load_calibrator`.

## Accepted tensor shapes

The class axis is **always** dimension 1.

| Use case | `predictions` | `targets` |
|----------|---------------|-----------|
| 2D segmentation | `(B, C, H, W)` | `(B, H, W)` |
| 3D segmentation | `(B, C, D, H, W)` | `(B, D, H, W)` |
| general n-D | `(B, C, *spatial)` | `(B, *spatial)` |
| per-pixel table | `(N, C)` | `(N,)` |

Inputs may be `logits` or `probs` (set `input_type`). Binary segmentation is two
channels (`C = 2`). `mask` and `ignore_index` exclude voxels from fitting.

## Calibrators

| Class | Alias | Guarantee |
|-------|-------|-----------|
| `TemperatureScaling` | `TS` | preserves argmax and full order |
| `EnsembleTemperatureScaling` | `ETS` | — |
| `VectorScaling` | `VS` | — |
| `MatrixScaling` | `MS` | — |
| `TranslationInvariantMatrixScaling` | — | invariant to logit translation |
| `DirichletCalibration` | — | — |
| `ClassConditionalMatrixScaling` | `CMS` | — |
| `ArgmaxPreservingMatrixScaling` | `CMSAP` | preserves argmax |
| `OrderPreservingMatrixScaling` | `CMSOP` | preserves full order |

Calibration metrics are included: `negative_log_likelihood`,
`expected_calibration_error`, `brier_score`.

## Documentation

Full guide and API reference: **https://fiducio-ai.github.io/Fiducio/**

- [Installation](https://fiducio-ai.github.io/Fiducio/installation/)
- [Quickstart](https://fiducio-ai.github.io/Fiducio/quickstart/)
- [Concepts](https://fiducio-ai.github.io/Fiducio/concepts/)
- [Calibrators](https://fiducio-ai.github.io/Fiducio/calibrators/)
- [Examples](https://fiducio-ai.github.io/Fiducio/examples/)

## Citation

If you use Fiducio in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## Status

Fiducio is **beta** (`0.x`): usable and tested, with an API that may still change
before `1.0`. It is not certified for safety-critical or clinical use.

## License

[Apache-2.0](LICENSE).
