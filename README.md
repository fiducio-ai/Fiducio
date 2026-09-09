# Fiducio

[![Tests](https://github.com/fiducio-ai/Fiducio/actions/workflows/tests.yml/badge.svg)](https://github.com/fiducio-ai/Fiducio/actions/workflows/tests.yml)
[![Docs](https://github.com/fiducio-ai/Fiducio/actions/workflows/docs.yml/badge.svg)](https://fiducio-ai.github.io/Fiducio/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Fiducio** is a model-agnostic Python library for **post-hoc calibration of 2D
and 3D semantic segmentation** models. Give it the logits or probabilities of any
segmentation model and a labelled calibration set, and it returns
calibrated probabilities without retraining the segmentation model. Whether the
segmentation is preserved depends on the calibrator; see the guarantees below.

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
`fiducio[plots]` (matplotlib reliability diagrams), `fiducio[docs]`,
`fiducio[dev]`, `fiducio[all]`.

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
| `TranslationInvariantMatrixScaling` | `MSc` | invariant to logit translation |
| `DirichletCalibration` | `DC` | — |
| `ClassConditionalMatrixScaling` | `CMS` / `CDC` | — |
| `ArgmaxPreservingMatrixScaling` | `CMSAP` / `CMSap` | preserves argmax |
| `OrderPreservingMatrixScaling` | `CMSOP` / `CMSop` | preserves full order |

Every calibrator also exposes `decision_function` (calibrated logits) and a
configurable `optimizer` (`"adam"` default, or `"lbfgs"`).

### Paper method mapping

| Paper method | Class |
|--------------|-------|
| TS | `TemperatureScaling` |
| MS | `MatrixScaling` |
| MSc | `TranslationInvariantMatrixScaling` |
| CDC | `ClassConditionalMatrixScaling` |
| CMSap | `ArgmaxPreservingMatrixScaling` |
| CMSop | `OrderPreservingMatrixScaling` |

CDC, CMSap and CMSop are class-conditional: they fit one affine map per
uncalibrated top class, all experts optimized **jointly** by a single
optimizer minimizing one cross-entropy loss over every voxel at once, with L2
regularization (`lambda_reg` / `mu_reg`) applied to the affine map induced in
the common *logit* space rather than to the raw per-expert parameters — this
keeps the regularization meaningful for CMSap/CMSop, whose parameters are
non-negative margins/gaps rather than raw matrix entries. Pass
`independent_experts=True` to instead fit each expert in its own optimization
loop on only the voxels routed to it.

Calibration metrics are included: `negative_log_likelihood`,
`expected_calibration_error` (ECE, population-weighted), `average_calibration_error`
(ACE, unweighted over the same bins — doesn't let a sparsely populated bin get
drowned out by a large one), `brier_score`, `reliability_curve`, plus an
optional reliability-diagram plot (`fiducio.plots.reliability_diagram`, needs
`fiducio[plots]`).

## Documentation

The [paper implementation guide](reproducibility/README.md) records method
mapping, numerical-reference coverage, fitting limits and the remaining steps
needed to reproduce the paper's experiments. For ensemble inputs, see
[`examples/ensemble_pooling.py`](examples/ensemble_pooling.py).

Full guide and API reference: **https://fiducio-ai.github.io/Fiducio/**

- [Installation](https://fiducio-ai.github.io/Fiducio/installation/)
- [Quickstart](https://fiducio-ai.github.io/Fiducio/quickstart/)
- [Concepts](https://fiducio-ai.github.io/Fiducio/concepts/)
- [Calibrators](https://fiducio-ai.github.io/Fiducio/calibrators/)
- [Examples](https://fiducio-ai.github.io/Fiducio/examples/)

## Citation

Fiducio accompanies the paper *Rethinking Post-Hoc Calibration in Semantic
Segmentation* ([arXiv:2607.01902](https://arxiv.org/abs/2607.01902)). If you use
Fiducio in your research, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## Status

Fiducio is **beta** (`0.x`): usable and tested, with an API that may still change
before `1.0`. It is not certified for safety-critical or clinical use.

## License

[Apache-2.0](LICENSE).
