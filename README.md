# Fiducio

[![Tests](https://github.com/fiducio-ai/Fiducio/actions/workflows/tests.yml/badge.svg)](https://github.com/fiducio-ai/Fiducio/actions/workflows/tests.yml)
[![Docs](https://github.com/fiducio-ai/Fiducio/actions/workflows/docs.yml/badge.svg)](https://fiducio-ai.github.io/Fiducio/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/fiducio-ai/Fiducio/blob/main/LICENSE)

**Fiducio** is a model-agnostic Python library of **post-hoc calibrators for 2D
and 3D semantic segmentation**. Give it the logits or probabilities of any
segmentation model and a labelled calibration set, and a calibrator returns
better-calibrated probabilities without retraining the segmentation model.
It implements temperature, vector, matrix and Dirichlet scaling, their
translation-invariant variants, and the class-conditional (CDC, CMSap, CMSop)
calibrators introduced in our paper
[*Rethinking Post-Hoc Calibration in Semantic Segmentation*](https://arxiv.org/abs/2607.01902),
accepted at TMLR. Whether the segmentation is preserved depends on the
calibrator; see the guarantees below.

It works with PyTorch U-Net, nnU-Net, SegFormer or any model that produces
logits: Fiducio only ever sees `(B, C, *spatial)` tensors and never needs to know
the architecture.

![Reliability diagrams of an over-confident model before and after calibration with TemperatureScaling and CMSop](https://raw.githubusercontent.com/fiducio-ai/Fiducio/main/docs/assets/reliability_before_after.png)

*Top-1 reliability diagrams on held-out synthetic data (regenerate with
[`examples/readme_figure.py`](https://github.com/fiducio-ai/Fiducio/blob/main/examples/readme_figure.py)).*

## Install

Fiducio is available on [PyPI](https://pypi.org/project/fiducio/):

```bash
pip install fiducio
```

or, in a uv application:

```bash
uv add fiducio
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.10. The core install depends only on `numpy` and `torch`. Optional extras:
`fiducio[plots]` (matplotlib reliability diagrams), `fiducio[docs]`,
`fiducio[dev]`, `fiducio[all]`.

To install the development version from GitHub:

```bash
pip install "git+https://github.com/fiducio-ai/Fiducio.git"
```

For development from a checkout, use the committed uv lockfile:

```bash
uv sync --locked
uv run --no-sync pytest
```

This uses CPU PyTorch for development. See [CONTRIBUTING.md](https://github.com/fiducio-ai/Fiducio/blob/main/CONTRIBUTING.md)
for the pinned uv version, documentation and build commands.

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

| Class | Alias | Translation-invariant | Decision preservation |
|-------|-------|:---------------------:|-----------------------|
| `TemperatureScaling` | `TS` | yes | argmax and full order |
| `EnsembleTemperatureScaling` | `ETS` | yes | argmax and full order |
| `VectorScaling` | `VS` | no | — |
| `MatrixScaling` | `MS` | no | — |
| `TranslationInvariantMatrixScaling` | `MSc` | yes | — |
| `DirichletCalibration` | `DC` | yes | — |
| `ClassConditionalMatrixScaling` | `CMS` / `CDC` | yes | — |
| `ArgmaxPreservingMatrixScaling` | `CMSAP` / `CMSap` | yes | argmax |
| `OrderPreservingMatrixScaling` | `CMSOP` / `CMSop` | yes | argmax and full order |

Translation-invariant means the output is unchanged when the same constant is
added to every input logit of a voxel.

Every calibrator also exposes `decision_function` (calibrated logits) and a
configurable `optimizer` (`"adam"` default, or `"lbfgs"`).

### Early stopping

Adam-fitted calibrators support the paper's recipe of early stopping on a
validation set (with an optional learning-rate decay on plateaus):

```python
from fiducio import OrderPreservingMatrixScaling

calibrator = OrderPreservingMatrixScaling(max_iter=2000, patience=20, lr_patience=10)
calibrator.fit(
    logits, labels,
    val_predictions=val_logits, val_targets=val_labels,  # held out from both
)
```

The iterate with the best validation NLL is kept. The default fixed budget
(`max_iter`) is short, so it can underfit expressive calibrators; prefer a larger
`max_iter` together with `patience` for CDC, CMSap and CMSop.

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

The [paper project page](https://fiducio-ai.github.io/Fiducio/paper/) presents the
methods, interactive illustrations and published results.

The [paper implementation guide](https://github.com/fiducio-ai/Fiducio/blob/main/reproducibility/README.md) records method
mapping, numerical-reference coverage, fitting limits and the remaining steps
needed to reproduce the paper's experiments. For ensemble inputs, see
[`examples/ensemble_pooling.py`](https://github.com/fiducio-ai/Fiducio/blob/main/examples/ensemble_pooling.py).

### Results in the paper

The paper compares these calibrators on real segmentation benchmarks. The
reliability diagrams below (paper appendix) show the uncalibrated ensemble
pooling (`p0`, `p̄`) against the best calibrator per dataset; the exact
protocol, datasets and metrics are in the paper. This repository ships the
calibrators, not the fitted models or benchmark pipelines, so it does not
regenerate these figures (see the
[implementation guide](https://github.com/fiducio-ai/Fiducio/blob/main/reproducibility/README.md)).

![Reliability diagrams from the paper on three datasets](https://raw.githubusercontent.com/fiducio-ai/Fiducio/main/docs/assets/paper_reliability.png)

Full guide and API reference: **https://fiducio-ai.github.io/Fiducio/**

- [Installation](https://fiducio-ai.github.io/Fiducio/installation/)
- [Quickstart](https://fiducio-ai.github.io/Fiducio/quickstart/)
- [Concepts](https://fiducio-ai.github.io/Fiducio/concepts/)
- [Calibrators](https://fiducio-ai.github.io/Fiducio/calibrators/)
- [Examples](https://fiducio-ai.github.io/Fiducio/examples/)

## Citation

Fiducio accompanies the paper *Rethinking Post-Hoc Calibration in Semantic
Segmentation* (Kirscher et al., Transactions on Machine Learning Research, 2026;
[OpenReview](https://openreview.net/forum?id=xwNoSNxgxV), preprint: [arXiv:2607.01902](https://arxiv.org/abs/2607.01902)).
If you use Fiducio in your research, please cite it using the metadata in
[`CITATION.cff`](https://github.com/fiducio-ai/Fiducio/blob/main/CITATION.cff), or:

```bibtex
@article{kirscher2026rethinking,
  title   = {Rethinking Post-Hoc Calibration in Semantic Segmentation},
  author  = {Kirscher, Tristan and Kahl, Kim-Celine and Kovacs, Balint and
             Rokuss, Maximilian and Maier-Hein, Klaus and Coubez, Xavier and
             Meyer, Philippe and Faisan, Sylvain},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=xwNoSNxgxV}
}
```

## Status

Fiducio is **beta** (`0.x`): usable and tested, with an API that may still change
before `1.0`. It is not certified for safety-critical or clinical use.

## License

[Apache-2.0](https://github.com/fiducio-ai/Fiducio/blob/main/LICENSE).
