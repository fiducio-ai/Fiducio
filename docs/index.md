# Fiducio

**Fiducio** is a model-agnostic Python library for **post-hoc calibration of 2D
and 3D semantic segmentation** models. It turns the logits or probabilities of
any segmentation model into better-calibrated probabilities, using a small
labelled calibration set and no retraining.

Fiducio does not depend on any particular architecture: a PyTorch U-Net,
nnU-Net, SegFormer or any other model that produces logits can be calibrated.
It accompanies the paper
[*Rethinking Post-Hoc Calibration in Semantic Segmentation*](https://openreview.net/forum?id=xwNoSNxgxV)
(Kirscher et al., Transactions on Machine Learning Research, 2026; preprint
[arXiv:2607.01902](https://arxiv.org/abs/2607.01902)), which introduces the
translation-invariant and class-conditional calibrators (MSc, CDC, CMSap, CMSop)
implemented here.

## Why calibrate?

Segmentation networks can be **over-confident**: predictions with confidence
0.99 need not be correct 99% of the time. Post-hoc calibration learns a
lightweight transform on a held-out calibration set that aims to make predicted
probabilities match observed frequencies, without retraining the model. See
[Concepts](concepts.md) for the setup and
[Calibrators](calibrators.md) for the decision and ordering guarantees.

## Install

```bash
pip install fiducio
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.10. See
[Installation](installation.md) for extras, the GitHub install and the security
rationale for the PyTorch floor.

## A 30-second example

```python
import torch
from fiducio import TemperatureScaling

torch.manual_seed(0)
logits = torch.randn(8, 4, 64, 64)  # (B, C, H, W) from your model
labels = torch.randint(0, 4, (8, 64, 64))  # (B, H, W) calibration labels

calibrator = TemperatureScaling(device="cpu").fit(logits, labels)
new_logits = torch.randn(2, 4, 64, 64)  # separate cases
probs = calibrator.transform(new_logits)  # calibrated probabilities
```

These random tensors demonstrate the API, not a calibration benchmark. Continue
with the [Quickstart](quickstart.md) or see
[before/after reliability diagrams](plots.md).

## Documentation

Start with the [Quickstart](quickstart.md), then
[Concepts](concepts.md), [Calibrators](calibrators.md) and the
[Examples](examples.md). The
[paper project page](https://fiducio-ai.github.io/Fiducio/paper/) has
interactive illustrations and published results.

## Project status

Fiducio is **beta** (`0.x`). The API is usable and tested, but may still change
before `1.0`. It is not certified for safety-critical clinical use.
