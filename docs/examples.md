# Examples

Runnable, CPU-only examples ship in the
[`examples/`](https://github.com/fiducio-ai/Fiducio/tree/main/examples) directory
**of the source repository** — they are not part of the installed wheel, so clone
the repository (or download a single script) before running the commands below.
The examples below do not download any model or dataset; the
[Oxford Pets tutorial](tutorial.md) downloads a public dataset (or runs offline with `--smoke`).

Install `fiducio[plots]` for the plotting and multi-rater examples. From a
development checkout, `uv sync --locked` includes matplotlib; prefix the commands
below with `uv run --no-sync` to use that environment.

## 1. Synthetic data

[`examples/basic_segmentation_calibration.py`](https://github.com/fiducio-ai/Fiducio/blob/main/examples/basic_segmentation_calibration.py)

Creates over-confident synthetic 2D logits, splits them into calibration and test
sets, fits `TemperatureScaling`, prints NLL and ECE before and after, then saves,
reloads and verifies the calibrator produces identical output.

```bash
python examples/basic_segmentation_calibration.py
```

## 2. A generic PyTorch model

[`examples/pytorch_model_calibration.py`](https://github.com/fiducio-ai/Fiducio/blob/main/examples/pytorch_model_calibration.py)

Uses a tiny `nn.Module` segmenter to show the real-world workflow: run the model
over a `DataLoader`, stack the logits and labels, fit a calibrator and apply it
at inference time.

```bash
python examples/pytorch_model_calibration.py
```

The key step is collecting logits in the layout Fiducio expects:

```python
@torch.no_grad()
def collect_logits(model, loader):
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(images))  # (B, C, H, W)
        labels.append(targets)  # (B, H, W)
    return torch.cat(logits), torch.cat(labels)
```

The exact same procedure applies to a **U-Net**, **nnU-Net** or **SegFormer** —
only the model and the source of `images`/`targets` change. See
[Concepts](concepts.md#integrating-with-u-net-nnu-net-and-segformer).

## 3. Reliability diagram

[`examples/reliability_diagram.py`](https://github.com/fiducio-ai/Fiducio/blob/main/examples/reliability_diagram.py)

Fits `TemperatureScaling` on calibration cases and plots separate test cases.
Saves `outputs/reliability.png`; the [plotting guide](plots.md) shows the figure
and explains how to read its bars, dots and ECE.

```bash
python examples/reliability_diagram.py
```

## 4. Ensemble pooling

`python examples/ensemble_pooling.py` demonstrates two separate CMSap fits:
one on the mean probabilities (`input_type="probs"`) and one on the mean logits
(`input_type="logits"`). Members are pooled before fitting. Calibration and test
cases are disjoint. These are synthetic examples, not paper benchmark results.

## 5. Multi-rater calibration

Compare single-rater and MHR targets for temperature scaling and binary isotonic
calibration, using separate calibration, validation and test cases:

```bash
python examples/multi_rater_calibration.py
```

The script reports both MHR-ECE and mean-rater-ECE, and saves the positive-class
diagram to `outputs/multi_rater_reliability.png`. See the
[multi-rater guide](multi_rater.md) for the metric definitions and a figure.
