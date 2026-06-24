# Examples

Two runnable, CPU-only examples ship in the
[`examples/`](https://github.com/fiducio-ai/Fiducio/tree/main/examples) directory.
Neither downloads any model or dataset.

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
        logits.append(model(images))   # (B, C, H, W)
        labels.append(targets)         # (B, H, W)
    return torch.cat(logits), torch.cat(labels)
```

The exact same procedure applies to a **U-Net**, **nnU-Net** or **SegFormer** —
only the model and the source of `images`/`targets` change. See
[Concepts](concepts.md#integrating-with-u-net-nnu-net-and-segformer).
