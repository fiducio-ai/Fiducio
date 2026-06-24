# Quickstart

## 1. Get model outputs and labels

You need two things from a **held-out, labelled calibration set** (see
[Concepts](concepts.md) for why it must be held out):

- `predictions`: the model output, shape `(B, C, *spatial)` with the class axis
  at dimension 1 (e.g. `(B, C, H, W)` in 2D, `(B, C, D, H, W)` in 3D);
- `targets`: integer labels, shape `(B, *spatial)`.

```python
import torch
import torch.nn.functional as F
from fiducio import TemperatureScaling, negative_log_likelihood, expected_calibration_error

# Stand-in for your model's outputs on the calibration set.
logits = torch.randn(8, 4, 64, 64)
labels = torch.randint(0, 4, (8, 64, 64))
```

## 2. Fit a calibrator

```python
calibrator = TemperatureScaling(input_type="logits")
calibrator.fit(logits, labels)
```

## 3. Apply it to new predictions

```python
new_logits = torch.randn(2, 4, 64, 64)
calibrated = calibrator.transform(new_logits)   # calibrated probabilities (B, C, H, W)
```

`transform` always returns **probabilities** that sum to 1 along the class axis.
`predict_proba` is an alias, and `fit_transform(logits, labels)` does both steps.
If you need calibrated **logits** instead, use `decision_function(new_logits)`
(`softmax` of its output equals `transform`).

## 4. Measure the effect

```python
raw = F.softmax(logits, dim=1)
print("NLL", negative_log_likelihood(raw, labels), "->",
      negative_log_likelihood(calibrator.transform(logits), labels))
print("ECE", expected_calibration_error(raw, labels), "->",
      expected_calibration_error(calibrator.transform(logits), labels))
```

## 5. Save and reload

```python
from fiducio import load_calibrator

calibrator.save("calibrator.pt")
calibrator = load_calibrator("calibrator.pt")   # loads on CPU by default
```

## Logits or probabilities?

Set `input_type` to match what you pass in:

```python
TemperatureScaling(input_type="logits")  # raw scores (default)
TemperatureScaling(input_type="probs")   # probabilities that sum to 1
```

## Accepted tensor shapes

| Use case | `predictions` | `targets` |
|----------|---------------|-----------|
| 2D segmentation | `(B, C, H, W)` | `(B, H, W)` |
| 3D segmentation | `(B, C, D, H, W)` | `(B, D, H, W)` |
| general n-D | `(B, C, *spatial)` | `(B, *spatial)` |
| per-pixel table | `(N, C)` | `(N,)` |

The class axis is **always** dimension 1. Binary segmentation is provided as two
channels (`C = 2`).
