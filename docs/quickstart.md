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
torch.manual_seed(0)
logits = torch.randn(8, 4, 64, 64)
labels = torch.randint(0, 4, (8, 64, 64))
```

## 2. Fit a calibrator

```python
calibrator = TemperatureScaling(input_type="logits", device="cpu")
calibrator.fit(logits, labels)
```

## 3. Apply it to new predictions

```python
new_logits = torch.randn(2, 4, 64, 64)
calibrated = calibrator.transform(new_logits)  # calibrated probabilities (B, C, H, W)
```

`transform` returns **probabilities** that sum to 1 at valid voxels.
`predict_proba` is an alias, and `fit_transform(logits, labels)` does both steps.
Evaluate on separate data rather than the output of `fit_transform`.
If you need calibrated **logits** instead, use `decision_function(new_logits)`
(`softmax` of its output equals `transform`).

## 4. Measure the effect

```python
# Separate test data, never used for fitting or model selection:
test_logits = torch.randn(8, 4, 64, 64)
test_labels = torch.randint(0, 4, (8, 64, 64))
raw = F.softmax(test_logits, dim=1)
calibrated_test = calibrator.transform(test_logits)
print(
    "NLL",
    negative_log_likelihood(raw, test_labels),
    "->",
    negative_log_likelihood(calibrated_test, test_labels),
)
print(
    "ECE",
    expected_calibration_error(raw, test_labels),
    "->",
    expected_calibration_error(calibrated_test, test_labels),
)
```

Metrics take probabilities, not logits. These random tensors show the API;
improvement on unseen data is not guaranteed. To inspect where predictions are
over- or under-confident, see the [reliability-diagram examples](plots.md).

## 5. Save and reload

```python
from fiducio import load_calibrator

calibrator.save("calibrator.pt")
calibrator = load_calibrator("calibrator.pt")  # loads on CPU by default
```

## Mask padding or select a region

Pass a boolean mask with the same shape as the labels. Use it consistently
when fitting, transforming and evaluating:

```python
fit_mask = torch.ones_like(labels, dtype=torch.bool)
fit_mask[:, :4] = False  # example padding rows
calibrator.fit(logits, labels, mask=fit_mask)

test_mask = torch.ones_like(test_labels, dtype=torch.bool)
test_mask[:, :4] = False
calibrated_test = calibrator.transform(test_logits, mask=test_mask)
ece = expected_calibration_error(calibrated_test, test_labels, mask=test_mask)
```

Masked output probabilities are all zero, so pass the same mask to metrics.
For missing hard labels, `ignore_index=-100` is available as well. For soft
targets, use a mask; see [multi-rater calibration](multi_rater.md).

## Logits or probabilities?

Set `input_type` to match what you pass in:

```python
TemperatureScaling(input_type="logits")  # raw scores (default)
TemperatureScaling(input_type="probs")  # probabilities that sum to 1
```

## Accepted tensor shapes

| Use case | `predictions` | `targets` |
|----------|---------------|-----------|
| 2D segmentation | `(B, C, H, W)` | `(B, H, W)` |
| 3D segmentation | `(B, C, D, H, W)` | `(B, D, H, W)` |
| general n-D | `(B, C, *spatial)` | `(B, *spatial)` |
| per-pixel table | `(N, C)` | `(N,)` |

The class axis is **always** dimension 1. Binary segmentation is provided as two
channels (`C = 2`); see [the single-channel conversion](concepts.md#binary-versus-multiclass).
Floating class-distribution targets, including MHR, have the same shape as predictions.
