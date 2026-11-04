# Diagnosing reliability before calibration

`fiducio.diagnose` summarises how reliable a segmenter is on a calibration set,
**before** you pick a calibrator: class balance, global and per-class calibration,
and the images or classes where the model fails (Dice close to zero).

```python
import torch
import fiducio

torch.manual_seed(0)
probs = torch.tensor([0.75, 0.245, 0.005])
labels = torch.multinomial(probs, 24 * 32 * 32, replacement=True).view(24, 32, 32)
onehot = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
logits = (onehot * 1.5 + torch.randn(24, 3, 32, 32)) * 3  # an over-confident model
logits[3] = 0.0
logits[3, 0] = 10.0  # on image 3 the model predicts the background everywhere

report = fiducio.diagnose(
    logits, labels, class_names=["background", "pet", "edge"], background_class=0
)
print(report)
```

```text
Reliability report: 24 images, 24576 valid voxels, 3 classes
  ECE   0.1164 [0.1077, 0.1346]
  NLL   0.7628 [0.6751, 1.0075]
  Brier 0.3569 [0.3462, 0.3788]
  Bias  overconfident (confidence - accuracy = 0.1164 [0.1077, 0.1346])
  Saturated voxels (confidence > 0.999): 17.6%
  First/last bin mass 0.0%/58.2%, gap n/a/0.0972, ECE contribution 0.0566

  class       label %  pred %    Dice  class ECE  routed vox  routed ECE
  background    74.74   62.28  0.8496     0.1675       15305      0.0647
  pet           24.74   26.32  0.7039     0.0820        6469      0.1814
  edge           0.52   11.40  0.0641     0.1226        2802      0.7371

  Per-class 95% intervals (bootstrap over images):
  class                     class ECE               routed ECE                     Dice
  background  0.1675 [0.1609, 0.1788]  0.0647 [0.0483, 0.1036]  0.8496 [0.8462, 0.8529]
  pet         0.0820 [0.0730, 0.1058]  0.1814 [0.1724, 0.1907]  0.7039 [0.6554, 0.7225]
  edge        0.1226 [0.1072, 0.1298]  0.7371 [0.7309, 0.7441]  0.0641 [0.0542, 0.0734]

Failed images (mean Dice < 0.1): 1/24
  ECE with all images 0.1164, without the failed ones 0.1100; mean confidence 0.9999 on failed vs 0.8768 on the rest
  Per class, a failure is Dice < 0.1 on an image where it is present:
  class       present  failed  spurious  Dice median     p10     min
  background       24       0         0       0.8495  0.8386  0.8344
  pet              24       1         0       0.7152  0.6910  0.0000
  edge             24      21         0       0.0611  0.0223  0.0000
  Lowest-Dice images:
    3: mean Dice 0.0000, worst class pet (0.0000)
    23: mean Dice 0.3609, worst class edge (0.0317)
    20: mean Dice 0.3648, worst class edge (0.0167)
    15: mean Dice 0.3663, worst class edge (0.0342)
    1: mean Dice 0.3721, worst class edge (0.0345)

Warnings
  - class 'edge' is rare (129 voxels, 0.52% of labels): its estimates are noisy
  - class 'edge' fails (Dice < 0.1) on 21/24 of the images where it is present
  - 1 image(s) have a mean Dice below 0.1 and enter the fit; ECE is 0.1164 with them and 0.1100 without
  - the model is overconfident on this set (confidence - accuracy = 0.1164)
```

The input is `(B, C, *spatial)` logits (or probabilities with `input_type="probs"`)
and `(B, *spatial)` integer labels; `B` is the number of images or cases. In a
notebook, `report` displays the same text. `report.plot()` returns a 2×2
matplotlib figure (reliability diagram, confidence histogram, per-class ECE and
per-image Dice; needs `fiducio[plots]`), and `report.to_dict()` returns a
JSON-serializable dictionary with NaN as `null`.

## Streaming over a model and DataLoader

Pass a model and a DataLoader instead of materialised tensors:

```python
from torch.utils.data import DataLoader

loader = DataLoader(calibration_dataset, batch_size=8)
report = fiducio.diagnose(model, loader, device="cuda", class_names=["background", "pet"])
```

Each batch is `(inputs, labels)` or `(inputs, labels, mask)`. `inputs` are
whatever the model accepts (tensors, or lists/tuples/dicts of tensors), moved to
`device`; the model must return `(B, C, *spatial)` logits (or probabilities with
`input_type="probs"`). It runs under `torch.no_grad()` with its `training` flag
unchanged, so call `model.eval()` yourself for inference-mode dropout or
batchnorm. `device` defaults to the first parameter's device. Only per-image
statistics are accumulated, so the report streams over data that does not fit in
memory. Masks come from the batch; the `mask` argument applies to tensor input.

## Soft targets (MHR)

`labels` may be floating class distributions `(B, C, *spatial)`, as produced by
`mean_human_response`. Top-1 and routed ECE compare the top-1 confidence with
`q[argmax(p)]`; per-class ECE compares `p_c` with `q_c`; NLL and Brier keep the
soft-target definitions from `fiducio.metrics`. `mask` selects soft targets;
`ignore_index` applies to hard labels only.

Hard-label Dice is undefined for class distributions, so `image_dice`, every
Dice field and the failure statistics are NaN (counts are 0) and the report
warns. Class balance still reports the expected label mass `sum(q_c)`, which may
be fractional. No soft Dice substitute is invented.

## Reading the report

- **Intervals** are 95% bootstrap intervals over **images**, not voxels, so their
  width comes from the number of images. They need at least two images and two
  resamples (`n_boot=0` disables them, `n_boot=1` yields none). Class ECE
  intervals need at least two images; routed ECE and Dice intervals additionally
  need the class routed or present on at least two images, else they are NaN.
- The **ECE** interval is approximate. ECE has a finite-sample noise floor that
  resampling raises further, so for a well-calibrated model it can sit slightly
  above the estimate (in a simulation of 100 calibrated models of 24 images it
  missed the estimate 15 times; `reproducibility/diagnose_interval_check.py`
  reruns it). Read it as an order of magnitude.
- The **bias** line always prints, and calls the model over- or under-confident
  only when the interval of the voxel-weighted *mean confidence − accuracy*
  excludes 0. That gap has no noise floor, but with few images its intervals are
  slightly narrow: the same simulation flagged 9 of 100 calibrated models,
  against 5 expected. Treat a weak flag with caution.
- **Dice** in the main table is pooled over all voxels. The failure table's Dice
  median, p10 and min are per image, over the images where the class is present.
  **Saturated** voxels have confidence above 0.999. ECE uses `n_bins` (default
  15) equal-width bins on the top-1 confidence.
- **First/last bin** describes the `[0, 1/n_bins]` and
  `[(n_bins-1)/n_bins, 1]` confidence bins: the share of voxels in each, their
  mean confidence minus accuracy (positive means over-confident), and their exact
  contribution to the reported `ece`. The `ece` definition is unchanged.
- **label % / pred %** show class balance. **class ECE** compares the probability
  of the class with its observed frequency. **routed** columns restrict to the
  voxels whose argmax is the class, the split that class-conditional calibrators
  (CDC, CMSap, CMSop) fit one map on, so few routed voxels mean a map fitted on
  little data.
- **Failures.** A class fails on an image when it is in the labels and its Dice
  is below `dice_failure_threshold` (default 0.1). An image fails when its mean
  Dice over the classes present is below it. Pass `background_class` to leave the
  background out of that mean, since Dice near 1 would hide a failure elsewhere.
  Without it, an image where the model predicts a single class can keep a mean
  Dice above the threshold; the per-class failure table still shows it. A class
  absent from both labels and prediction has undefined Dice and is not a failure;
  a class predicted where it is absent is counted as *spurious*. If
  `background_class` leaves no class present in the labels, the per-image mean
  Dice is undefined: the report warns and counts no image as failed. The report
  gives the ECE with and without the failed images, since they enter any
  calibrator's fit.

!!! note
    The report describes the calibration set, using its labels. It does not predict
    which new images will fail, and it does not choose a calibrator. The warning
    thresholds (`min_images`, `min_class_fraction`, `min_class_voxels`) are
    heuristics; adjust them to your data. `min_images` and `min_class_voxels`
    must be non-negative integers, `min_class_fraction` and
    `dice_failure_threshold` must lie in `[0, 1]`.

`logits` must have a spatial axis: a `(N, C)` voxel table is rejected, since
`diagnose` works per image.

The accumulator keeps only `O(n_images)` summaries, not the voxels, so the
bootstrap resamples summaries; with a model and DataLoader the forward pass
dominates. See the [`diagnose`](api/metrics.md) reference for cost details.
