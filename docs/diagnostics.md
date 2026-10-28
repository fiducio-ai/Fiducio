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

  class       label %  pred %    Dice  class ECE  routed vox  routed ECE
  background    74.74   62.28  0.8496     0.1675       15305      0.0647
  pet           24.74   26.32  0.7039     0.0820        6469      0.1814
  edge           0.52   11.40  0.0641     0.1226        2802      0.7371

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
and `(B, *spatial)` integer labels, where `B` is the number of images or cases. Evaluating
`report` in a notebook cell shows the same text. `report.plot()` returns a 2×2
matplotlib figure (reliability diagram, confidence histogram, per-class ECE and
per-image Dice; needs `fiducio[plots]`), and `report.to_dict()` returns a
JSON-serializable dictionary.

## Reading the report

- **Intervals** are 95% bootstrap intervals over **images**, not voxels. Voxels of
  one image are strongly correlated, so the uncertainty comes from the number of
  images. With a single image there is no interval.
- The **ECE** interval is approximate. ECE has a finite-sample noise floor that
  resampling raises further, so for a well-calibrated model the interval can sit
  slightly above the estimate (in a simulation of 100 calibrated models of 24 images it
  missed the estimate 15 times; `reproducibility/diagnose_interval_check.py` reruns it).
  Read it as an order of magnitude.
- The **bias** line always prints, and calls the model over- or under-confident only
  when the interval of the voxel-weighted *mean confidence − accuracy* excludes 0. This
  gap has no noise floor, but with few images the intervals are slightly narrow: in the same simulation 9 of 100 calibrated models were flagged, against 5
  expected. Treat a weak flag with caution.
- **Dice** in the main table is pooled over all voxels of the set. The *Dice median /
  p10 / min* columns of the failure table are per image, over the images where the
  class is present, so the two differ. **Saturated** voxels have a confidence above
  0.999. The ECE uses `n_bins` (15) equal-width bins on the top-1 confidence.
- **label % / pred %** show class balance. **class ECE** compares the probability
  of the class with its observed frequency. **routed** columns restrict to the voxels
  whose argmax is the class: this is the split that class-conditional calibrators
  (CDC, CMSap, CMSop) fit one map on, so few routed voxels means a map fitted on
  little data.
- **Failures.** A class fails on an image when it is in the labels and its Dice is
  below `dice_failure_threshold` (0.1). An image fails when its mean Dice over the
  classes present is below it. Pass `background_class` to leave the background out
  of that mean: its Dice near 1 would otherwise hide a failure elsewhere. Without it,
  an image where the model predicts a single class can keep a mean Dice above the
  threshold and not count as failed; the per-class failure table still shows it.
  A class absent from both labels and prediction has an undefined Dice and is not a
  failure; a class predicted where it is absent is counted as *spurious*.
- The report gives the ECE with and without the failed images, since they enter the
  fit of any calibrator.

!!! note
    The report describes the calibration set, using its labels. It does not predict
    which new images will fail, and it does not choose a calibrator. Warning
    thresholds (`min_images`, `min_class_fraction`, `min_class_voxels`) are
    heuristics; adjust them to your data.

`logits` must have a spatial axis: a `(N, C)` table of voxels is rejected, since
`diagnose` works per image.

Cost is `2C + 1` reliability curves per image (global, per-class and routed), plus a
few passes for NLL, Brier and the confusion matrix, so it grows with the number of classes and the volume size. Soft targets (multi-rater
class distributions) are not supported.
