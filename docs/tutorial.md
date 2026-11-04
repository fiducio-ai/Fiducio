# Tutorial: calibrating a segmenter on Oxford-IIIT Pets

This tutorial runs the whole workflow on a public dataset, from the splits to the
metrics on a separate test set. It is one script,
[`examples/tutorial_oxford_pets.py`](https://github.com/fiducio-ai/Fiducio/blob/main/examples/tutorial_oxford_pets.py),
which trains a small U-Net on CPU, so there are no pretrained weights to fetch.

The [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) dataset gives
each photo a trimap with three classes: pet, background and pet boundary.

```bash
pip install "fiducio[plots]"   # Pillow and matplotlib come with the plots extra
python examples/tutorial_oxford_pets.py
```

The first run downloads about 810 MB (SHA-256 verified) into
`~/.cache/fiducio/oxford-pets` and takes a few minutes on CPU. Use `--smoke` for
a seconds-long run on synthetic data with no download; CI runs that mode.

## 1. Three disjoint splits

The script resizes 1200 random images to 64×64 and splits them 50/25/25:

| Split | Used for |
| --- | --- |
| train | fitting the U-Net |
| calibration | fitting the calibrators |
| test | reporting metrics, never fitted on |

!!! warning
    Calibrate on images the network was not trained on, and report on images the
    calibrator did not see. Fitting on the training split calibrates an over-fitted
    model; fitting on the test split leaks the answer and flatters the metrics.
    The split here is per image. With patient or volume data, split per patient.

## 2. Collect logits

After training, stack the logits as `(B, C, H, W)` and labels as `(B, H, W)`,
the layout Fiducio expects:

```python
cal_logits = logits_of(model, x[splits["calibration"]])
test_logits = logits_of(model, x[splits["test"]])
```

## 3. Diagnose the model on the calibration split

Before choosing a calibrator, look at the model's reliability on the calibration
split with [`diagnose`](diagnostics.md). The background is class 1 in these
labels, so it is excluded from the per-image mean Dice:

```python
from fiducio import diagnose

cal_labels = y[splits["calibration"]]
print(
    diagnose(
        cal_logits, cal_labels, class_names=["pet", "background", "boundary"], background_class=1
    )
)
```

```text
Reliability report: 300 images, 1228800 valid voxels, 3 classes
  ECE   0.0130 [0.0069, 0.0202]
  NLL   0.4791 [0.4539, 0.5045]
  Brier 0.2668 [0.2535, 0.2808]
  Bias  overconfident (confidence - accuracy = 0.0130 [0.0066, 0.0202])
  Saturated voxels (confidence > 0.999): 4.7%

  class       label %  pred %    Dice  class ECE  routed vox  routed ECE
  pet           28.45   26.94  0.7701     0.0265      330991      0.0386
  background    59.81   64.85  0.8908     0.0205      796859      0.0222
  boundary      11.74    8.22  0.4517     0.0108      100950      0.0530

Failed images (mean Dice < 0.1): 0/300
```

This is an excerpt; [reliability diagnostics](diagnostics.md) describes every
field. On this run the model is slightly over-confident (ECE 0.013) and no image
fails, so there is nothing to remove from the calibration set. Image 187, the
lowest, still has a mean Dice of 0.14 and is the first one to inspect. No class
is rare, and every class is predicted on over 100 000 voxels, so a
class-conditional calibrator has enough data per class. The two warnings about a
class predicted where it is absent concern one image each, the kind of case this
report is meant to surface.

## 4. Fit on the calibration split only

```python
ts = TemperatureScaling(device="cpu").fit(cal_logits, cal_labels)
ms = MatrixScaling(lambda_reg=1e-2, device="cpu").fit(cal_logits, cal_labels)
cmsap = CMSap(device="cpu").fit(cal_logits, cal_labels)
cmsop = CMSop(device="cpu").fit(cal_logits, cal_labels)
```

## 5. Evaluate on the test split

```python
raw = F.softmax(test_logits, dim=1)
calibrated = ts.transform(test_logits)
expected_calibration_error(calibrated, test_labels)
```

One run on CPU (1200 images, 40 epochs, seed 0) gave:

| | NLL | ECE | Brier | Dice |
| --- | --- | --- | --- | --- |
| uncalibrated | 0.4952 | 0.0144 | 0.2734 | 0.6947 |
| `TemperatureScaling` | 0.4929 | 0.0039 | 0.2731 | 0.6947 |
| `MatrixScaling` | 0.4902 | 0.0043 | 0.2714 | 0.6926 |
| `CMSap` | 0.4916 | 0.0046 | 0.2724 | 0.6947 |
| `CMSop` | 0.4929 | 0.0051 | 0.2726 | 0.6947 |

Your numbers will differ slightly with the PyTorch version and thread count. Read
them as follows:

- This small network is already close to calibrated, so the gains are modest. ECE
  falls by nearly a factor of four; NLL barely moves.
- `MatrixScaling` is the only method here that can change the predicted class, so
  it is the only one whose Dice moves (0.6947 to 0.6926). `TemperatureScaling`,
  `CMSap` and `CMSop` preserve the argmax (or the class ranking), so their Dice
  is unchanged.
- No flexible calibrator beats temperature scaling on ECE here. With three classes
  and a model this close to calibrated, a richer map has little to correct. Reach
  for `MatrixScaling` or the class-conditional `CMSap` / `CMSop` when
  miscalibration depends on the class, and check Dice or volume metrics before
  adopting them.
- ECE here is top-1, pixel-pooled ECE with 15 bins, not the per-image paper
  protocol (see `expected_calibration_error`).

The script also writes `outputs/tutorial_reliability.png`, a before/after
[reliability diagram](plots.md) on the test split.
