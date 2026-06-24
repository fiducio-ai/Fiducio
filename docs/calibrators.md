# Calibrators

All calibrators share the same API and tensor convention (see
[Quickstart](quickstart.md)). They differ in how expressive the learned
transform is and in what they guarantee.

## Available calibrators

| Class | Alias | Parameters | Guarantee |
|-------|-------|------------|-----------|
| [`TemperatureScaling`](api/calibrators.md#fiducio.TemperatureScaling) | `TS` | one scalar `T` | preserves argmax **and** full order |
| [`EnsembleTemperatureScaling`](api/calibrators.md#fiducio.EnsembleTemperatureScaling) | `ETS` | `T` + 3 mixture weights | — |
| [`VectorScaling`](api/calibrators.md#fiducio.VectorScaling) | `VS` | per-class scale + bias | — |
| [`MatrixScaling`](api/calibrators.md#fiducio.MatrixScaling) | `MS` | full `C × C` matrix + bias | — |
| [`TranslationInvariantMatrixScaling`](api/calibrators.md#fiducio.TranslationInvariantMatrixScaling) | — | constrained `C × C` matrix + bias | invariant to logit translation |
| [`DirichletCalibration`](api/calibrators.md#fiducio.DirichletCalibration) | — | `C × C` matrix on log-probs | — |
| [`ClassConditionalMatrixScaling`](api/calibrators.md#fiducio.ClassConditionalMatrixScaling) | `CMS` | one affine map per top class | — |
| [`ArgmaxPreservingMatrixScaling`](api/calibrators.md#fiducio.ArgmaxPreservingMatrixScaling) | `CMSAP` | per-class margin map | preserves argmax |
| [`OrderPreservingMatrixScaling`](api/calibrators.md#fiducio.OrderPreservingMatrixScaling) | `CMSOP` | per-class gap map | preserves full order |

Short aliases (`TS`, `MS`, `CMS`, ...) are provided for convenience; the explicit
names are recommended in code that others will read.

## Choosing a calibrator

A practical decision guide:

- **Start with `TemperatureScaling`.** It has a single parameter, is very hard to
  overfit, never changes the segmentation, and is a strong baseline. If your only
  problem is over/under-confidence, it is often enough.
- **Need per-class flexibility?** Try `VectorScaling`, then `MatrixScaling`. Use
  the off-diagonal/bias regularisation (`lambda_reg`, `mu_reg`) when the number
  of classes is large relative to the calibration set, to avoid overfitting.
- **Want matrix scaling without the gauge redundancy?** Use
  `TranslationInvariantMatrixScaling`, which is invariant to adding a constant to
  all input logits.
- **Probability-space transform?** `DirichletCalibration` applies an affine map
  to log-probabilities (log-linear in probability space).
- **Calibration must not change the predicted labels?** Use
  `ArgmaxPreservingMatrixScaling`, which guarantees the calibrated argmax equals
  the uncalibrated one for every voxel — so Dice/IoU are unchanged.
- **Calibration must not change the class ranking at all?** Use
  `OrderPreservingMatrixScaling`, which preserves the full per-voxel ordering.
- **Want a flexible per-region map with no constraints?**
  `ClassConditionalMatrixScaling` fits one affine map per uncalibrated top class.

The class-conditional calibrators (`CMS`, `CMSAP`, `CMSOP`) optimise each expert
independently and regularise the affine map induced in the common logit space.
They are the most expressive option and benefit most from a reasonably sized
calibration set.

## Regularisation

`MatrixScaling`, `TranslationInvariantMatrixScaling`, `DirichletCalibration` and
the class-conditional calibrators accept:

- `lambda_reg` — L2 penalty on off-diagonal matrix entries (keeps the map close
  to the identity);
- `mu_reg` — L2 penalty on the bias.

`VectorScaling`'s `lambda_reg` pulls the scale vector towards 1. Start at `0` and
increase if the calibrator overfits a small calibration set.
