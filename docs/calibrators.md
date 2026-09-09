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
| [`TranslationInvariantMatrixScaling`](api/calibrators.md#fiducio.TranslationInvariantMatrixScaling) | `MSc` | constrained `C × C` matrix + bias | invariant to logit translation |
| [`DirichletCalibration`](api/calibrators.md#fiducio.DirichletCalibration) | `DC` | `C × C` matrix on log-probs | — |
| [`ClassConditionalMatrixScaling`](api/calibrators.md#fiducio.ClassConditionalMatrixScaling) | `CMS` / `CDC` | one affine map per top class | — |
| [`ArgmaxPreservingMatrixScaling`](api/calibrators.md#fiducio.ArgmaxPreservingMatrixScaling) | `CMSAP` / `CMSap` | per-class margin map | preserves argmax |
| [`OrderPreservingMatrixScaling`](api/calibrators.md#fiducio.OrderPreservingMatrixScaling) | `CMSOP` / `CMSop` | per-class gap map | preserves full order |

Short aliases (`TS`, `MS`, `CMS`, ...) are provided for convenience; the explicit
names are recommended in code that others will read.

## Paper method mapping

The following paper methods are implemented here. LTS is not included in this
release; this table is not the complete experimental method inventory.

| Paper method | Class | Alias |
|--------------|-------|-------|
| TS | [`TemperatureScaling`](api/calibrators.md#fiducio.TemperatureScaling) | `TS` |
| MS | [`MatrixScaling`](api/calibrators.md#fiducio.MatrixScaling) | `MS` |
| MSc | [`TranslationInvariantMatrixScaling`](api/calibrators.md#fiducio.TranslationInvariantMatrixScaling) | `MSc` |
| CDC | [`ClassConditionalMatrixScaling`](api/calibrators.md#fiducio.ClassConditionalMatrixScaling) | `CDC` |
| CMSap | [`ArgmaxPreservingMatrixScaling`](api/calibrators.md#fiducio.ArgmaxPreservingMatrixScaling) | `CMSap` |
| CMSop | [`OrderPreservingMatrixScaling`](api/calibrators.md#fiducio.OrderPreservingMatrixScaling) | `CMSop` |

CDC, CMSap and CMSop are class-conditional calibrators: each fits one affine
map per uncalibrated top class. By default (`independent_experts=False`,
matching the paper) all experts are optimized **jointly** — a single optimizer
minimizes one cross-entropy loss over every voxel at once (each voxel passing
through its own expert's map) — with `lambda_reg` / `mu_reg` regularizing the
affine map induced in the common *logit* space (not the raw per-expert
parameters), averaged across experts. This logit-space regularization is what
makes the penalty meaningful for CMSap/CMSop, whose raw parameters are
non-negative margins/gaps rather than matrix entries — a naive penalty on the
raw parameters would not correspond to "close to the identity" in the space
the map actually operates in.

Set `independent_experts=True` to instead fit each expert in its own
optimization loop, using only the voxels routed to it. This avoids an expert
with few routed voxels being drowned out by the joint loss, at the cost of
`C` times the optimizer work (one full `max_iter`-step fit per class instead
of one shared fit across all classes).

## Choosing a calibrator

A practical decision guide:

- **Start with `TemperatureScaling`.** It has a single parameter, is very hard to
  overfit, never changes the segmentation, and is a strong baseline. If your only
  problem is over/under-confidence, it is often enough.
- **Need per-class flexibility?** Try `VectorScaling`, then `MatrixScaling`. Use
  the off-diagonal/bias regularisation (`lambda_reg`, `mu_reg`) when the number
  of classes is large relative to the calibration set, to avoid overfitting.
- **Want matrix scaling invariant to logit shifts?** Use
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

The class-conditional calibrators (`CDC`/`CMS`, `CMSap`/`CMSAP`, `CMSop`/`CMSOP`)
jointly optimise all experts in a single loss and regularise the affine map
induced in the common logit space. They are the most expressive option and
benefit most from a reasonably sized calibration set.

## Regularisation

`MatrixScaling`, `TranslationInvariantMatrixScaling`, `DirichletCalibration` and
the class-conditional calibrators accept:

- `lambda_reg` — L2 penalty on off-diagonal matrix entries (diagonal entries
  are not directly penalized);
- `mu_reg` — L2 penalty on the bias.

`VectorScaling`'s `lambda_reg` pulls the scale vector towards 1. Start at `0` and
increase if the calibrator overfits a small calibration set.

MSc optimizes `C-1` free columns and a learned common row sum; the final column
is reconstructed before applying the off-diagonal penalty. Initialization is
the identity. This differs from the older pre-release row-centering approach.

The guarantees above apply to valid unmasked voxels, subject to floating-point
precision. Exact ties follow PyTorch selection/sorting behavior; preserving
strict ranks does not imply preserving every set of tied scores.
