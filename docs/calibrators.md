# Calibrators

All calibrators share the same API and tensor convention (see
[Quickstart](quickstart.md)). They differ in how expressive the learned
transform is and in what they guarantee.

## Available calibrators

| Class | Alias | Parameters | Translation-invariant | Decision preservation |
|-------|-------|------------|:---------------------:|-----------------------|
| [`TemperatureScaling`](api/calibrators.md#fiducio.TemperatureScaling) | `TS` | one scalar `T` | yes | argmax **and** full order |
| [`EnsembleTemperatureScaling`](api/calibrators.md#fiducio.EnsembleTemperatureScaling) | `ETS` | `T` + 3 mixture weights | yes | argmax **and** full order |
| [`VectorScaling`](api/calibrators.md#fiducio.VectorScaling) | `VS` | per-class scale + bias | no | — |
| [`MatrixScaling`](api/calibrators.md#fiducio.MatrixScaling) | `MS` | full `C × C` matrix + bias | no | — |
| [`TranslationInvariantMatrixScaling`](api/calibrators.md#fiducio.TranslationInvariantMatrixScaling) | `MSc` | constrained `C × C` matrix + bias | yes | — |
| [`DirichletCalibration`](api/calibrators.md#fiducio.DirichletCalibration) | `DC` | `C × C` matrix on log-probs | yes | — |
| [`ClassConditionalMatrixScaling`](api/calibrators.md#fiducio.ClassConditionalMatrixScaling) | `CMS` / `CDC` | one affine map per top class | yes | — |
| [`ArgmaxPreservingMatrixScaling`](api/calibrators.md#fiducio.ArgmaxPreservingMatrixScaling) | `CMSAP` / `CMSap` | per-class margin map | yes | argmax |
| [`OrderPreservingMatrixScaling`](api/calibrators.md#fiducio.OrderPreservingMatrixScaling) | `CMSOP` / `CMSop` | per-class gap map | yes | argmax **and** full order |
| [`BinaryIsotonicCalibration`](api/calibrators.md#fiducio.BinaryIsotonicCalibration) | — | up to `n_bins` fitted points | yes | — |

Translation-invariant means the output is unchanged when the same constant is
added to every input logit of a voxel. Short aliases (`TS`, `MS`, `CMS`, ...) are
provided for convenience; explicit names are easier to read.

The table describes logit inputs. With `input_type="probs"`, an upstream logit
shift has already been removed by softmax.

The `ETS` guarantee holds in exact arithmetic — a non-negative weighted sum of
monotone maps preserves the order — but not bit-for-bit in `float32` once the
mixture collapses towards the uniform component: ties and near-ties (top-2 gaps
below the float32 resolution) can then resolve differently. Treat `ETS` as
order-preserving up to floating-point rounding.

## Binary isotonic calibration and MHR

[`BinaryIsotonicCalibration`](api/calibrators.md#fiducio.BinaryIsotonicCalibration)
fits a nondecreasing positive-class map with weighted PAVA on at most `n_bins`
bins (default 250).
It accepts two-channel probabilities or logits and hard or MHR targets. This
approach is inspired by [TwinTrack](https://arxiv.org/abs/2604.15950v2)
(Kirscher et al., MIDL 2026, short paper). Tie handling and interpolation are
explicit Fiducio conventions; exact reproduction has not been verified.
It can change a segmentation at threshold 0.5, and has no optimizer or early
stopping. The nine parametric calibrators also accept MHR distributions.
See [multi-rater calibration](multi_rater.md) for details and metric conventions.

Choose fitting `n_bins` on validation cases held out from the fit. The fitting
bins and the evaluation bins are separate settings:

```python
from fiducio import BinaryIsotonicCalibration, expected_calibration_error

# cal_* and val_* come from disjoint case splits.
candidates = []
for n_bins in (16, 64, 250):  # fitting bins
    cal = BinaryIsotonicCalibration(input_type="probs", n_bins=n_bins, device="cpu").fit(
        cal_probabilities, cal_mhr, mask=cal_annotated
    )
    ece = expected_calibration_error(
        cal.transform(val_probabilities, mask=val_annotated),
        val_mhr,
        mask=val_annotated,
        class_index=1,
        n_bins=50,  # evaluation bins
    )
    candidates.append((ece, cal))
validation_ece, calibrator = min(candidates, key=lambda item: item[0])
```

Evaluate the selected calibrator once on separate test cases. Ties can reduce
the actual number of fitting bins. A small high-confidence tail may be merged
into the final bin, so extreme scores can be pulled towards that bin's mean.

Decision and ordering guarantees in the table hold in exact arithmetic and
are subject to floating-point ties and rounding.

## Paper method mapping

The following methods of [*Rethinking Post-Hoc Calibration in Semantic Segmentation*](https://openreview.net/forum?id=xwNoSNxgxV) (Kirscher et al., Transactions on Machine Learning Research, 2026; preprint [arXiv:2607.01902](https://arxiv.org/abs/2607.01902)) are implemented here. LTS is not included in this
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
  overfit, preserves the argmax up to floating-point ties and rounding, and is a
  useful baseline for over- or under-confidence.
- **Need per-class flexibility?** Try `VectorScaling`, then `MatrixScaling`. Use
  the off-diagonal/bias regularisation (`lambda_reg`, `mu_reg`) when the number
  of classes is large relative to the calibration set, to avoid overfitting.
- **Want matrix scaling invariant to logit shifts?** Use
  `TranslationInvariantMatrixScaling`, which is invariant to adding a constant to
  all input logits.
- **Probability-space transform?** `DirichletCalibration` applies an affine map
  to log-probabilities (log-linear in probability space).
- **Calibration must not change the predicted labels?** Use
  `ArgmaxPreservingMatrixScaling`: the calibrated argmax equals the uncalibrated
  one at every valid voxel up to floating-point ties, so an argmax segmentation,
  and Dice/IoU computed from it, is unchanged. Thresholded or post-processed
  segmentations are not covered.
- **Calibration must not change the class ranking at all?** Use
  `OrderPreservingMatrixScaling`, which preserves the full per-voxel ordering.
- **Want a flexible per-region map with no constraints?**
  `ClassConditionalMatrixScaling` fits one affine map per uncalibrated top class.
- **Only two classes?** `MSc` is very restrictive for `C = 2`; the paper finds
  it significantly worse on BA-ECE. Prefer a less constrained calibrator.
- **Binary problem with a flexible monotone map?** `BinaryIsotonicCalibration`
  fits a nondecreasing, piecewise-linear map on equal-mass bins (see *Binary isotonic
  calibration and MHR* above). Unlike the parametric maps it can pull extreme
  scores toward the last bin mean when the top-confidence voxels weigh less
  than one bin; tune `n_bins` for your data.

The class-conditional calibrators (`CDC`/`CMS`, `CMSap`/`CMSAP`, `CMSop`/`CMSOP`)
jointly optimise all experts in a single loss and regularise the affine map
induced in the common logit space. They are the most expressive option and
benefit most from a reasonably sized calibration set. Their parameters grow as
`O(C^3)`, so they suit low-to-moderate class counts, and
`independent_experts=True` costs one full optimization per class.

## Regularisation

`MatrixScaling`, `TranslationInvariantMatrixScaling`, `DirichletCalibration` and
the class-conditional calibrators accept:

- `lambda_reg` — L2 penalty on off-diagonal matrix entries (diagonal entries
  are not directly penalized);
- `mu_reg` — L2 penalty on the bias.

The penalty is `lambda_reg * mean(off-diagonal W^2) + mu_reg * mean(b^2)` (the
mean is taken over experts for the class-conditional family). Both terms are
means, matching Kull et al.'s (2019) normalized ODIR (their code with
`reg_norm=True`); values from un-normalized settings are not transferable.

`VectorScaling`'s `lambda_reg` pulls the scale vector towards 1. Start at `0` and
increase if the calibrator overfits a small calibration set.

MSc optimizes `C-1` free columns and a learned common row sum; the final column
is reconstructed before applying the off-diagonal penalty. Initialization is
the identity. This differs from the older pre-release row-centering approach.

The guarantees above apply to valid unmasked voxels, subject to floating-point
precision. Exact ties follow PyTorch selection/sorting behavior; preserving
strict ranks does not imply preserving every set of tied scores.
