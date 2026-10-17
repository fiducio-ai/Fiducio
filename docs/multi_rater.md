# Multi-rater calibration to the mean human response

[TwinTrack: Post-hoc Multi-Rater Calibration for Medical Image Segmentation](https://arxiv.org/abs/2604.15950v2)
(Kirscher et al., MIDL 2026, short paper) motivates calibrating tumor probabilities
against the **mean human response (MHR)**: the fraction of annotators assigning
each class to a voxel. With two positive votes out of five, the binary target
is `[0.6, 0.4]`. It describes the available raters, rather than a unique anatomical
truth or a guaranteed response from another rater population.

Fiducio provides a binary isotonic map inspired by that approach and extends its
nine parametric calibrators to accept class distributions. The parametric
extension is not a claim that TwinTrack evaluated all nine methods.

## Data contract

- Predictions: `(B, C, *spatial)`, or a voxel table `(N, C)`.
- Hard targets: `(B, *spatial)`, integer labels, with `ignore_index` as before.
- Soft targets: `(B, C, *spatial)`, floating distributions, finite, in `[0,1]`
  and summing to 1 along classes at valid positions. The sum tolerance matches
  probability inputs (`1e-3`, widened for half precision); inputs are not silently
  normalized. Binary soft targets need both channels.
- Rater annotations: `(B, R, *spatial)`, integer labels with rater axis 1.
- Masks: `(B, *spatial)`, `True` for valid voxels. Use masks for soft targets;
  `ignore_index` only removes hard labels or missing individual votes.

```python
from fiducio import BinaryIsotonicCalibration, mean_human_response

# annotations: (B, R, *spatial), missing annotations encoded as -100
mhr, annotated = mean_human_response(annotations, num_classes=2)
cal = BinaryIsotonicCalibration(input_type="probs", device="cpu")
cal.fit(probabilities, mhr, mask=annotated)
calibrated = cal.transform(test_probabilities)
```

The helper returns float32 class proportions and a boolean mask on the input
device (NumPy inputs use CPU). Missing votes are excluded from the denominator.
Voxels without any annotation contain zero mass and **must be masked out**.
An empty rater axis yields zero mass and an all-false mask; fitting on no valid
voxels raises an error. There is no one-hot tensor per rater.

Each valid voxel contributes equally, even when annotation counts differ.
Thus fitting to MHR averages each voxel's available-rater loss, rather than
weighting more heavily annotated voxels more. Missingness can still bias the
target. There is no implicit rater-confidence or patient weighting.

## Parametric fits and validation

`TemperatureScaling`, ETS, vector/matrix/Dirichlet scaling and all three
class-conditional methods accept these distributions through the existing
`fit` API. They minimize cross-entropy against MHR. For a fixed number of raters,
this equals the mean cross-entropy against the individual labels, including its
gradient. Exact one-hot targets use the hard-label fitting kernel so their
optimizer path matches integer labels. Genuinely soft targets retain all classes.

ETS stage 2 computes cross-entropy **after** forming the mixture, using every
class probability for soft targets. It requires more memory than its hard-label
true-class-only path. Conditional experts still route voxels by the original
predicted class, never by the MHR.

Training and early-stopping validation can independently use hard or soft
targets. Validation must be separate from calibration and test cases; Adam's
existing early-stopping rules apply. Isotonic regression does not use gradient
optimization or early stopping and rejects validation data.

## Binary isotonic fitting

`BinaryIsotonicCalibration(n_bins=250)` learns a nondecreasing map for class 1:

1. Sort valid positive-class probabilities, grouping exact ties.
2. Make at most 250 approximately equal-mass bins. Cut ranks are rounded up
   to the end of a tie group, so equal scores never receive different mappings.
3. Fit bin target means with PAVA weighted by voxel counts.
4. Interpolate linearly between the fitted bin centers; outside the learned
   score range, return the nearest endpoint value.

One distinct score gives a constant map. Plateaus are allowed; this preserves
nondecreasing voxel-score order and can change a segmentation thresholded at 0.5.
It does not promise to preserve the per-voxel class argmax. Only `C=2` is supported;
there is no independent multiclass isotonic renormalization.

Coordinates, sums and PAVA use float64; public probabilities use float32.
Exact input zeros are retained during fitting. When the learned map outputs
exactly 0 or 1, those values are returned without clipping. An input score of
0 or 1 is still calibrated and need not remain 0 or 1; outside the learned
score range, it maps to the nearest fitted endpoint value.
`decision_function` returns log-probabilities and legitimately contains `-inf`
at zero probabilities. Saved coordinates remain finite and are validated for
monotonicity when loaded. Existing parametric checkpoints remain readable.

The binning, tie handling and interpolation above are explicit Fiducio
conventions. The paper specifies 250 equal-mass fitting bins; identity with
TwinTrack's original code has not been verified. Binned fitting also need not
equal an unbinned isotonic optimum on all individual voxels.

## Metrics: two different ECEs

Use `class_index=1` on `reliability_curve`, `expected_calibration_error`,
`unweighted_calibration_error` (also `average_calibration_error`) and
`plots.reliability_diagram` for positive-class probability versus target proportion.
Without `class_index`, the existing top-1 definition remains the default:
observed agreement is `q[argmax(p)]` for a soft target.

**ECE against MHR** bins predictions, then compares average probability with
average MHR. **Mean per-rater ECE** computes an ECE against each rater first,
then averages the results. They differ because absolute values precede the
rater mean. One voxel predicted at 0.5 with votes `[0,1]` has MHR-ECE 0 but
mean per-rater ECE 0.5. The plot annotates ECE against its supplied targets;
it does not infer mean per-rater ECE from an MHR tensor.

TwinTrack's appendices B and D specify 50 uniform evaluation bins and ECE
computed per rater then averaged. Retain individual annotations for this metric.
Use each rater's valid mask; average evaluable raters equally, report their
number and missing votes, and exclude undefined ECEs explicitly rather than
replacing them with zero. Different masks imply different evaluated populations.
If no rater is evaluable, report NaN. `ReliabilityCurve.bin_accuracy` stores the
observed proportion in class mode; empty bins retain zero statistics and ACE
uses only nonempty bins. Empty evaluated populations return NaN metrics.

NLL is `-mean(sum_c q_c log(p_c))`, with the existing `1e-12` floor.
Brier-MHR is `mean(sum_c (p_c-q_c)^2)`. Mean per-rater Brier adds
`mean(1-sum_c q_c^2)` on the same voxel population. Binary two-channel Brier is
twice the squared error on the positive channel. Neither is the TwinTrack
**volume CRPS**, which needs the original volume-evaluation protocol.

## Reproducible use

Run `uv run python examples/multi_rater_calibration.py` from the repository.
It creates synthetic independent case splits, compares no calibration,
single-rater calibration and MHR calibration for TS and isotonic regression,
and writes `outputs/multi_rater_reliability.png`. It needs the `plots` extra and does
not download patient data. It reports metrics pooled across test voxels, rather
than averaging per-case metrics. The diagram is against MHR; the separately
reported mean-rater ECE uses individual votes.

For real data, split by patient before selecting voxels; fix the ensemble
pooling, class merging and ROI before fitting. Record splits, raters, versions,
mask policy and any sampling seed with the checkpoint. Sampling to balance
foreground/background changes the target voxel distribution; calibration on that
sample does not guarantee calibration on complete volumes. All fits remain
in memory, including isotonic sorting (O(N log N) time, O(N) memory). A sampled
voxel table must therefore be prepared and described explicitly for large data.

Do not select the calibrator, fitting bins or ROI on the test set. MHR fitting
need not improve every test metric; isotonic squared error, parametric
cross-entropy and ECE are different objectives. Fiducio does not ship the nnU-Net
segmenter, volume CRPS, vascular metrics or a reproduction of CURVAS results.

## Reference

Tristan Kirscher, Alexandra Ertl, Klaus Maier-Hein, Xavier Coubez, Philippe Meyer
and Sylvain Faisan. *TwinTrack: Post-hoc Multi-Rater Calibration for Medical
Image Segmentation*. MIDL 2026, short paper.
[arXiv:2604.15950v2](https://arxiv.org/abs/2604.15950v2).
