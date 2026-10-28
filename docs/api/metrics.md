# API: Metrics & helpers

The metrics take probabilities of shape `(B, C, *spatial)` and integer labels of
shape `(B, *spatial)`, or floating class distributions with the same shape as
the probabilities. Use `mask` to exclude soft targets; `ignore_index` applies
to hard labels. See [multi-rater calibration](../multi_rater.md).

::: fiducio.negative_log_likelihood

::: fiducio.expected_calibration_error

::: fiducio.average_calibration_error

::: fiducio.unweighted_calibration_error

::: fiducio.brier_score

::: fiducio.reliability_curve

::: fiducio.ReliabilityCurve

## Diagnostics

See [reliability diagnostics](../diagnostics.md).

::: fiducio.diagnose

::: fiducio.ReliabilityReport

## Helpers

::: fiducio.mean_human_response

::: fiducio.two_channel_from_binary

::: fiducio.get_calibrator_class

::: fiducio.registered_ids
