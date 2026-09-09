# API: Metrics & helpers

The metrics take probabilities of shape `(B, C, *spatial)` and integer labels of
shape `(B, *spatial)`, with optional `mask` and `ignore_index`.

::: fiducio.negative_log_likelihood

::: fiducio.expected_calibration_error

::: fiducio.average_calibration_error

::: fiducio.brier_score

::: fiducio.reliability_curve

::: fiducio.ReliabilityCurve

## Helpers

::: fiducio.two_channel_from_binary

::: fiducio.get_calibrator_class

::: fiducio.registered_ids
