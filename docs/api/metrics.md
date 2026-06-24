# API: Metrics

All metrics take probabilities of shape `(B, C, *spatial)` and integer labels of
shape `(B, *spatial)`, with optional `mask` and `ignore_index`, and return a
Python float.

::: fiducio.negative_log_likelihood

::: fiducio.expected_calibration_error

::: fiducio.brier_score
