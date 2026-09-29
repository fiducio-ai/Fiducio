"""Tests for input validation and error handling."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import synthetic_logits, to_probs
from fiducio import (
    MatrixScaling,
    NotFittedError,
    TemperatureScaling,
    negative_log_likelihood,
    reliability_curve,
    two_channel_from_binary,
)


def test_transform_before_fit_raises():
    cal = TemperatureScaling(device="cpu")
    with pytest.raises(NotFittedError):
        cal.transform(torch.randn(2, 3, 4, 4))


def test_single_class_rejected():
    cal = TemperatureScaling(device="cpu")
    with pytest.raises(ValueError, match="at least 2 classes"):
        cal.fit(torch.randn(2, 1, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long))


def test_shape_mismatch_targets():
    cal = TemperatureScaling(device="cpu")
    logits = torch.randn(2, 3, 4, 4)
    with pytest.raises(ValueError, match="targets must have shape"):
        cal.fit(logits, torch.zeros(2, 5, 5, dtype=torch.long))


def test_fit_without_targets_raises():
    cal = TemperatureScaling(device="cpu")
    with pytest.raises(ValueError, match="targets are required"):
        cal.fit(torch.randn(2, 3, 4, 4), None)


def test_mask_shape_mismatch_rejected():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=25)
    bad_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask must have shape"):
        cal.fit(logits, labels, mask=bad_mask)


def test_class_count_mismatch_on_transform():
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=20)
    cal = MatrixScaling(device="cpu").fit(logits, labels)
    with pytest.raises(ValueError, match="fitted for C=3"):
        cal.transform(torch.randn(2, 4, 4, 4))


def test_probs_not_summing_to_one_rejected():
    cal = TemperatureScaling(input_type="probs", device="cpu")
    bad = torch.full((2, 3, 4, 4), 0.5)  # sums to 1.5
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="sum to 1"):
        cal.fit(bad, labels)


def test_non_finite_predictions_rejected():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=21)
    logits = logits.clone()
    logits[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        cal.fit(logits, labels)


def test_label_out_of_range_rejected():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=22)
    labels = labels.clone()
    labels[0, 0, 0] = 9
    with pytest.raises(ValueError, match=r"valid labels must lie"):
        cal.fit(logits, labels)


def test_all_masked_out_raises():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=23)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="no valid voxels"):
        cal.fit(logits, labels, mask=mask)


def test_bad_input_type_rejected():
    with pytest.raises(ValueError, match="input_type"):
        TemperatureScaling(input_type="scores", device="cpu")


def test_numpy_inputs_accepted():

    logits, labels = synthetic_logits((2, 3, 5, 5), seed=24)
    cal = TemperatureScaling(device="cpu")
    out = cal.fit_transform(logits.numpy(), labels.numpy())
    assert out.shape == to_probs(logits).shape


def test_masked_transform_output_is_valid_probs_input():
    logits, labels = synthetic_logits((2, 3, 6, 6), seed=26)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    mask[:, :2] = False
    cal = MatrixScaling(device="cpu").fit(logits, labels, mask=mask)
    probs = cal.transform(logits, mask=mask)
    valid = probs.sum(dim=1)[mask]
    assert torch.allclose(valid, torch.ones_like(valid), atol=1e-4)
    chained = TemperatureScaling(input_type="probs", device="cpu").fit(probs, labels, mask=mask)
    assert chained.transform(probs, mask=mask).shape == probs.shape
    assert math.isfinite(negative_log_likelihood(probs, labels, mask=mask))


def test_zero_rows_are_rejected_without_mask():
    probs = torch.zeros(2, 3, 4, 4)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="sum to 1"):
        negative_log_likelihood(probs, labels)


def test_non_tensor_predictions_and_targets_rejected():
    with pytest.raises(TypeError, match="torch.Tensor"):
        TemperatureScaling(device="cpu").fit([[1.0, 2.0]], [[0]])
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=27)
    with pytest.raises(TypeError, match="torch.Tensor"):
        TemperatureScaling(device="cpu").fit(logits, [[0]])


def test_invalid_n_bins_rejected():
    probs = to_probs(synthetic_logits((2, 3, 4, 4), seed=28)[0])
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="n_bins"):
        reliability_curve(probs, labels, n_bins=0)


def test_two_channel_from_binary_rejects_bad_input_type():
    with pytest.raises(ValueError, match="input_type"):
        two_channel_from_binary(torch.randn(2, 1, 4, 4), input_type="scores")
