"""Tests for input validation and error handling."""

from __future__ import annotations

import pytest
import torch

from conftest import synthetic_logits, to_probs
from fiducio import MatrixScaling, NotFittedError, TemperatureScaling


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
