"""Tests for the secondary public API: decision_function, repr, factory,
two_channel_from_binary, reliability_curve and version handling."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn.functional as F

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits, to_probs
from fiducio import (
    MatrixScaling,
    NotFittedError,
    ReliabilityCurve,
    get_calibrator_class,
    load_calibrator,
    registered_ids,
    reliability_curve,
    two_channel_from_binary,
)


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_decision_function_matches_transform(calibrator_id):
    logits, labels = synthetic_logits((3, 3, 6, 6), seed=60)
    cal = make_calibrator(calibrator_id).fit(logits, labels)
    decision = cal.decision_function(logits)
    assert decision.requires_grad is False
    assert torch.allclose(F.softmax(decision, dim=1), cal.transform(logits), atol=1e-5)


def test_decision_function_before_fit_raises():
    with pytest.raises(NotFittedError):
        MatrixScaling(device="cpu").decision_function(torch.randn(2, 3, 4, 4))


def test_transform_does_not_build_graph():
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=61)
    cal = make_calibrator("matrix_scaling").fit(logits, labels)
    x = logits.clone().requires_grad_(True)
    assert cal.transform(x).requires_grad is False


def test_repr_reports_fit_state():
    cal = make_calibrator("vector_scaling")
    assert "unfitted" in repr(cal)
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=62)
    cal.fit(logits, labels)
    assert "fitted" in repr(cal)


def test_factory_builds_by_id():
    for calibrator_id in registered_ids():
        cls = get_calibrator_class(calibrator_id)
        assert cls(device="cpu").calibrator_id == calibrator_id


def test_two_channel_from_binary_logits():
    z = torch.randn(2, 1, 5, 5)
    out = two_channel_from_binary(z, input_type="logits")
    assert out.shape == (2, 2, 5, 5)
    assert torch.allclose(F.softmax(out, dim=1)[:, 1], torch.sigmoid(z[:, 0]), atol=1e-6)


def test_two_channel_from_binary_probs_then_calibrate():
    p = torch.rand(2, 1, 6, 6)
    out = two_channel_from_binary(p, input_type="probs")
    assert out.shape == (2, 2, 6, 6)
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    # the two-channel output is directly usable by a calibrator
    labels = out.argmax(dim=1)
    cal = make_calibrator("temperature_scaling", input_type="probs").fit(out, labels)
    assert cal.transform(out).shape == out.shape


def test_two_channel_from_binary_rejects_bad_shape():
    with pytest.raises(ValueError, match="singleton class axis"):
        two_channel_from_binary(torch.randn(2, 3, 5, 5))


def test_reliability_curve_structure():
    logits, labels = synthetic_logits((4, 3, 8, 8), seed=63)
    curve = reliability_curve(to_probs(logits), labels, n_bins=12)
    assert isinstance(curve, ReliabilityCurve)
    assert curve.bin_confidence.shape == (12,)
    assert curve.bin_accuracy.shape == (12,)
    assert int(curve.bin_counts.sum()) == labels.numel()
    assert 0.0 <= curve.ece <= 1.0


def test_load_warns_on_major_version_mismatch(tmp_path, caplog):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=64)
    cal = make_calibrator("temperature_scaling").fit(logits, labels)
    path = tmp_path / "cal.pt"
    cal.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["fiducio_version"] = "99.0.0"
    torch.save(payload, path)
    with caplog.at_level(logging.WARNING, logger="fiducio"):
        load_calibrator(path)
    assert any("may not be fully compatible" in r.message for r in caplog.records)
