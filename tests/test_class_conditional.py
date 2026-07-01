"""Tests for the joint vs. independent expert optimization option (CDC/CMSap/CMSop)."""

from __future__ import annotations

import pytest
import torch

from conftest import make_calibrator, synthetic_logits, to_probs
from fiducio import load_calibrator, negative_log_likelihood
from fiducio.registry import get_calibrator_class

CLASS_CONDITIONAL_IDS = [
    "class_conditional_matrix_scaling",
    "argmax_preserving_matrix_scaling",
    "order_preserving_matrix_scaling",
]


@pytest.mark.parametrize("calibrator_id", CLASS_CONDITIONAL_IDS)
def test_independent_experts_defaults_to_false(calibrator_id):
    cal = get_calibrator_class(calibrator_id)(device="cpu")
    assert cal.independent_experts is False


@pytest.mark.parametrize("calibrator_id", CLASS_CONDITIONAL_IDS)
def test_independent_experts_fits_and_reduces_nll(calibrator_id):
    logits, labels = synthetic_logits((6, 4, 10, 10), seed=60, scale=5.0)
    raw = to_probs(logits)
    cal = make_calibrator(calibrator_id, independent_experts=True)
    out = cal.fit_transform(logits, labels)
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    assert torch.isfinite(out).all()
    assert negative_log_likelihood(out, labels) <= negative_log_likelihood(raw, labels) + 1e-3


@pytest.mark.parametrize("calibrator_id", CLASS_CONDITIONAL_IDS)
def test_independent_experts_persisted_in_roundtrip(tmp_path, calibrator_id):
    logits, labels = synthetic_logits((3, 4, 6, 6), seed=61)
    cal = make_calibrator(calibrator_id, independent_experts=True).fit(logits, labels)
    expected = cal.transform(logits)
    path = tmp_path / "cal.pt"
    cal.save(path)
    loaded = load_calibrator(path)
    assert loaded.independent_experts is True
    assert torch.allclose(expected, loaded.transform(logits), atol=1e-6)
