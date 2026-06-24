"""Tests for the configurable optimizer (adam / lbfgs)."""

from __future__ import annotations

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits, to_probs
from fiducio import load_calibrator, negative_log_likelihood
from fiducio.registry import get_calibrator_class


def test_default_optimizer_is_adam():
    for calibrator_id in ALL_CALIBRATOR_IDS:
        cal = get_calibrator_class(calibrator_id)(device="cpu")
        assert cal.optimizer == "adam"


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_both_optimizers_reduce_nll(calibrator_id, optimizer):
    logits, labels = synthetic_logits((6, 3, 12, 12), seed=50, scale=5.0)
    raw = to_probs(logits)
    cal = make_calibrator(calibrator_id, optimizer=optimizer)
    out = cal.fit_transform(logits, labels)
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    assert torch.isfinite(out).all()
    assert negative_log_likelihood(out, labels) <= negative_log_likelihood(raw, labels) + 1e-3


def test_invalid_optimizer_rejected():
    with pytest.raises(ValueError, match="optimizer must be one of"):
        make_calibrator("temperature_scaling", optimizer="rmsprop")


def test_per_optimizer_lr_defaults():
    ts_adam = make_calibrator("temperature_scaling", optimizer="adam")
    ts_lbfgs = make_calibrator("temperature_scaling", optimizer="lbfgs")
    assert ts_adam.lr == 0.1 and ts_adam.max_iter == 200
    assert ts_lbfgs.lr == 1.0 and ts_lbfgs.max_iter == 100
    # CMS keeps its own tuned Adam learning rate.
    cms_adam = get_calibrator_class("class_conditional_matrix_scaling")(device="cpu")
    assert cms_adam.lr == 1e-2


def test_explicit_lr_and_max_iter_override():
    cal = make_calibrator("matrix_scaling", optimizer="adam", lr=0.05, max_iter=37)
    assert cal.lr == 0.05 and cal.max_iter == 37


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_optimizer_persisted_in_roundtrip(tmp_path, optimizer):
    logits, labels = synthetic_logits((3, 3, 6, 6), seed=51)
    cal = make_calibrator("matrix_scaling", optimizer=optimizer).fit(logits, labels)
    expected = cal.transform(logits)
    path = tmp_path / "cal.pt"
    cal.save(path)
    loaded = load_calibrator(path)
    assert loaded.optimizer == optimizer
    assert loaded.lr == cal.lr
    assert loaded.max_iter == cal.max_iter
    assert torch.allclose(expected, loaded.transform(logits), atol=1e-6)
