"""Tests for the structural guarantees claimed by specific calibrators."""

from __future__ import annotations

import pytest
import torch

from conftest import make_calibrator, synthetic_logits, to_probs
from fiducio import (
    ArgmaxPreservingMatrixScaling,
    OrderPreservingMatrixScaling,
)


def test_cmsap_preserves_argmax():
    logits, labels = synthetic_logits((4, 4, 8, 8), seed=10)
    cal = ArgmaxPreservingMatrixScaling(device="cpu", max_iter=80).fit(logits, labels)
    out = cal.transform(logits)
    assert torch.equal(to_probs(logits).argmax(dim=1), out.argmax(dim=1))


def test_cmsap_preserves_argmax_with_independent_experts():
    # The argmax guarantee comes from the margin parameterization, not from how
    # the experts are optimized, so it must hold under independent_experts=True too.
    logits, labels = synthetic_logits((4, 4, 8, 8), seed=15)
    cal = ArgmaxPreservingMatrixScaling(
        device="cpu", max_iter=80, independent_experts=True
    ).fit(logits, labels)
    out = cal.transform(logits)
    assert torch.equal(to_probs(logits).argmax(dim=1), out.argmax(dim=1))


def test_cmsop_preserves_full_order():
    logits, labels = synthetic_logits((3, 5, 6, 6), seed=11)
    cal = OrderPreservingMatrixScaling(device="cpu", max_iter=80).fit(logits, labels)
    out = cal.transform(logits)
    c = logits.shape[1]
    raw_order = to_probs(logits).movedim(1, -1).reshape(-1, c).argsort(dim=1)
    cal_order = out.movedim(1, -1).reshape(-1, c).argsort(dim=1)
    assert torch.equal(raw_order, cal_order)


def test_cmsop_preserves_full_order_with_independent_experts():
    logits, labels = synthetic_logits((3, 5, 6, 6), seed=16)
    cal = OrderPreservingMatrixScaling(
        device="cpu", max_iter=80, independent_experts=True
    ).fit(logits, labels)
    out = cal.transform(logits)
    c = logits.shape[1]
    raw_order = to_probs(logits).movedim(1, -1).reshape(-1, c).argsort(dim=1)
    cal_order = out.movedim(1, -1).reshape(-1, c).argsort(dim=1)
    assert torch.equal(raw_order, cal_order)


def test_cmsop_preserves_argmax_too():
    logits, labels = synthetic_logits((3, 4, 6, 6), seed=12)
    cal = OrderPreservingMatrixScaling(device="cpu", max_iter=60).fit(logits, labels)
    out = cal.transform(logits)
    assert torch.equal(to_probs(logits).argmax(dim=1), out.argmax(dim=1))


def test_cmsap_preserves_argmax_on_unrelated_random_inputs():
    # Preservation is a structural property of the map: it must hold for inputs
    # unrelated to the calibration set, including near-ties from small logits.
    torch.manual_seed(99)
    logits = torch.randn(4, 5, 8, 8)
    labels = torch.randint(0, 5, (4, 8, 8))
    cal = ArgmaxPreservingMatrixScaling(device="cpu", max_iter=50).fit(logits, labels)
    test_logits = torch.randn(3, 5, 8, 8) * 0.5
    out = cal.transform(test_logits)
    assert torch.equal(to_probs(test_logits).argmax(dim=1), out.argmax(dim=1))


def test_cmsop_preserves_order_on_unrelated_random_inputs():
    torch.manual_seed(100)
    logits = torch.randn(4, 5, 8, 8)
    labels = torch.randint(0, 5, (4, 8, 8))
    cal = OrderPreservingMatrixScaling(device="cpu", max_iter=50).fit(logits, labels)
    test_logits = torch.randn(3, 5, 8, 8) * 0.5
    out = cal.transform(test_logits)
    c = test_logits.shape[1]
    raw_order = to_probs(test_logits).movedim(1, -1).reshape(-1, c).argsort(dim=1)
    cal_order = out.movedim(1, -1).reshape(-1, c).argsort(dim=1)
    assert torch.equal(raw_order, cal_order)


@pytest.mark.parametrize(
    "calibrator_id",
    [
        "temperature_scaling",
        "ensemble_temperature_scaling",
        "translation_invariant_matrix_scaling",
        "dirichlet_calibration",
        "class_conditional_matrix_scaling",
        "argmax_preserving_matrix_scaling",
        "order_preserving_matrix_scaling",
    ],
)
def test_translation_invariance(calibrator_id):
    logits, labels = synthetic_logits((3, 4, 6, 6), seed=13)
    cal = make_calibrator(calibrator_id).fit(logits, labels)
    base = cal.transform(logits)
    shifted = cal.transform(logits + 5.3)
    assert torch.allclose(base, shifted, atol=1e-4)


def test_non_invariant_matrix_scaling_is_not_translation_invariant():
    from fiducio import MatrixScaling

    torch.manual_seed(14)
    x = torch.randn(3, 2, 6, 6)
    cal = MatrixScaling(device="cpu")
    cal._set_state({"weight": torch.tensor([[1., 2.], [3., 4.]]), "bias": torch.zeros(2)})
    cal._num_classes = 2
    cal._fitted = True
    base = cal.transform(x)
    shifted = cal.transform(x + 5.3)
    # A general matrix is not expected to be translation invariant.
    assert not torch.allclose(base, shifted, atol=1e-3)


def test_ets_preserves_argmax_and_order():
    from fiducio import EnsembleTemperatureScaling

    logits, labels = synthetic_logits((4, 5, 8, 8), seed=18, scale=5.0)
    cal = EnsembleTemperatureScaling(device="cpu").fit(logits, labels)
    assert cal.weights[:2].sum() > 0
    out = cal.transform(logits)
    assert torch.equal(to_probs(logits).argmax(dim=1), out.argmax(dim=1))
    c = logits.shape[1]
    raw = to_probs(logits).movedim(1, -1).reshape(-1, c)
    calibrated = out.movedim(1, -1).reshape(-1, c)
    # The mixture cannot invert the raw ordering, but it can tie: the calibrated
    # probabilities of low-confidence classes sit at the 1e-12 safe_log floor,
    # where their true ~1e-28 differences are below float32 resolution. Compare
    # in raw-probability order, requiring non-decreasing (ties allowed).
    ordered = calibrated.gather(1, raw.argsort(dim=1))
    assert (ordered.diff(dim=1) >= 0).all()
