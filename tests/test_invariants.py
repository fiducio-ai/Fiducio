"""Tests for the structural guarantees claimed by specific calibrators."""

from __future__ import annotations

import torch

from conftest import synthetic_logits, to_probs
from fiducio import (
    ArgmaxPreservingMatrixScaling,
    OrderPreservingMatrixScaling,
    TranslationInvariantMatrixScaling,
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


def test_translation_invariance():
    logits, labels = synthetic_logits((3, 4, 6, 6), seed=13)
    cal = TranslationInvariantMatrixScaling(device="cpu").fit(logits, labels)
    base = cal.transform(logits)
    shifted = cal.transform(logits + 5.3)
    assert torch.allclose(base, shifted, atol=1e-4)


def test_non_invariant_matrix_scaling_is_not_translation_invariant():
    from fiducio import MatrixScaling

    logits, labels = synthetic_logits((3, 4, 6, 6), seed=14)
    cal = MatrixScaling(device="cpu").fit(logits, labels)
    base = cal.transform(logits)
    shifted = cal.transform(logits + 5.3)
    # A general matrix is not expected to be translation invariant.
    assert not torch.allclose(base, shifted, atol=1e-3)
