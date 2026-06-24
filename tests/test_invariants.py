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


def test_cmsop_preserves_full_order():
    logits, labels = synthetic_logits((3, 5, 6, 6), seed=11)
    cal = OrderPreservingMatrixScaling(device="cpu", max_iter=80).fit(logits, labels)
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
