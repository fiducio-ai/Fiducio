"""Behavioural tests covering every calibrator across input variants."""

from __future__ import annotations

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits, to_probs


def _check_output(out, like):
    assert out.shape == like.shape
    assert torch.isfinite(out).all(), "output contains NaN/Inf"
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), "probabilities must sum to 1"
    assert (out >= -1e-6).all() and (out <= 1 + 1e-6).all()


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_fit_transform_2d_logits(calibrator_id):
    logits, labels = synthetic_logits((4, 3, 10, 10), seed=1)
    cal = make_calibrator(calibrator_id, input_type="logits")
    assert cal.is_fitted is False
    out = cal.fit_transform(logits, labels)
    assert cal.is_fitted is True
    assert cal.num_classes == 3
    _check_output(out, to_probs(logits))


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_fit_transform_3d(calibrator_id):
    logits, labels = synthetic_logits((2, 4, 4, 5, 5), seed=2)
    cal = make_calibrator(calibrator_id)
    out = cal.fit_transform(logits, labels)
    _check_output(out, to_probs(logits))


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_binary_segmentation(calibrator_id):
    logits, labels = synthetic_logits((3, 2, 8, 8), seed=3)
    cal = make_calibrator(calibrator_id)
    out = cal.fit_transform(logits, labels)
    _check_output(out, to_probs(logits))


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_probs_input_matches_shapes(calibrator_id):
    logits, labels = synthetic_logits((3, 3, 6, 6), seed=4)
    probs = to_probs(logits)
    cal = make_calibrator(calibrator_id, input_type="probs")
    out = cal.fit_transform(probs, labels)
    _check_output(out, probs)


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_tabular_2d_inputs(calibrator_id):
    # (N, C) with no spatial axes must also work.
    logits, labels = synthetic_logits((200, 3), seed=5)
    cal = make_calibrator(calibrator_id)
    out = cal.fit_transform(logits, labels)
    _check_output(out, to_probs(logits))


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_mask_and_ignore_index(calibrator_id):
    logits, labels = synthetic_logits((3, 3, 8, 8), seed=6)
    g = torch.Generator().manual_seed(6)
    mask = torch.rand(3, 8, 8, generator=g) > 0.25
    labels = labels.clone()
    labels[(torch.rand(3, 8, 8, generator=g) > 0.7)] = -100
    cal = make_calibrator(calibrator_id)
    out = cal.fit(logits, labels, mask=mask).transform(logits, mask=mask)
    # valid voxels sum to 1, masked-out voxels are zeroed
    valid = mask
    s = out.sum(dim=1)
    assert torch.allclose(s[valid], torch.ones_like(s[valid]), atol=1e-4)
    assert torch.allclose(s[~valid], torch.zeros_like(s[~valid]), atol=1e-6)


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_custom_ignore_index(calibrator_id):
    logits, labels = synthetic_logits((2, 3, 6, 6), seed=7)
    labels = labels.clone()
    labels[0, 0, 0] = 255
    cal = make_calibrator(calibrator_id, ignore_index=255)
    out = cal.fit_transform(logits, labels)
    _check_output(out, to_probs(logits))


def test_predict_proba_alias_equivalent():
    logits, labels = synthetic_logits((2, 3, 6, 6), seed=8)
    cal = make_calibrator("temperature_scaling").fit(logits, labels)
    assert torch.equal(cal.transform(logits), cal.predict_proba(logits))


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_refit_overwrites_previous_fit(calibrator_id):
    logits, labels = synthetic_logits((4, 3, 10, 10), seed=10)
    cal = make_calibrator(calibrator_id)
    cal.fit(logits, labels)
    out_first = cal.transform(logits)

    # Refitting on the same data must be deterministic and not raise.
    cal.fit(logits, labels)
    out_second = cal.transform(logits)
    assert cal.is_fitted is True
    assert torch.allclose(out_first, out_second, atol=1e-5)

    # Refitting with a different number of classes must also work cleanly,
    # discarding all state from the previous fit.
    logits5, labels5 = synthetic_logits((4, 5, 10, 10), seed=11)
    cal.fit(logits5, labels5)
    assert cal.num_classes == 5
    out_new = cal.transform(logits5)
    _check_output(out_new, to_probs(logits5))


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_reduces_nll_on_overconfident_data(calibrator_id):
    from fiducio import negative_log_likelihood

    logits, labels = synthetic_logits((6, 3, 12, 12), seed=9, scale=5.0)
    raw = to_probs(logits)
    cal = make_calibrator(calibrator_id)
    out = cal.fit_transform(logits, labels)
    # Overconfident inputs: a sane calibrator should not increase NLL much and
    # generally reduces it.
    assert negative_log_likelihood(out, labels) <= negative_log_likelihood(raw, labels) + 1e-3
