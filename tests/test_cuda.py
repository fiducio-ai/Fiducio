"""CUDA tests — skipped automatically when no GPU is available."""

from __future__ import annotations

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_fit_transform_on_cuda(calibrator_id):
    logits, labels = synthetic_logits((3, 3, 8, 8), seed=80)
    cal = make_calibrator(calibrator_id, device="cuda")
    out = cal.fit_transform(logits.cuda(), labels.cuda())
    assert out.is_cuda
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    assert torch.isfinite(out).all()


def test_save_on_cuda_loads_on_cpu(tmp_path):
    from fiducio import load_calibrator

    logits, labels = synthetic_logits((2, 3, 6, 6), seed=81)
    cal = make_calibrator("matrix_scaling", device="cuda").fit(logits.cuda(), labels.cuda())
    path = tmp_path / "cuda_cal.pt"
    cal.save(path)
    loaded = load_calibrator(path)  # default map_location="cpu"
    assert str(loaded.device) == "cpu"
    out = loaded.transform(logits)  # cpu input
    assert not out.is_cuda


def test_metrics_on_cuda_in_deterministic_mode():
    from fiducio import expected_calibration_error

    logits, labels = synthetic_logits((2, 3, 6, 6), seed=82)
    probs = torch.softmax(logits, dim=1).cuda()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected_calibration_error(probs, labels.cuda())
    finally:
        torch.use_deterministic_algorithms(previous)


def test_multi_rater_isotonic_and_metrics_on_cuda(tmp_path):
    from fiducio import (
        BinaryIsotonicCalibration,
        brier_score,
        load_calibrator,
        mean_human_response,
        negative_log_likelihood,
        reliability_curve,
    )

    a = torch.tensor([[0, 1, 1], [1, 1, -100], [-100, -100, -100]], device="cuda")
    q, valid = mean_human_response(a, num_classes=2)
    p = torch.tensor([[0.8, 0.2], [0.1, 0.9], [float("nan"), float("nan")]], device="cuda")
    cal = BinaryIsotonicCalibration(input_type="probs", device="cuda").fit(p, q, mask=valid)
    out = cal.transform(p, mask=valid)
    assert out.is_cuda and torch.isfinite(out).all()
    assert negative_log_likelihood(out, q, mask=valid) >= 0
    assert brier_score(out, q, mask=valid) >= 0
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        assert reliability_curve(out, q, mask=valid, class_index=1).ece >= 0
    finally:
        torch.use_deterministic_algorithms(previous)
    path = tmp_path / "isotonic_cuda.pt"
    cal.save(path)
    loaded = load_calibrator(path)
    torch.testing.assert_close(loaded.transform(p.cpu(), mask=valid.cpu()), out.cpu())


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_soft_targets_on_cuda(calibrator_id):
    logits, _ = synthetic_logits((2, 3, 4, 4), seed=83)
    q = (logits / 2).softmax(1).cuda()
    cal = make_calibrator(calibrator_id, device="cuda", max_iter=3).fit(logits.cuda(), q)
    assert torch.isfinite(cal.transform(logits.cuda())).all()
