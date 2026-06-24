"""CUDA tests — skipped automatically when no GPU is available."""

from __future__ import annotations

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


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
