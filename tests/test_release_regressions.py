"""Regression cases found in the 0.1.0 source and published-wheel audit."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator
from fiducio import (
    EnsembleTemperatureScaling,
    MatrixScaling,
    TemperatureScaling,
    average_calibration_error,
    brier_score,
    expected_calibration_error,
    load_calibrator,
    negative_log_likelihood,
    reliability_curve,
)
from fiducio.calibrators._optim import minimize
from fiducio.utils.stopping import StoppingRule


def data():
    g = torch.Generator().manual_seed(711)
    z = torch.randn(12, 3, generator=g)
    return z, torch.arange(12) % 3


def test_adam_keeps_initial_state_when_steps_are_worse():
    x = torch.tensor(0., requires_grad=True)
    minimize("adam", [x], lambda: (x - .1).square(), lr=1., max_iter=2)
    assert x.detach().item() == 0.


def test_adam_considers_final_step():
    x = torch.tensor(0., requires_grad=True)
    minimize("adam", [x], lambda: (x - 1.).square(), lr=.1, max_iter=1)
    assert x.detach().item() == pytest.approx(.1)


def test_min_delta_does_not_discard_actual_validation_minimum():
    x = torch.tensor(0., requires_grad=True)
    minimize("adam", [x], lambda: (x - 1.).square(), lr=.1, max_iter=2,
             val_fn=lambda: (x - 1.).square(), stopping=StoppingRule(patience=2, min_delta=100.))
    assert x.detach().item() > .19


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_nonfinite_optimization_is_reported(optimizer):
    x = torch.tensor(1., requires_grad=True)
    with pytest.raises(ValueError, match="non-finite"):
        minimize(optimizer, [x], lambda: x * float("inf"), lr=.1, max_iter=2)


def test_million_voxel_metrics_do_not_drift():
    p = torch.tensor([.9, .1]).repeat(1_000_000, 1)
    y = torch.zeros(1_000_000, dtype=torch.long)
    curve = reliability_curve(p, y)
    assert curve.ece == pytest.approx(.1, abs=1e-6)
    assert curve.ace == pytest.approx(.1, abs=1e-6)
    assert curve.bin_counts.dtype == torch.int64
    assert curve.bin_confidence.dtype == torch.float64
    assert curve.bin_counts.sum().item() == len(y)


def test_reliability_accumulators_follow_device_with_fake_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        p = torch.empty(2, 3, device="meta")
        y = torch.empty(2, dtype=torch.long, device="meta")
        # Scalar materialization is unsupported for fake tensors. Reaching it
        # proves that bucketize and scatter_add have consistent device layouts.
        with patch("fiducio.metrics.calibration._flatten_valid", return_value=(p, y)):
            from torch._subclasses.fake_tensor import DataDependentOutputException
            with pytest.raises(DataDependentOutputException):
                reliability_curve(p, y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_metrics_with_cpu_labels_and_masks():
    p = torch.tensor([[.9, .1], [.8, .2]], device="cuda")
    y = torch.zeros(2, dtype=torch.long)
    mask = torch.ones(2, dtype=torch.bool)
    curve = reliability_curve(p, y, mask)
    assert curve.ece == pytest.approx(.15, abs=1e-6)
    assert curve.bin_counts.device.type == "cuda"
    assert negative_log_likelihood(p, y, mask) > 0
    assert brier_score(p, y, mask) > 0


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_live_model_outputs_do_not_backpropagate_into_model(calibrator_id):
    model = torch.nn.Linear(4, 3)
    z = model(torch.ones(12, 4))
    _, y = data()
    cal = make_calibrator(calibrator_id, max_iter=3, patience=2)
    cal.fit(z, y, val_predictions=z, val_targets=y)
    assert cal.is_fitted
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_failed_refit_preserves_old_parameters_and_metadata(calibrator_id):
    z, y = data()
    cal = make_calibrator(calibrator_id, max_iter=2).fit(z, y)
    before = cal.transform(z)
    with pytest.raises(ValueError, match="no valid voxels"):
        cal.fit(torch.ones(12, 4), torch.full((12,), -100))
    assert cal.is_fitted and cal.num_classes == 3
    assert torch.equal(before, cal.transform(z))
    # Failure inside optimization must also preserve the original learned state.
    cal.lr = float("nan")
    with pytest.raises(ValueError):
        cal.fit(z, y)
    assert torch.equal(before, cal.transform(z))


@pytest.mark.parametrize("mask_shape", [(1, 2, 2), (2, 1, 2, 2)])
def test_transform_rejects_broadcastable_masks(mask_shape):
    z, y = data()
    cal = TemperatureScaling(device="cpu", max_iter=2).fit(z, y)
    with pytest.raises(ValueError, match="mask must have shape"):
        cal.transform(torch.zeros(2, 3, 2, 2), torch.ones(mask_shape))


@pytest.mark.parametrize("fn", [negative_log_likelihood, brier_score,
                                expected_calibration_error, average_calibration_error])
@pytest.mark.parametrize("probs", [[[2., -1.]], [[.2, .2]], [[float("nan"), .5]]])
def test_metrics_reject_invalid_probabilities(fn, probs):
    with pytest.raises(ValueError):
        fn(torch.tensor(probs), torch.tensor([0]))


@pytest.mark.parametrize("labels", [[.9, 1.9], [float("nan"), 1.], [float("inf"), 1.]])
def test_fractional_and_nonfinite_labels_rejected(labels):
    z = torch.tensor([[2., 1.], [1., 2.]])
    y = torch.tensor(labels)
    with pytest.raises(ValueError, match="integer"):
        TemperatureScaling(device="cpu").fit(z, y)
    with pytest.raises(ValueError, match="integer"):
        negative_log_likelihood(z.softmax(1), y)


def test_integral_float_labels_supported():
    z, y = data()
    assert negative_log_likelihood(z.softmax(1), y.float()) == negative_log_likelihood(z.softmax(1), y)


def test_metrics_reject_reshaped_labels_and_masks():
    z = torch.zeros(2, 3, 2, 2).softmax(1)
    with pytest.raises(ValueError, match="targets must have shape"):
        reliability_curve(z, torch.zeros(8, dtype=torch.long))
    with pytest.raises(ValueError, match="mask must have shape"):
        reliability_curve(z, torch.zeros(2, 2, 2, dtype=torch.long), torch.ones(8))


@pytest.mark.parametrize("cls", [TemperatureScaling, EnsembleTemperatureScaling])
def test_large_initial_temperature_stays_finite(cls):
    z, y = data()
    cal = cls(device="cpu", init_temperature=100., max_iter=3).fit(z, y)
    assert math.isfinite(cal.temperature)
    assert torch.isfinite(cal.transform(z)).all()


@pytest.mark.parametrize("kwargs", [{"lr": float("nan")}, {"lr": float("inf")},
                                   {"max_iter": 1.5}, {"lambda_reg": -1.},
                                   {"mu_reg": float("nan")}, {"min_delta": float("nan")}])
def test_invalid_hyperparameters_rejected(kwargs):
    with pytest.raises(ValueError):
        MatrixScaling(device="cpu", **kwargs)


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_unfitted_roundtrip(tmp_path, calibrator_id):
    cal = make_calibrator(calibrator_id)
    path = tmp_path / "unfitted.pt"
    cal.save(path)
    out = load_calibrator(path)
    assert not out.is_fitted and out.num_classes is None


@pytest.mark.parametrize("change", ["future", "missing", "shape", "nan", "fitted", "config"])
def test_corrupt_payload_rejected(tmp_path, change):
    z, y = data()
    path = tmp_path / "bad.pt"
    MatrixScaling(device="cpu", max_iter=2).fit(z, y).save(path)
    payload = torch.load(path, weights_only=True)
    if change == "future":
        payload["format_version"] = 999
    elif change == "missing":
        del payload["state"]["bias"]
    elif change == "shape":
        payload["state"]["weight"] = torch.ones(2, 2)
    elif change == "nan":
        payload["state"]["weight"][0, 0] = float("nan")
    elif change == "fitted":
        payload["fitted"] = "yes"
    elif change == "config":
        payload["config"]["device"] = "cpu"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_calibrator(path)


def test_loader_never_retries_without_weights_only(tmp_path):
    path = tmp_path / "exists.pt"
    path.touch()
    with patch("fiducio.persistence.torch.load", side_effect=TypeError("bad file")) as loader:
        with pytest.raises(TypeError):
            load_calibrator(path)
        loader.assert_called_once_with(str(path), map_location="cpu", weights_only=True)


def test_legacy_full_matrix_msc_loads(tmp_path):
    from fiducio import TranslationInvariantMatrixScaling
    z, y = data()
    cal = TranslationInvariantMatrixScaling(device="cpu", max_iter=2).fit(z, y)
    path = tmp_path / "legacy.pt"
    cal.save(path)
    payload = torch.load(path, weights_only=True)
    payload["state"]["weight"] = cal._effective_weight()
    del payload["state"]["row_sum"]
    payload["fiducio_version"] = "0.1.0"
    torch.save(payload, path)
    assert torch.allclose(load_calibrator(path).transform(z), cal.transform(z), atol=1e-6)


V010 = Path(__file__).parent / "fixtures" / "v0_1_0"
V010_REFERENCE = torch.load(V010 / "reference.pt", weights_only=True)


@pytest.mark.parametrize("name", sorted(V010_REFERENCE["expected"]))
def test_files_written_by_published_v010_load_and_match(name):
    """Files saved by the published 0.1.0 wheel (see make_v0_1_0_fixtures.py)."""
    cal = load_calibrator(V010 / f"{name}_fitted.pt")
    z = V010_REFERENCE["eval_logits"]
    x = z.softmax(1) if cal.input_type == "probs" else z
    assert cal.is_fitted and cal.num_classes == 3
    assert torch.allclose(cal.transform(x), V010_REFERENCE["expected"][name], atol=1e-6)


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_unfitted_files_written_by_published_v010_load(calibrator_id):
    cal = load_calibrator(V010 / f"{calibrator_id}_unfitted.pt")
    assert not cal.is_fitted and cal.num_classes is None
    z, y = data()
    val = {"val_predictions": z, "val_targets": y} if cal.patience is not None else {}
    assert cal.fit(z, y, **val).transform(z).shape == z.shape


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
@pytest.mark.parametrize("shape", [(2, 3, 4, 4), (1, 3, 2, 3, 4), (6, 3)])
def test_outputs_keep_prediction_shape_with_and_without_mask(calibrator_id, shape):
    g = torch.Generator().manual_seed(5)
    z = torch.randn(*shape, generator=g)
    y = z.argmax(1)
    mask = torch.rand(y.shape, generator=g) > 0.3
    cal = make_calibrator(calibrator_id, max_iter=5).fit(z, y)
    assert cal.transform(z).shape == z.shape
    assert cal.transform(z, mask).shape == z.shape
    assert cal.decision_function(z).shape == z.shape
