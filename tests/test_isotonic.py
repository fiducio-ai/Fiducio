"""Known optima and lifecycle tests for binary isotonic calibration."""

import numpy as np
import pytest
import torch

from fiducio import BinaryIsotonicCalibration, NotFittedError, load_calibrator


def probabilities(scores):
    p = torch.as_tensor(scores, dtype=torch.float32)
    return torch.stack([1 - p, p], dim=1)


def calibrator(**kwargs):
    return BinaryIsotonicCalibration(input_type="probs", device="cpu", **kwargs)


def test_weighted_pava_known_optimum_and_interpolation():
    p = probabilities([0.1, 0.5, 0.5, 0.5, 0.9])
    q = probabilities([0.8, 0.2, 0.2, 0.2, 0.9])
    cal = calibrator(n_bins=5).fit(p, q)
    torch.testing.assert_close(cal.x, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))
    torch.testing.assert_close(cal.y, torch.tensor([0.35, 0.35, 0.9], dtype=torch.float64))
    result = cal.transform(probabilities([0, 0.3, 0.7, 1]))[:, 1]
    torch.testing.assert_close(result, torch.tensor([0.35, 0.35, 0.625, 0.9]))
    assert torch.equal(cal.predict_proba(p), cal.transform(p))


def test_binning_uses_voxel_weights_and_moves_cuts_past_ties():
    p = probabilities([0.1, 0.2, 0.2, 0.2, 0.8, 0.9])
    q = probabilities([0, 0, 0, 0, 1, 1])
    cal = calibrator(n_bins=3).fit(p, q)
    # Rank 2 is inside the .2 tie group; rank 4 ends that group: one cut at 4.
    torch.testing.assert_close(cal.x, torch.tensor([0.175, 0.85], dtype=torch.float64))
    torch.testing.assert_close(cal.y, torch.tensor([0.0, 1.0], dtype=torch.float64))


def test_decreasing_targets_require_repeated_pooling():
    p = probabilities([0.1, 0.3, 0.6, 0.9])
    cal = calibrator(n_bins=4).fit(p, probabilities([1, 0.8, 0.2, 0]))
    torch.testing.assert_close(cal.transform(p)[:, 1], torch.full((4,), 0.5))


@pytest.mark.parametrize("n_bins", [1, 2, 20])
def test_equal_score_permutation_invariant(n_bins):
    p = probabilities([0.1, 0.1, 0.1, 0.4, 0.4, 0.8])
    q = probabilities([0, 0.2, 1, 0, 1, 0.6])
    permutation = torch.tensor([2, 0, 1, 4, 3, 5])
    first = calibrator(n_bins=n_bins).fit(p, q)
    second = calibrator(n_bins=n_bins).fit(p[permutation], q[permutation])
    torch.testing.assert_close(first.x, second.x, rtol=0, atol=0)
    torch.testing.assert_close(first.y, second.y, rtol=0, atol=0)


def test_single_distinct_score_constant_and_hard_targets():
    p = probabilities([0.5, 0.5, 0.5, 0.5])
    cal = calibrator().fit(p, torch.tensor([0, 1, 0, 0]))
    assert torch.equal(cal.transform(probabilities([0, 1]))[:, 1], torch.tensor([0.25, 0.25]))


def test_exact_zero_one_and_log_probabilities():
    p = probabilities([0, 1])
    cal = calibrator().fit(p, torch.tensor([0, 1]))
    assert torch.equal(cal.transform(p), p)
    decision = cal.decision_function(p)
    assert torch.isneginf(decision[0, 1]) and torch.isneginf(decision[1, 0])
    assert torch.equal(decision.softmax(1), cal.transform(p))


def test_tiny_scores_are_distinct_before_fit():
    p = probabilities([0, 1e-13, 1e-12])
    cal = calibrator().fit(p, probabilities([0, 0.5, 1]))
    assert cal.x.numel() == 3
    torch.testing.assert_close(cal.transform(p)[:, 1], torch.tensor([0.0, 0.5, 1.0]))


@pytest.mark.parametrize("shape", [(2, 2, 3, 4), (2, 2, 2, 3, 4)])
@pytest.mark.parametrize("input_type", ["logits", "probs"])
def test_shapes_masks_numpy_and_detachment(shape, input_type):
    z = torch.randn(shape, generator=torch.Generator().manual_seed(81))
    q = (0.5 * z).softmax(1)
    predictions = z if input_type == "logits" else z.softmax(1)
    mask = torch.ones_like(z[:, 0], dtype=torch.bool)
    mask[0] = False
    predictions[0] = float("nan")
    q[0] = float("nan")
    cal = BinaryIsotonicCalibration(input_type=input_type, device="cpu")
    cal.fit(predictions.numpy(), q.numpy(), mask=mask.numpy())
    out = cal.transform(predictions.requires_grad_(), mask=mask)
    assert out.shape == shape and out.dtype == torch.float32 and not out.requires_grad
    assert torch.isfinite(out).all() and not out[0].any()
    torch.testing.assert_close(out[1].sum(0), torch.ones_like(out[1, 0]))


def test_unfitted_validation_and_failed_refit_are_transactional():
    cal = calibrator()
    with pytest.raises(NotFittedError):
        cal.transform(probabilities([0.5]))
    with pytest.raises(NotFittedError):
        cal.decision_function(probabilities([0.5]))
    p = probabilities([0.1, 0.9])
    cal.fit(p, torch.tensor([0, 1]))
    expected = cal.transform(p)
    with pytest.raises(ValueError, match="exactly 2"):
        cal.fit(torch.full((2, 3), 1 / 3), torch.tensor([0, 1]))
    assert torch.equal(expected, cal.transform(p))
    with pytest.raises(ValueError, match="no valid voxels"):
        cal.fit(p, torch.tensor([0, 1]), mask=torch.tensor([False, False]))
    with pytest.raises(ValueError, match="early stopping"):
        cal.fit(p, torch.tensor([0, 1]), val_predictions=p, val_targets=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="fitted for C=2"):
        cal.transform(torch.full((1, 3), 1 / 3))
    with pytest.raises(ValueError, match="non-finite"):
        cal.transform(torch.tensor([[float("nan"), 1.0]]))


@pytest.mark.parametrize("bins", [True, 0, -1, 2.5])
def test_invalid_bins(bins):
    with pytest.raises(ValueError, match="n_bins"):
        calibrator(n_bins=bins)


def test_assigned_early_stopping_is_rejected():
    cal = calibrator()
    cal.patience = 2
    with pytest.raises(ValueError, match="does not support early stopping"):
        cal.fit(probabilities([0.1, 0.9]), torch.tensor([0, 1]))


@pytest.mark.parametrize("fitted", [True, False])
def test_roundtrip(tmp_path, fitted):
    cal = calibrator(n_bins=np.int64(3))
    p = probabilities([0.1, 0.2, 0.8])
    if fitted:
        cal.fit(p, probabilities([0.2, 0.4, 0.7]))
    path = tmp_path / "isotonic.pt"
    cal.save(path)
    loaded = load_calibrator(path)
    assert loaded.is_fitted == fitted and loaded.get_config() == cal.get_config()
    if fitted:
        assert loaded.x.dtype == torch.float64
        assert torch.equal(loaded.transform(p), cal.transform(p))
    else:
        assert loaded.x is None and loaded.y is None


def test_nearby_bin_means_survive_storage(tmp_path):
    # Means of neighboring float32 scores retain their extra precision in storage.
    base = torch.tensor(0.5)
    next1 = torch.nextafter(base, torch.tensor(1.0))
    next2 = torch.nextafter(next1, torch.tensor(1.0))
    p = probabilities([base, next1, next1, next2])
    cal = calibrator(n_bins=2).fit(p, probabilities([0, 0.3, 0.6, 1]))
    # The first cut moves past the tied middle group, producing a non-float32 mean.
    assert cal.x.dtype == torch.float64 and (cal.x[1:] > cal.x[:-1]).all()
    assert not torch.equal(cal.x, cal.x.float().double())
    path = tmp_path / "near.pt"
    cal.save(path)
    loaded = load_calibrator(path)
    assert torch.equal(loaded.x, cal.x)
    assert torch.equal(loaded.transform(p), cal.transform(p))


@pytest.mark.parametrize(
    "field,value",
    [
        ("x", torch.tensor([0.5, 0.5], dtype=torch.float64)),
        ("x", torch.tensor([0.8, 0.2], dtype=torch.float64)),
        ("x", torch.tensor([-0.1, 0.9], dtype=torch.float64)),
        ("x", torch.tensor([float("nan"), 0.9], dtype=torch.float64)),
        ("x", torch.tensor([0.1, 0.9])),
        ("x", torch.empty(0, dtype=torch.float64)),
        ("y", torch.tensor([0.9, 0.1], dtype=torch.float64)),
        ("y", torch.tensor([0, 1.1], dtype=torch.float64)),
        ("y", torch.tensor([[0, 1]], dtype=torch.float64)),
        ("y", None),
    ],
)
def test_corrupt_isotonic_state_rejected(tmp_path, field, value):
    cal = calibrator().fit(probabilities([0.1, 0.9]), torch.tensor([0, 1]))
    path = tmp_path / "corrupt.pt"
    cal.save(path)
    payload = torch.load(path, weights_only=True)
    payload["state"][field] = value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_calibrator(path)
