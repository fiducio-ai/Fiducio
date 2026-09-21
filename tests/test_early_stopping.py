"""Tests for validation-based early stopping."""

from __future__ import annotations

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits
from fiducio import load_calibrator, negative_log_likelihood
from fiducio.calibrators._optim import minimize
from fiducio.utils.stopping import StoppingRule

CLASS_CONDITIONAL_IDS = [
    "class_conditional_matrix_scaling",
    "argmax_preserving_matrix_scaling",
    "order_preserving_matrix_scaling",
]


def _splits():
    fit_logits, fit_labels = synthetic_logits((4, 3, 10, 10), seed=70, scale=4.0)
    val_logits, val_labels = synthetic_logits((4, 3, 10, 10), seed=71, scale=4.0)
    return fit_logits, fit_labels, val_logits, val_labels


# ----------------------------------------------------------------- minimize()


def _quadratic(target_train: float, target_val: float):
    x = torch.zeros((), requires_grad=True)
    calls = {"val": 0}

    def loss_fn():
        return (x - target_train) ** 2

    def val_fn():
        calls["val"] += 1
        return (x - target_val) ** 2

    return x, calls, loss_fn, val_fn


def test_minimize_restores_best_validation_iterate_and_stops_early():
    # Training pulls x to 3, validation prefers x = 1: validation degrades after x passes 1.
    x, calls, loss_fn, val_fn = _quadratic(3.0, 1.0)
    rule = StoppingRule(patience=5)
    minimize("adam", [x], loss_fn, lr=0.05, max_iter=500, val_fn=val_fn, stopping=rule)
    assert calls["val"] < 500
    assert abs(float(x.detach()) - 1.0) < 0.1


def test_minimize_without_stopping_follows_training_loss():
    x, calls, loss_fn, val_fn = _quadratic(3.0, 1.0)
    minimize("adam", [x], loss_fn, lr=0.05, max_iter=300, val_fn=val_fn, stopping=None)
    assert calls["val"] == 0
    assert float(x) > 2.0


def test_minimize_lr_patience_alone_runs_to_max_iter():
    x, calls, loss_fn, val_fn = _quadratic(3.0, 1.0)
    rule = StoppingRule(lr_patience=3, lr_factor=0.5)
    minimize("adam", [x], loss_fn, lr=0.05, max_iter=80, val_fn=val_fn, stopping=rule)
    assert calls["val"] == 81  # initial state plus 80 updates
    assert abs(float(x.detach()) - 1.0) < 0.1


def test_minimize_min_delta_stops_on_marginal_gains():
    x = torch.tensor(5.0, requires_grad=True)

    def loss_fn():
        return (x - 0.0) ** 2

    calls = {"n": 0}

    def val_fn():
        calls["n"] += 1
        return (x - 0.0) ** 2

    # A huge min_delta means no step ever counts as an improvement after the first.
    minimize(
        "adam", [x], loss_fn, lr=0.1, max_iter=200, val_fn=val_fn,
        stopping=StoppingRule(patience=3, min_delta=1e6),
    )
    assert calls["n"] == 4  # initial validation plus 3 stale steps


# ------------------------------------------------------------- configuration


def test_early_stopping_is_off_by_default():
    for calibrator_id in ALL_CALIBRATOR_IDS:
        cal = make_calibrator(calibrator_id)
        assert cal.patience is None and cal.lr_patience is None
        assert "patience" not in cal.get_config()


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_early_stopping_requires_adam(calibrator_id):
    with pytest.raises(ValueError, match="requires optimizer='adam'"):
        make_calibrator(calibrator_id, optimizer="lbfgs", patience=5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"patience": 0}, "patience must be > 0"),
        ({"lr_patience": -1}, "lr_patience must be > 0"),
        ({"patience": 3, "min_delta": -0.1}, "min_delta must be >= 0"),
        ({"patience": 3, "lr_factor": 1.5}, "lr_factor must be in"),
        ({"patience": 3, "lr_factor": 0.0}, "lr_factor must be in"),
    ],
)
def test_invalid_stopping_arguments_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_calibrator("temperature_scaling", **kwargs)


def test_fit_requires_validation_data_when_stopping_configured():
    logits, labels, _, _ = _splits()
    cal = make_calibrator("temperature_scaling", patience=5)
    with pytest.raises(ValueError, match="requires val_predictions and val_targets"):
        cal.fit(logits, labels)


def test_validation_data_rejected_without_stopping():
    logits, labels, val_logits, val_labels = _splits()
    cal = make_calibrator("temperature_scaling")
    with pytest.raises(ValueError, match="early stopping is not configured"):
        cal.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)


def test_validation_arguments_must_be_paired():
    logits, labels, val_logits, _ = _splits()
    cal = make_calibrator("temperature_scaling", patience=5)
    with pytest.raises(ValueError, match="must be given together"):
        cal.fit(logits, labels, val_predictions=val_logits)
    with pytest.raises(ValueError, match="val_mask given without"):
        cal.fit(logits, labels, val_mask=torch.ones(4, 10, 10, dtype=torch.bool))


def test_validation_class_count_must_match():
    logits, labels, _, _ = _splits()
    val_logits, val_labels = synthetic_logits((2, 4, 10, 10), seed=72)
    cal = make_calibrator("temperature_scaling", patience=5)
    with pytest.raises(ValueError, match="validation predictions have 4 classes"):
        cal.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)


def test_validation_without_valid_voxels_rejected():
    logits, labels, val_logits, val_labels = _splits()
    cal = make_calibrator("temperature_scaling", patience=5)
    with pytest.raises(ValueError, match="no valid validation voxels"):
        cal.fit(
            logits, labels, val_predictions=val_logits, val_targets=val_labels,
            val_mask=torch.zeros_like(val_labels, dtype=torch.bool),
        )


# ------------------------------------------------------------- fitting


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_all_calibrators_fit_with_early_stopping(calibrator_id):
    logits, labels, val_logits, val_labels = _splits()
    cal = make_calibrator(calibrator_id, patience=4, lr_patience=2, max_iter=150)
    cal.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    probs = cal.transform(val_logits)
    assert torch.isfinite(probs).all()
    raw = torch.softmax(val_logits, dim=1)
    assert negative_log_likelihood(probs, val_labels) <= negative_log_likelihood(raw, val_labels)
    # validation data is not retained after fitting
    assert cal._val is None


@pytest.mark.parametrize("calibrator_id", ["matrix_scaling", "class_conditional_matrix_scaling"])
def test_early_stopping_never_worse_on_validation_than_fixed_budget(calibrator_id):
    # A tiny calibration set and a long budget let an expressive map overfit.
    g = torch.Generator().manual_seed(5)
    fit_logits = torch.randn(1, 4, 6, 6, generator=g) * 4
    fit_labels = torch.randint(0, 4, (1, 6, 6), generator=g)
    val_logits, val_labels = synthetic_logits((4, 4, 12, 12), seed=73, scale=4.0)
    kwargs = {"max_iter": 400, "lr": 0.05}
    plain = make_calibrator(calibrator_id, **kwargs).fit(fit_logits, fit_labels)
    stopped = make_calibrator(calibrator_id, patience=10, **kwargs).fit(
        fit_logits, fit_labels, val_predictions=val_logits, val_targets=val_labels
    )
    nll_plain = negative_log_likelihood(plain.transform(val_logits), val_labels)
    nll_stopped = negative_log_likelihood(stopped.transform(val_logits), val_labels)
    assert nll_stopped <= nll_plain + 1e-4


@pytest.mark.parametrize("calibrator_id", CLASS_CONDITIONAL_IDS)
def test_class_conditional_independent_experts_with_early_stopping(calibrator_id):
    logits, labels, val_logits, val_labels = _splits()
    cal = make_calibrator(calibrator_id, independent_experts=True, patience=4, max_iter=100)
    cal.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    assert torch.isfinite(cal.transform(val_logits)).all()


def test_independent_expert_without_validation_voxels_still_fits():
    logits, labels, _, _ = _splits()
    # Validation set where class 2 is never the top class: expert 2 has no validation voxels.
    val_logits = torch.zeros(2, 3, 8, 8)
    val_logits[:, 0] = 2.0
    val_logits[:, 1] = 1.0
    val_labels = torch.randint(0, 2, (2, 8, 8))
    cal = make_calibrator(
        "class_conditional_matrix_scaling", independent_experts=True, patience=3, max_iter=50
    )
    cal.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    assert torch.isfinite(cal.transform(logits)).all()


def test_preservation_guarantees_hold_with_early_stopping():
    logits, labels, val_logits, val_labels = _splits()
    ap = make_calibrator("argmax_preserving_matrix_scaling", patience=4, max_iter=120)
    ap.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    assert torch.equal(ap.transform(val_logits).argmax(1), val_logits.argmax(1))
    op = make_calibrator("order_preserving_matrix_scaling", patience=4, max_iter=120)
    op.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    assert torch.equal(op.transform(val_logits).argsort(1), val_logits.argsort(1))


def test_fit_transform_forwards_validation_data():
    logits, labels, val_logits, val_labels = _splits()
    cal = make_calibrator("temperature_scaling", patience=4)
    out = cal.fit_transform(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    assert out.shape == logits.shape


def test_validation_supports_probs_input_and_mask():
    logits, labels, val_logits, val_labels = _splits()
    mask = torch.ones_like(val_labels, dtype=torch.bool)
    mask[:, :2] = False
    cal = make_calibrator("temperature_scaling", patience=4, input_type="probs")
    cal.fit(
        torch.softmax(logits, dim=1), labels,
        val_predictions=torch.softmax(val_logits, dim=1), val_targets=val_labels, val_mask=mask,
    )
    assert cal.is_fitted


def test_stopping_settings_persist_in_roundtrip(tmp_path):
    logits, labels, val_logits, val_labels = _splits()
    cal = make_calibrator("matrix_scaling", patience=7, min_delta=1e-4, lr_patience=3, lr_factor=0.5)
    cal.fit(logits, labels, val_predictions=val_logits, val_targets=val_labels)
    path = tmp_path / "cal.pt"
    cal.save(path)
    loaded = load_calibrator(path)
    assert (loaded.patience, loaded.min_delta, loaded.lr_patience, loaded.lr_factor) == (
        7, 1e-4, 3, 0.5,
    )
    assert torch.allclose(cal.transform(val_logits), loaded.transform(val_logits), atol=1e-6)
