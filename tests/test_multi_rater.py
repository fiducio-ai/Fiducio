"""Scientific and API regressions for calibration to class distributions."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits
from fiducio import mean_human_response


def test_mhr_counts_available_votes_and_masks_empty_voxels():
    annotations = torch.tensor(
        [[[0, 1, -100], [0, -100, -100], [0, 0, -100], [1, 1, -100], [1, -100, -100]]]
    )
    q, mask = mean_human_response(annotations, num_classes=3)
    torch.testing.assert_close(q, torch.tensor([[[0.6, 1 / 3, 0], [0.4, 2 / 3, 0], [0, 0, 0]]]))
    assert mask.tolist() == [[True, True, False]]
    assert q.dtype == torch.float32 and q.device == annotations.device


def test_mhr_empty_rater_axis_and_numpy():
    q, mask = mean_human_response(np.empty((2, 0, 3), dtype=np.int64), num_classes=2)
    assert q.shape == (2, 2, 3) and not q.any() and not mask.any()
    q, mask = mean_human_response(np.array([[2]]), num_classes=np.int64(3))
    assert q.tolist() == [[0, 0, 1]] and mask.all()


@pytest.mark.parametrize("classes", [True, 1, 2.5])
def test_mhr_rejects_invalid_classes(classes):
    with pytest.raises(ValueError, match="num_classes"):
        mean_human_response(torch.zeros(1, 1), num_classes=classes)


@pytest.mark.parametrize(
    "a",
    [torch.tensor([1]), torch.tensor([[0.2]]), torch.tensor([[3]]), torch.tensor([[float("nan")]])],
)
def test_mhr_rejects_invalid_annotations(a):
    with pytest.raises(ValueError):
        mean_human_response(a, num_classes=3)


def test_ce_mhr_equals_average_rater_loss_and_gradient():
    z = torch.tensor([[2.0, -1.0, 0.5], [-1.0, 2.0, 0.2]], requires_grad=True)
    a = torch.tensor([[0, 1, 0], [2, 1, 0]])
    q, _ = mean_human_response(a, num_classes=3)
    soft = F.cross_entropy(z, q)
    votes = torch.stack([F.cross_entropy(z, a[:, r]) for r in range(3)]).mean()
    torch.testing.assert_close(soft, votes)
    torch.testing.assert_close(torch.autograd.grad(soft, z)[0], torch.autograd.grad(votes, z)[0])


@pytest.mark.parametrize("cid", ALL_CALIBRATOR_IDS)
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_one_hot_targets_match_labels_for_every_method(cid, optimizer):
    z, y = synthetic_logits((2, 3, 3, 4), seed=41)
    q = F.one_hot(y, 3).movedim(-1, 1).float()
    hard = make_calibrator(cid, optimizer=optimizer, max_iter=5).fit(z, y)
    soft = make_calibrator(cid, optimizer=optimizer, max_iter=5).fit(z, q)
    torch.testing.assert_close(hard.transform(z), soft.transform(z), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("cid", ALL_CALIBRATOR_IDS)
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_soft_fit_with_padding_and_detached_targets(cid, optimizer):
    z, _ = synthetic_logits((2, 3, 2, 3, 2), seed=42)
    q = z.softmax(1).detach().requires_grad_()
    mask = torch.ones(2, 2, 3, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = False
    z[0, :, 0, 0, 0] = float("nan")
    q = q.masked_fill(~mask.unsqueeze(1), float("nan"))
    cal = make_calibrator(cid, optimizer=optimizer, max_iter=3).fit(z, q, mask=mask)
    out = cal.transform(z, mask=mask)
    assert torch.isfinite(out).all() and not out.requires_grad
    assert torch.equal(out[0, :, 0, 0, 0], torch.zeros(3))


@pytest.mark.parametrize("cid", ALL_CALIBRATOR_IDS)
@pytest.mark.parametrize("train_soft,val_soft", [(True, True), (False, True), (True, False)])
def test_soft_validation_formats_are_independent(cid, train_soft, val_soft):
    z, y = synthetic_logits((12, 3), seed=43)
    q = 0.8 * F.one_hot(y, 3).float() + 0.2 / 3
    cal = make_calibrator(cid, max_iter=4, patience=2)
    cal.fit(z, q if train_soft else y, val_predictions=z, val_targets=q if val_soft else y)
    assert torch.isfinite(cal.transform(z)).all()


@pytest.mark.parametrize("cid", ALL_CALIBRATOR_IDS[-3:])
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_independent_experts_accept_soft_targets(cid, optimizer):
    z, y = synthetic_logits((18, 3), seed=44)
    q = 0.7 * F.one_hot(y, 3).float() + 0.1
    cal = make_calibrator(cid, independent_experts=True, optimizer=optimizer, max_iter=3)
    kwargs = {} if optimizer == "lbfgs" else dict(val_predictions=z, val_targets=q)
    if kwargs:
        cal.patience = 2
    cal.fit(z, q, **kwargs)
    assert torch.isfinite(cal.transform(z)).all()


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([[0.2, 0.2]]),
        torch.tensor([[1.2, -0.2]]),
        torch.tensor([[float("nan"), 1.0]]),
        torch.tensor([[1, 0]]),
    ],
)
def test_invalid_soft_targets_rejected_transactionally(bad):
    cal = make_calibrator("temperature_scaling", max_iter=2).fit(
        torch.tensor([[2.0, 0.0]]), torch.tensor([0])
    )
    before = cal.transform(torch.tensor([[2.0, 0.0]]))
    with pytest.raises(ValueError):
        cal.fit(torch.tensor([[2.0, 0.0]]), bad)
    assert torch.equal(before, cal.transform(torch.tensor([[2.0, 0.0]])))


def test_ets_soft_stage_two_loss_is_cross_entropy_of_mixture(monkeypatch):
    import fiducio.calibrators.ensemble_temperature as module
    from fiducio import EnsembleTemperatureScaling

    z = torch.tensor([[2.0, 0.0], [-1.0, 1.0]])
    q = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    cal = EnsembleTemperatureScaling(device="cpu")
    cal.temperature = 2.0

    def check(optimizer, params, loss_fn, **kwargs):
        weights = params[0].softmax(0)
        p = (
            weights[0] * (z / 2).softmax(1).double()
            + weights[1] * z.softmax(1).double()
            + weights[2] / 2
        )
        expected = -(q.double() * p.log()).sum(1).mean()
        torch.testing.assert_close(loss_fn(), expected)
        torch.testing.assert_close(
            torch.autograd.grad(loss_fn(), params)[0], torch.autograd.grad(expected, params)[0]
        )

    monkeypatch.setattr(module, "minimize", check)
    cal._fit_weights(z, q, 2)
