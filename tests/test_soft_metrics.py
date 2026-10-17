"""Metrics must distinguish MHR error from average error against individual votes."""

import math

import pytest
import torch

from fiducio import (
    average_calibration_error,
    brier_score,
    expected_calibration_error,
    mean_human_response,
    negative_log_likelihood,
    reliability_curve,
    unweighted_calibration_error,
)


def test_class_ece_mhr_is_not_mean_rater_ece():
    p = torch.tensor([[0.5, 0.5]])
    votes = torch.tensor([[0, 1]])
    q, _ = mean_human_response(votes, num_classes=2)
    assert expected_calibration_error(p, q, class_index=1) == 0
    per_rater = [expected_calibration_error(p, votes[:, i], class_index=1) for i in range(2)]
    assert sum(per_rater) / 2 == 0.5


def test_soft_metrics_against_manual_formulas_and_disagreement_term():
    p = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=torch.float64)
    votes = torch.tensor([[0, 0, 1], [1, 2, 2]])
    q, _ = mean_human_response(votes, num_classes=3)
    nll = -(q.double() * p.log()).sum(1).mean()
    brier = (p - q.double()).square().sum(1).mean()
    assert negative_log_likelihood(p, q) == pytest.approx(nll.item())
    assert brier_score(p, q) == pytest.approx(brier.item())
    per_rater = sum(brier_score(p, votes[:, r]) for r in range(3)) / 3
    disagreement = (1 - q.double().square().sum(1)).mean().item()
    assert per_rater == pytest.approx(brier.item() + disagreement)
    top = reliability_curve(p, q, n_bins=1)
    assert top.bin_accuracy.item() == pytest.approx((2 / 3 + 1 / 3) / 2)
    for fn in [average_calibration_error, unweighted_calibration_error]:
        assert fn(p, q, class_index=1) == reliability_curve(p, q, class_index=1).ace


def test_brier_binary_two_channel_factor_two():
    p = torch.tensor([[0.2, 0.8]])
    q = torch.tensor([[0.6, 0.4]])
    assert brier_score(p, q) == pytest.approx(2 * (0.8 - 0.4) ** 2)


@pytest.mark.parametrize("fn", [negative_log_likelihood, brier_score, reliability_curve])
def test_soft_metrics_ignore_nan_padding(fn):
    p = torch.tensor([[0.7, 0.3], [float("nan"), float("nan")]])
    q = torch.tensor([[0.4, 0.6], [float("nan"), float("nan")]])
    value = fn(p, q, mask=torch.tensor([True, False]))
    expected = fn(p[:1], q[:1])
    if fn is reliability_curve:
        assert value.ece == expected.ece
    else:
        assert value == expected
    empty = fn(p, q, mask=torch.tensor([False, False]))
    assert math.isnan(empty.ece if fn is reliability_curve else empty)


@pytest.mark.parametrize("fn", [negative_log_likelihood, brier_score, reliability_curve])
def test_one_hot_metric_matches_labels(fn):
    p = torch.tensor([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])
    y = torch.tensor([1, 2])
    q = torch.nn.functional.one_hot(y, 3).float()
    a, b = fn(p, y), fn(p, q)
    if fn is reliability_curve:
        assert a.ece == b.ece and a.ace == b.ace
    else:
        assert a == b


@pytest.mark.parametrize("class_index", [True, -1, 2, 0.5])
def test_invalid_class_index(class_index):
    with pytest.raises(ValueError, match="class_index"):
        reliability_curve(torch.tensor([[0.3, 0.7]]), torch.tensor([1]), class_index=class_index)


def test_class_plot_propagates_metric_and_labels():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from fiducio.plots import reliability_diagram

    p = torch.tensor([[0.5, 0.5]])
    ax = reliability_diagram(p, p, class_index=1)
    assert "class 1" in ax.get_xlabel() and "human response" in ax.get_ylabel()
    assert "0.0000" in ax.texts[0].get_text()
    plt.close(ax.figure)
