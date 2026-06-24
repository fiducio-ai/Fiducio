"""Tests for calibration metrics."""

from __future__ import annotations

import math

import torch

from conftest import synthetic_logits, to_probs
from fiducio import brier_score, expected_calibration_error, negative_log_likelihood


def test_perfect_predictions_have_zero_metrics():
    labels = torch.tensor([[0, 1, 2]])
    probs = torch.zeros(1, 3, 3)
    for i, c in enumerate([0, 1, 2]):
        probs[0, c, i] = 1.0
    assert negative_log_likelihood(probs, labels) < 1e-5
    assert brier_score(probs, labels) < 1e-6
    assert expected_calibration_error(probs, labels) < 1e-6


def test_nll_matches_manual():
    probs = torch.tensor([[[0.7], [0.2], [0.1]]])  # (1, 3, 1)
    labels = torch.tensor([[0]])
    assert math.isclose(
        negative_log_likelihood(probs, labels), -math.log(0.7), rel_tol=1e-5
    )


def test_metrics_respect_mask_and_ignore_index():
    logits, labels = synthetic_logits((2, 3, 6, 6), seed=40)
    probs = to_probs(logits)
    full = negative_log_likelihood(probs, labels)
    labels_ignored = labels.clone()
    labels_ignored[0] = -100
    partial = negative_log_likelihood(probs, labels_ignored)
    assert not math.isclose(full, partial)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    mask[0] = False
    masked = negative_log_likelihood(probs, labels, mask=mask)
    assert math.isclose(masked, partial, rel_tol=1e-5)


def test_ece_in_unit_interval():
    logits, labels = synthetic_logits((4, 3, 8, 8), seed=41)
    probs = to_probs(logits)
    ece = expected_calibration_error(probs, labels, n_bins=20)
    assert 0.0 <= ece <= 1.0


def test_empty_valid_returns_nan():
    probs = to_probs(torch.randn(1, 3, 4, 4))
    labels = torch.full((1, 4, 4), -100, dtype=torch.long)
    assert math.isnan(negative_log_likelihood(probs, labels))
