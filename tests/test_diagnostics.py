"""Tests for the pre-calibration reliability report."""

from __future__ import annotations

import json
import math

import pytest
import torch

from conftest import synthetic_logits, to_probs
from fiducio import (
    ReliabilityReport,
    brier_score,
    diagnose,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)


def noisy_logits(n_images=24, num_classes=3, size=16, *, scale, seed=0):
    """Logits whose reliability is set by ``scale``: ~1 is calibrated, larger is overconfident."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, num_classes, (n_images, size, size), generator=g)
    onehot = torch.nn.functional.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
    return (onehot * 1.5 + torch.randn(onehot.shape, generator=g)) * scale, labels


def test_pooled_metrics_match_the_global_functions():
    logits, labels = synthetic_logits((6, 4, 12, 12), seed=1)
    labels = labels.clone()
    labels[0, :2] = -100
    mask = torch.ones(6, 12, 12, dtype=torch.bool)
    mask[1, 3:5] = False
    report = diagnose(logits, labels, mask=mask, n_boot=20)
    probs = to_probs(logits)
    assert report.metrics["ece"][0] == pytest.approx(
        expected_calibration_error(probs, labels, mask), rel=1e-6
    )
    assert report.metrics["nll"][0] == pytest.approx(
        negative_log_likelihood(probs, labels, mask), rel=1e-6
    )
    assert report.metrics["brier"][0] == pytest.approx(brier_score(probs, labels, mask), rel=1e-6)
    assert report.n_valid == int((mask & (labels != -100)).sum())
    assert int(report.curve.bin_counts.sum()) == report.n_valid


def test_probs_input_matches_logits_input():
    logits, labels = synthetic_logits((4, 3, 8, 8), seed=2)
    a = diagnose(logits, labels, n_boot=0)
    b = diagnose(to_probs(logits), labels, input_type="probs", n_boot=0)
    assert a.metrics["ece"][0] == pytest.approx(b.metrics["ece"][0], rel=1e-6)


def test_calibrated_model_has_no_significant_bias_or_warnings():
    logits, labels = noisy_logits(scale=1.0)
    # Labels drawn from the model's own probabilities are calibrated by construction.
    g = torch.Generator().manual_seed(0)
    probs = to_probs(logits).permute(0, 2, 3, 1)
    labels = torch.multinomial(probs.reshape(-1, 3), 1, generator=g).view(labels.shape)
    report = diagnose(logits, labels, min_class_voxels=10)
    assert report.bias == "no significant bias"
    assert report.warnings == []


def test_overconfident_model_is_flagged():
    logits, labels = noisy_logits(scale=4.0)
    report = diagnose(logits, labels)
    assert report.bias == "overconfident"
    assert any("overconfident" in w for w in report.warnings)
    assert report.metrics["gap"][1] > 0


def test_underconfident_model_is_flagged():
    logits, labels = noisy_logits(scale=0.2)
    assert diagnose(logits, labels).bias == "underconfident"


def test_failed_image_is_detected_and_listed_first():
    logits, labels = noisy_logits(scale=1.0, num_classes=3)
    logits[3] = 0.0
    logits[3, 0] = 10.0  # predicts class 0 (the background here) everywhere
    ids = [f"img{i}" for i in range(len(logits))]
    # Class 0 scores 0.5 on that image, which would lift the mean Dice above the threshold.
    report = diagnose(logits, labels, image_ids=ids, background_class=0, n_boot=0)
    assert report.failures.n_failed == 1
    assert report.failures.worst[0][0] == "img3"
    assert report.failures.worst[0][1] < 0.1
    assert report.failures.confidence_failed > report.failures.confidence_rest
    assert any("1 image(s) have a mean Dice below" in w for w in report.warnings)
    # ECE without the failed image equals the global ECE of the other images.
    keep = torch.arange(len(logits)) != 3
    expected = expected_calibration_error(to_probs(logits[keep]), labels[keep])
    assert report.failures.ece_without_failed == pytest.approx(expected, rel=1e-6)


def test_background_class_is_excluded_from_the_image_dice():
    logits, labels = noisy_logits(scale=1.0)
    labels[:] = 0  # every pixel is background
    logits[:] = 0.0
    logits[:, 0] = 10.0
    labels[2, :4] = 1  # one image has a foreground structure the model misses
    report = diagnose(logits, labels, background_class=0, n_boot=0)
    assert report.failures.n_failed == 1
    assert report.failures.worst[0][2] == "1"
    # Without the exclusion, the background Dice of 1 hides the failure.
    assert diagnose(logits, labels, n_boot=0).failures.n_failed == 0


def test_class_absent_everywhere_is_not_a_failure():
    logits, labels = noisy_logits(scale=1.0, num_classes=3)
    logits[:, 2] = -50.0  # class 2 never predicted
    labels[labels == 2] = 0  # and never in the labels
    report = diagnose(logits, labels, n_boot=0)
    c = report.classes[2]
    assert c.n_present == 0 and c.n_failed == 0 and c.n_spurious == 0
    assert math.isnan(c.dice) and math.isnan(c.dice_median)
    assert any("absent from the labels" in w for w in report.warnings)
    assert "n/a" in report.summary()


def test_spurious_and_never_predicted_classes():
    logits, labels = noisy_logits(scale=1.0, num_classes=3)
    labels[labels == 2] = 1  # class 2 absent from labels but still predicted
    report = diagnose(logits, labels, n_boot=0)
    assert report.classes[2].n_spurious > 0
    assert any("predicted on" in w and "absent" in w for w in report.warnings)
    logits[:, 1] = -50.0  # class 1 present in labels, never predicted
    assert any("never predicted" in w for w in diagnose(logits, labels, n_boot=0).warnings)


def test_rare_and_sparsely_routed_class_warnings():
    logits, labels = noisy_logits(scale=1.0, size=32)
    labels[labels == 2] = 0
    labels[0, 0, :3] = 2  # three voxels of class 2
    report = diagnose(logits, labels, n_boot=0)
    assert any("is rare" in w for w in report.warnings)
    logits[:, 2] = -50.0
    logits[0, 2, 0, :3] = 50.0  # predicted on three voxels only
    assert any("class-conditional" in w for w in diagnose(logits, labels, n_boot=0).warnings)


def test_failing_class_rate_warning():
    logits, labels = noisy_logits(scale=1.0)
    logits[:4] = 0.0
    logits[:4, 0] = 10.0  # 4 of 24 images fail on classes 1 and 2
    report = diagnose(logits, labels, n_boot=0)
    assert any("fails (Dice < 0.1)" in w for w in report.warnings)
    assert report.classes[1].n_failed >= 4


def test_few_images_and_single_image():
    logits, labels = noisy_logits(n_images=1, scale=2.0)
    report = diagnose(logits, labels)
    assert any("only 1 image (" in w for w in report.warnings)
    assert math.isnan(report.metrics["ece"][1])
    assert report.bias.startswith("unknown")


def test_bootstrap_is_deterministic_and_brackets_the_estimate():
    logits, labels = noisy_logits(scale=3.0)
    a, b = diagnose(logits, labels, seed=3), diagnose(logits, labels, seed=3)
    assert a.metrics == b.metrics
    for key, (_, lo, hi) in a.metrics.items():
        # The ECE interval can miss its estimate (see the docstring), so only check order.
        assert math.isfinite(lo) and lo <= hi, key
    assert diagnose(logits, labels, seed=4).metrics["ece"] != a.metrics["ece"]


def test_to_dict_is_strict_json_and_text_names_the_classes():
    logits, labels = noisy_logits(scale=2.0)
    logits[:, 2] = -50.0
    labels[labels == 2] = 0
    report = diagnose(logits, labels, class_names=["bg", "pet", "edge"], n_boot=0)
    data = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert data["classes"][2]["dice"] is None
    assert data["bias"] == report.bias
    text = str(report)
    assert text == repr(report) == report.summary()
    assert "pet" in text and "edge" in text and "Lowest-Dice images" in text


def test_summary_without_failures_or_warnings():
    logits, labels = noisy_logits(scale=1.0)
    g = torch.Generator().manual_seed(0)
    probs = to_probs(logits).permute(0, 2, 3, 1)
    labels = torch.multinomial(probs.reshape(-1, 3), 1, generator=g).view(labels.shape)
    text = diagnose(logits, labels, dice_failure_threshold=0.0, min_class_voxels=10).summary()
    assert "No warnings" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_type": "scores"}, "input_type"),
        ({"n_boot": -1}, "n_boot"),
        ({"dice_failure_threshold": 1.5}, "threshold"),
        ({"class_names": ["a"]}, "class_names"),
        ({"image_ids": ["a"]}, "image_ids"),
        ({"background_class": 7}, "background_class"),
    ],
)
def test_invalid_arguments(kwargs, match):
    logits, labels = noisy_logits(scale=1.0)
    with pytest.raises(ValueError, match=match):
        diagnose(logits, labels, **kwargs)


def test_soft_targets_and_empty_inputs_are_rejected():
    logits, labels = noisy_logits(n_images=2, scale=1.0)
    with pytest.raises(ValueError, match="hard integer labels"):
        diagnose(logits, to_probs(logits))
    with pytest.raises(ValueError, match="at least one image"):
        diagnose(logits[:0], labels[:0])
    with pytest.raises(ValueError, match="no valid voxels"):
        diagnose(logits, torch.full_like(labels, -100))


def test_plot_returns_a_figure():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logits, labels = noisy_logits(scale=2.0)
    fig = diagnose(logits, labels).plot()
    assert len(fig.axes) == 4
    plt.close(fig)


def test_public_exports():
    import fiducio

    assert fiducio.diagnose is diagnose
    assert isinstance(diagnose(*noisy_logits(scale=1.0), n_boot=0), ReliabilityReport)


def test_single_resample_gives_no_interval_and_integer_logits_are_accepted():
    logits, labels = noisy_logits(n_images=4, scale=3.0)
    assert math.isnan(diagnose(logits, labels, n_boot=1).metrics["ece"][1])
    rounded = logits.round()
    from_ints = diagnose(rounded.long().numpy(), labels.numpy(), n_boot=0)
    from_floats = diagnose(rounded, labels, n_boot=0)
    # to_dict maps NaN to None, which compares equal, unlike NaN itself.
    assert from_ints.to_dict() == from_floats.to_dict()


def test_class_and_routed_statistics_match_the_reference_metrics():
    logits, labels = synthetic_logits((6, 4, 12, 12), seed=5)
    labels = labels.clone()
    labels[0, :2] = -100
    mask = torch.ones(6, 12, 12, dtype=torch.bool)
    mask[1, 3:5] = False
    report = diagnose(logits, labels, mask=mask, n_boot=0)
    probs = to_probs(logits)
    valid = mask & (labels != -100)
    pred = probs.argmax(1)
    for c, stats in enumerate(report.classes):
        assert stats.class_ece == pytest.approx(
            reliability_curve(probs, labels, mask, class_index=c).ece, rel=1e-6
        )
        routed = valid & (pred == c)
        assert stats.pred_voxels == int(routed.sum())
        assert stats.routed_ece == pytest.approx(
            expected_calibration_error(probs, labels, routed), rel=1e-6
        )
        assert stats.label_voxels == int((valid & (labels == c)).sum())
        inter = int((routed & (labels == c)).sum())
        assert stats.dice == pytest.approx(2 * inter / (stats.label_voxels + stats.pred_voxels))


def test_ignore_index_must_be_an_int():
    with pytest.raises(TypeError, match="ignore_index"):
        diagnose(torch.randn(2, 3, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long), ignore_index=None)


def test_perfect_model_lists_no_worst_images():
    labels = torch.randint(0, 3, (4, 8, 8))
    logits = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float() * 50
    assert diagnose(logits, labels).failures.worst == []


def test_small_but_frequent_class_is_not_called_rare():
    labels = torch.randint(0, 3, (1, 16, 16))
    report = diagnose(torch.randn(1, 3, 16, 16), labels)
    assert not any("is rare" in w for w in report.warnings)
    assert any("has only" in w for w in report.warnings)


def test_flat_input_is_rejected():
    with pytest.raises(ValueError, match="spatial"):
        diagnose(torch.randn(100, 3), torch.randint(0, 3, (100,)))
