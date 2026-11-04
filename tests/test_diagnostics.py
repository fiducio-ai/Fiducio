"""Tests for the pre-calibration reliability report."""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from conftest import synthetic_logits, to_probs
from fiducio import (
    ReliabilityReport,
    brier_score,
    diagnose,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)


class IdentityModel(torch.nn.Module):
    """Return its input unchanged, with one parameter to check no-grad inference."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class UnwrapModel(torch.nn.Module):
    """Accept a dict or a sequence of inputs and return the predictions inside."""

    def forward(self, x):
        return x["logits"] if isinstance(x, dict) else x[0]


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
    assert data["soft_targets"] is False
    assert set(data["boundary"]) == {
        "low_bin_mass",
        "low_bin_gap",
        "high_bin_mass",
        "high_bin_gap",
        "ece_contribution",
    }
    # Per-class intervals serialise as [low, high] lists, None when undefined.
    assert data["classes"][0]["class_ece_ci"] == [None, None]
    text = str(report)
    assert text == repr(report) == report.summary()
    assert "pet" in text and "edge" in text and "Lowest-Dice images" in text
    assert "Per-class 95% intervals" in text


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_boot": 1.5}, "n_boot"),
        ({"n_boot": True}, "n_boot"),
        ({"n_worst": 1.5}, "n_worst"),
        ({"min_images": -1}, "min_images"),
        ({"min_images": float("nan")}, "min_images"),
        ({"min_images": 1.5}, "min_images"),
        ({"min_images": True}, "min_images"),
        ({"min_class_voxels": -1}, "min_class_voxels"),
        ({"min_class_voxels": float("nan")}, "min_class_voxels"),
        ({"min_class_fraction": float("nan")}, "min_class_fraction"),
        ({"min_class_fraction": -0.1}, "min_class_fraction"),
        ({"min_class_fraction": 1.5}, "min_class_fraction"),
        ({"min_class_fraction": True}, "min_class_fraction"),
    ],
)
def test_invalid_thresholds_and_resample_counts(kwargs, match):
    logits, labels = noisy_logits(n_images=4, scale=1.0)
    with pytest.raises(ValueError, match=match):
        diagnose(logits, labels, **{"n_boot": 0, **kwargs})


def test_threshold_boundaries_are_accepted():
    logits, labels = noisy_logits(n_images=4, scale=1.0)
    diagnose(logits, labels, n_boot=0, min_images=0, min_class_voxels=0, min_class_fraction=0.0)
    diagnose(logits, labels, n_boot=0, min_class_fraction=1.0)
    diagnose(logits, labels, n_worst=0)


def test_background_class_excluding_every_present_class_warns():
    logits, labels = noisy_logits(n_images=4, scale=1.0)
    labels[:] = 0  # every label is the background class
    logits[:] = 0.0
    logits[:, 0] = 10.0
    report = diagnose(logits, labels, background_class=0, n_boot=0)
    assert report.failures.n_failed == 0
    assert report.failures.worst == []
    assert any("failure statistics are undefined" in w for w in report.warnings)
    # The report still renders and serialises with undefined failure statistics.
    assert "0/4" in report.summary()
    json.loads(json.dumps(report.to_dict(), allow_nan=False))


def test_invalid_soft_targets_and_empty_inputs_are_rejected():
    logits, labels = noisy_logits(n_images=2, scale=1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        diagnose(logits, to_probs(logits) / 2)
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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"seed": "1"}, "seed"),
        ({"dice_failure_threshold": True}, "dice_failure_threshold"),
        ({"dice_failure_threshold": "0.5"}, "dice_failure_threshold"),
        ({"dice_failure_threshold": None}, "dice_failure_threshold"),
        ({"dice_failure_threshold": 1j}, "dice_failure_threshold"),
        ({"background_class": True}, "background_class"),
        ({"background_class": 1.0}, "background_class"),
        ({"background_class": "1"}, "background_class"),
        ({"class_names": "ab"}, "class_names"),
        ({"class_names": ["a", 1, "c"]}, "class_names"),
        ({"image_ids": "abc"}, "image_ids"),
        ({"image_ids": ["a", 1, "c", "d"]}, "image_ids"),
        ({"ignore_index": True}, "ignore_index"),
    ],
)
def test_invalid_argument_types(kwargs, match):
    logits, labels = noisy_logits(n_images=4, scale=1.0)
    with pytest.raises((TypeError, ValueError), match=match):
        diagnose(logits, labels, n_boot=0, **kwargs)


def test_numpy_integer_arguments_accepted():
    import numpy as np

    logits, labels = noisy_logits(n_images=4, scale=1.0)
    report = diagnose(
        logits,
        labels,
        n_boot=np.int64(2),
        seed=np.int64(1),
        ignore_index=np.int64(-100),
        background_class=np.int64(0),
    )
    assert report.n_images == 4


def test_nan_predictions_at_ignore_index_voxels_are_excluded():
    logits, labels = noisy_logits(n_images=4, scale=1.0)
    ignored = labels.clone()
    ignored[0, 0, 0] = -100
    contaminated = logits.clone()
    contaminated[0, :, 0, 0] = float("nan")
    clean = diagnose(logits, ignored, n_boot=0)
    padded = diagnose(contaminated, ignored, n_boot=0)
    assert padded.to_dict() == clean.to_dict()
    # A NaN at a voxel that is actually labelled still fails validation.
    with pytest.raises(ValueError, match="non-finite"):
        diagnose(contaminated, labels, n_boot=0)


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


# --- soft / MHR targets ---------------------------------------------------------------


def test_soft_targets_match_the_global_metrics_and_leave_dice_undefined():
    logits, _ = synthetic_logits((6, 4, 12, 12), seed=11)
    soft = to_probs(logits)
    mask = torch.ones(6, 12, 12, dtype=torch.bool)
    mask[1, 3:5] = False
    report = diagnose(logits, soft, mask=mask, n_boot=0)

    assert report.soft_targets is True
    assert report.metrics["ece"][0] == pytest.approx(
        expected_calibration_error(soft, soft, mask), abs=1e-6
    )
    assert report.metrics["nll"][0] == pytest.approx(
        negative_log_likelihood(soft, soft, mask), abs=1e-6
    )
    assert report.metrics["brier"][0] == pytest.approx(brier_score(soft, soft, mask), abs=1e-6)
    valid = mask
    pred = soft.argmax(1)
    for c, stats in enumerate(report.classes):
        assert stats.class_ece == pytest.approx(
            reliability_curve(soft, soft, mask, class_index=c).ece, abs=1e-6
        )
        assert stats.routed_ece == pytest.approx(
            expected_calibration_error(soft, soft, valid & (pred == c)), abs=1e-6
        )
        assert stats.label_voxels == pytest.approx(float(soft[:, c][valid].sum()), rel=1e-6)
        assert stats.n_present == stats.n_failed == stats.n_spurious == 0
        assert math.isnan(stats.dice) and math.isnan(stats.dice_ci[0])
    assert bool(report.image_dice.isnan().all())
    assert report.failures.n_failed == 0 and report.failures.worst == []
    assert math.isnan(report.failures.ece_without_failed)
    assert any("soft targets" in w for w in report.warnings)
    # ignore_index selects hard labels only: it changes nothing for a soft target.
    assert (
        diagnose(logits, soft, mask=mask, ignore_index=7, n_boot=0).to_dict()
        == report.to_dict()
    )
    json.loads(json.dumps(report.to_dict(), allow_nan=False))


def test_soft_targets_and_bootstrap_give_per_class_intervals():
    logits, _ = synthetic_logits((6, 4, 8, 8), seed=12)
    soft = to_probs(logits)
    a = diagnose(logits, soft, n_boot=20, seed=5)
    b = diagnose(logits, soft, n_boot=20, seed=5)
    for ca, cb in zip(a.classes, b.classes, strict=True):
        assert ca.class_ece_ci == cb.class_ece_ci
        assert ca.routed_ece_ci == cb.routed_ece_ci
        assert ca.dice_ci == cb.dice_ci
    assert all(math.isnan(c.dice_ci[0]) for c in a.classes)


# --- model + DataLoader ---------------------------------------------------------------


def test_loader_path_matches_the_tensor_path_and_preserves_model_state():
    logits, labels = noisy_logits(n_images=11, scale=2.0)
    model = IdentityModel()
    loader = DataLoader(TensorDataset(logits, labels), batch_size=4)
    streamed = diagnose(model, loader, device="cpu", n_boot=12, seed=3)
    direct = diagnose(logits, labels, n_boot=12, seed=3)
    assert streamed.to_dict() == direct.to_dict()
    assert model.training is True
    assert model.scale.grad is None


def test_loader_mask_is_used_and_probs_input_matches():
    logits, labels = noisy_logits(n_images=8, scale=2.0)
    mask = torch.ones(8, 16, 16, dtype=torch.bool)
    mask[1, :3] = False
    mask[2, 0, 0] = False
    loader = DataLoader(
        [(logits[i], labels[i], mask[i]) for i in range(len(logits))], batch_size=3
    )
    streamed = diagnose(IdentityModel(), loader, n_boot=0)
    assert streamed.to_dict() == diagnose(logits, labels, mask=mask, n_boot=0).to_dict()

    probs = to_probs(logits)
    loader = DataLoader(TensorDataset(probs, labels), batch_size=5)
    report = diagnose(IdentityModel(), loader, input_type="probs", n_boot=0)
    assert report.to_dict() == diagnose(probs, labels, input_type="probs", n_boot=0).to_dict()


def test_loader_soft_targets_match_the_tensor_path():
    logits, _ = synthetic_logits((5, 3, 8, 8), seed=13)
    soft = to_probs(logits)
    loader = DataLoader(TensorDataset(logits, soft), batch_size=2)
    streamed = diagnose(IdentityModel(), loader, n_boot=0)
    direct = diagnose(logits, soft, n_boot=0)
    assert streamed.to_dict() == direct.to_dict()
    assert streamed.soft_targets is True


def test_loader_accepts_dict_and_sequence_inputs():
    logits, labels = noisy_logits(n_images=4, scale=2.0)
    expected = diagnose(logits, labels, n_boot=0)
    dict_batches = [({"logits": logits[i : i + 2]}, labels[i : i + 2]) for i in (0, 2)]
    tuple_batches = [((logits[i : i + 2], "unused"), labels[i : i + 2]) for i in (0, 2)]
    assert diagnose(UnwrapModel(), dict_batches, n_boot=0).to_dict() == expected.to_dict()
    assert diagnose(UnwrapModel(), tuple_batches, n_boot=0).to_dict() == expected.to_dict()


def test_loader_errors():
    logits, labels = noisy_logits(n_images=3, scale=1.0)
    model = IdentityModel()
    with pytest.raises(ValueError, match="at least one image"):
        diagnose(model, [])
    with pytest.raises(ValueError, match="labels must be a DataLoader"):
        diagnose(model, None)
    with pytest.raises(ValueError, match="labels must be a DataLoader"):
        diagnose(model, labels)
    with pytest.raises(ValueError, match="pass each mask in the batch"):
        diagnose(model, [(logits[0], labels[0])], mask=torch.ones(16, 16, dtype=torch.bool))
    with pytest.raises(ValueError, match="loader batch"):
        diagnose(model, [(logits[0],)])
    with pytest.raises(ValueError, match="loader batch"):
        diagnose(model, [torch.tensor(1)])
    with pytest.raises(ValueError, match="image_ids"):
        diagnose(model, [(logits[:1], labels[:1])], image_ids=["a", "b"])
    with pytest.raises(ValueError, match="class_names"):
        diagnose(model, [(logits[:1], labels[:1])], class_names=5)


def test_loader_rejects_bad_model_output_and_mixed_batches():
    logits, labels = noisy_logits(n_images=3, scale=1.0)

    class NotATensor(torch.nn.Module):
        def forward(self, x):
            return [x]

    class Flat(torch.nn.Module):
        def forward(self, x):
            return x[:, :, 0, 0]

    class Wider(torch.nn.Module):
        """Return three classes on the first call and four on the second."""

        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return x if self.calls == 1 else torch.cat([x, x[:, :1]], dim=1)

    class IntegerLogits(torch.nn.Module):
        def forward(self, x):
            return (x * 10).round().long()

    # Integer model outputs are converted to float, as for tensor inputs.
    report = diagnose(IntegerLogits(), [(logits[:1], labels[:1])], n_boot=0)
    assert report.n_images == 1 and report.n_valid > 0

    with pytest.raises(TypeError, match="must return a tensor"):
        diagnose(NotATensor(), [(logits[:1], labels[:1])])
    with pytest.raises(ValueError, match="model output must have shape"):
        diagnose(Flat(), [(logits[:1], labels[:1])])
    with pytest.raises(ValueError, match="fitted for C=3"):
        diagnose(Wider(), [(logits[:1], labels[:1]), (logits[1:2], labels[1:2])])
    # A hard batch followed by a soft batch.
    with pytest.raises(ValueError, match="must not be mixed"):
        diagnose(
            IdentityModel(),
            [(logits[:1], labels[:1]), (logits[1:2], to_probs(logits[1:2]))],
        )


# --- boundary bins --------------------------------------------------------------------


def test_boundary_bins_at_the_probability_edges():
    probs = torch.full((1, 20, 3), 1 / 20)
    probs[0, 0, 0] = 1.0
    probs[0, 1:, 0] = 0.0
    labels = torch.tensor([[0, 1, 1]])  # two wrong, minimum-confidence voxels
    report = diagnose(probs, labels, input_type="probs", n_bins=10, n_boot=0)
    boundary = report.boundary
    assert boundary.low_bin_mass == pytest.approx(2 / 3)
    assert boundary.high_bin_mass == pytest.approx(1 / 3)
    assert boundary.low_bin_gap == pytest.approx(0.05)
    assert boundary.high_bin_gap == pytest.approx(0.0)
    # Only the first bin is miscalibrated, so it contributes the whole ECE.
    assert boundary.ece_contribution == pytest.approx(0.1 / 3)
    assert boundary.ece_contribution == pytest.approx(report.metrics["ece"][0])
    # A single bin counts once, not twice.
    one_bin = diagnose(probs, labels, input_type="probs", n_bins=1, n_boot=0)
    assert one_bin.boundary.ece_contribution == pytest.approx(one_bin.metrics["ece"][0])


def test_empty_edge_bins_have_undefined_gaps():
    probs = torch.full((2, 2, 4, 4), 0.5)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    report = diagnose(probs, labels, input_type="probs", n_bins=15, n_boot=0)
    assert report.boundary.low_bin_mass == 0.0
    assert report.boundary.high_bin_mass == 0.0
    assert math.isnan(report.boundary.low_bin_gap)
    assert math.isnan(report.boundary.high_bin_gap)


# --- per-class intervals --------------------------------------------------------------


def test_per_class_intervals_are_deterministic_and_guarded():
    logits, labels = noisy_logits(n_images=6, scale=2.0)
    logits[:, 2] = -50.0
    logits[0, 2, 0, :4] = 50.0  # class 2 routed on one image only
    a = diagnose(logits, labels, n_boot=40, seed=7)
    b = diagnose(logits, labels, n_boot=40, seed=7)
    assert a.to_dict()["classes"] == b.to_dict()["classes"]
    for c in a.classes:
        for lo, hi in (c.class_ece_ci, c.routed_ece_ci, c.dice_ci):
            if not math.isnan(lo):
                assert lo <= hi
    assert math.isfinite(a.classes[0].class_ece_ci[0])
    assert math.isnan(a.classes[2].routed_ece_ci[0])
    assert all(
        math.isnan(c.class_ece_ci[0])
        for c in diagnose(*noisy_logits(1, scale=1.0), n_boot=20).classes
    )


def test_class_present_in_one_image_has_no_dice_interval():
    logits, labels = noisy_logits(n_images=5, scale=1.0)
    labels[labels == 2] = 0
    labels[0, 0, 0] = 2
    c = diagnose(logits, labels, n_boot=20).classes[2]
    assert c.n_present == 1
    assert math.isnan(c.dice_ci[0])
