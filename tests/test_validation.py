"""Tests for input validation and error handling."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import synthetic_logits, to_probs
from fiducio import (
    ArgmaxPreservingMatrixScaling,
    MatrixScaling,
    NotFittedError,
    TemperatureScaling,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    two_channel_from_binary,
)


def test_transform_before_fit_raises():
    cal = TemperatureScaling(device="cpu")
    with pytest.raises(NotFittedError):
        cal.transform(torch.randn(2, 3, 4, 4))


def test_single_class_rejected():
    cal = TemperatureScaling(device="cpu")
    with pytest.raises(ValueError, match="at least 2 classes"):
        cal.fit(torch.randn(2, 1, 4, 4), torch.zeros(2, 4, 4, dtype=torch.long))


def test_shape_mismatch_targets():
    cal = TemperatureScaling(device="cpu")
    logits = torch.randn(2, 3, 4, 4)
    with pytest.raises(ValueError, match="targets must have shape"):
        cal.fit(logits, torch.zeros(2, 5, 5, dtype=torch.long))


def test_fit_without_targets_raises():
    cal = TemperatureScaling(device="cpu")
    with pytest.raises(ValueError, match="targets are required"):
        cal.fit(torch.randn(2, 3, 4, 4), None)


def test_mask_shape_mismatch_rejected():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=25)
    bad_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="mask must have shape"):
        cal.fit(logits, labels, mask=bad_mask)


def test_class_count_mismatch_on_transform():
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=20)
    cal = MatrixScaling(device="cpu").fit(logits, labels)
    with pytest.raises(ValueError, match="fitted for C=3"):
        cal.transform(torch.randn(2, 4, 4, 4))


def test_probs_not_summing_to_one_rejected():
    cal = TemperatureScaling(input_type="probs", device="cpu")
    bad = torch.full((2, 3, 4, 4), 0.5)  # sums to 1.5
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="sum to 1"):
        cal.fit(bad, labels)


def test_calibrator_out_of_range_probs_suggests_input_type():
    """The calibrator branch keeps its own advice: input_type is a real argument."""
    cal = TemperatureScaling(input_type="probs", device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=23)
    with pytest.raises(ValueError, match="pass input_type='logits'"):
        cal.fit(logits, labels)


def test_soft_target_out_of_range_gives_target_advice():
    """Soft targets have no input_type argument, so the advice must differ."""
    cal = TemperatureScaling(input_type="probs", device="cpu")
    probs = torch.full((2, 2, 4, 4), 0.5)
    targets = probs.clone()
    targets[:, 0] = -0.1
    with pytest.raises(ValueError, match="soft targets must lie in") as exc:
        cal.fit(probs, targets)
    assert "input_type" not in str(exc.value)


def test_soft_target_not_summing_to_one_gives_target_advice():
    cal = TemperatureScaling(input_type="probs", device="cpu")
    probs = torch.full((2, 2, 4, 4), 0.5)
    targets = torch.full((2, 2, 4, 4), 0.25)  # sums to 0.5
    with pytest.raises(ValueError, match="soft targets must sum to 1") as exc:
        cal.fit(probs, targets)
    assert "input_type" not in str(exc.value)


def test_non_finite_soft_targets_named_as_targets():
    cal = TemperatureScaling(input_type="probs", device="cpu")
    probs = torch.full((2, 2, 4, 4), 0.5)
    targets = torch.full((2, 2, 4, 4), 0.5)
    targets[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="targets contain non-finite"):
        cal.fit(probs, targets)


def test_non_finite_predictions_rejected():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=21)
    logits = logits.clone()
    logits[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        cal.fit(logits, labels)


def test_label_out_of_range_rejected():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=22)
    labels = labels.clone()
    labels[0, 0, 0] = 9
    with pytest.raises(ValueError, match=r"valid labels must lie"):
        cal.fit(logits, labels)


def test_all_masked_out_raises():
    cal = TemperatureScaling(device="cpu")
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=23)
    mask = torch.zeros(2, 4, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="no valid voxels"):
        cal.fit(logits, labels, mask=mask)


def test_bad_input_type_rejected():
    with pytest.raises(ValueError, match="input_type"):
        TemperatureScaling(input_type="scores", device="cpu")


def test_numpy_inputs_accepted():

    logits, labels = synthetic_logits((2, 3, 5, 5), seed=24)
    cal = TemperatureScaling(device="cpu")
    out = cal.fit_transform(logits.numpy(), labels.numpy())
    assert out.shape == to_probs(logits).shape


def test_read_only_numpy_arrays_accepted_under_warnings_as_errors():
    import warnings

    logits, labels = synthetic_logits((2, 3, 5, 5), seed=70)
    x, y = logits.numpy(), labels.numpy()
    probs = to_probs(logits).numpy()
    for array in (x, y, probs):
        array.setflags(write=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cal = TemperatureScaling(max_iter=5, device="cpu").fit(x, y)
        assert cal.transform(x).shape == logits.shape
        assert math.isfinite(negative_log_likelihood(probs, y))
        assert math.isfinite(reliability_curve(probs, y, n_bins=5).ece)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_safe_log_preserves_the_documented_floor_in_low_precision(dtype):
    from fiducio.utils import safe_log

    out = safe_log(torch.tensor([0.0], dtype=dtype))
    assert torch.isfinite(out).all()
    assert float(out) == pytest.approx(math.log(1e-12), abs=0.05)
    assert float(safe_log(torch.tensor([1.0], dtype=dtype))) == 0.0


def test_complex_predictions_are_rejected_not_silently_cast():
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=71)
    complex_logits = logits.to(torch.complex64)
    with pytest.raises(TypeError, match="complex"):
        TemperatureScaling(max_iter=1, device="cpu").fit(complex_logits, labels)
    with pytest.raises(TypeError, match="complex"):
        TemperatureScaling(max_iter=1, device="cpu").fit(
            logits.numpy().astype(np.complex64), labels
        )
    complex_probs = to_probs(logits).to(torch.complex64)
    with pytest.raises(TypeError, match="complex"):
        negative_log_likelihood(complex_probs, labels)
    with pytest.raises(TypeError, match="complex"):
        reliability_curve(complex_probs, labels)


def test_ignore_index_voxels_are_not_validated():
    z = torch.randn(1, 3, 2, 2)
    y = torch.zeros(1, 2, 2, dtype=torch.long)
    y[0, 0, 0] = -100
    z[0, :, 0, 0] = float("nan")
    cal = TemperatureScaling(max_iter=2, device="cpu").fit(z, y)
    assert cal.is_fitted
    # The same voxel is validated once it carries a real label.
    y[0, 0, 0] = 0
    with pytest.raises(ValueError, match="non-finite"):
        TemperatureScaling(max_iter=2, device="cpu").fit(z, y)


def test_ignore_index_voxels_are_not_validated_in_validation_data():
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=72)
    val_predictions = logits.clone()
    val_targets = labels.clone()
    val_targets[0, 0, 0] = -100
    val_predictions[0, :, 0, 0] = float("nan")
    TemperatureScaling(max_iter=5, patience=2, device="cpu").fit(
        logits, labels, val_predictions=val_predictions, val_targets=val_targets
    )


def test_isotonic_validation_data_has_a_class_specific_message():
    from fiducio import BinaryIsotonicCalibration

    logits, labels = synthetic_logits((2, 2, 4, 4), seed=73)
    with pytest.raises(ValueError, match="BinaryIsotonicCalibration.*early stopping"):
        BinaryIsotonicCalibration(device="cpu").fit(
            logits, labels, val_predictions=logits, val_targets=labels
        )


def test_masked_transform_output_is_valid_probs_input():
    logits, labels = synthetic_logits((2, 3, 6, 6), seed=26)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    mask[:, :2] = False
    cal = MatrixScaling(device="cpu").fit(logits, labels, mask=mask)
    probs = cal.transform(logits, mask=mask)
    valid = probs.sum(dim=1)[mask]
    assert torch.allclose(valid, torch.ones_like(valid), atol=1e-4)
    chained = TemperatureScaling(input_type="probs", device="cpu").fit(probs, labels, mask=mask)
    assert chained.transform(probs, mask=mask).shape == probs.shape
    assert math.isfinite(negative_log_likelihood(probs, labels, mask=mask))


def test_zero_rows_are_rejected_without_mask():
    probs = torch.zeros(2, 3, 4, 4)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="sum to 1"):
        negative_log_likelihood(probs, labels)


def test_non_tensor_predictions_and_targets_rejected():
    with pytest.raises(TypeError, match="torch.Tensor"):
        TemperatureScaling(device="cpu").fit([[1.0, 2.0]], [[0]])
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=27)
    with pytest.raises(TypeError, match="torch.Tensor"):
        TemperatureScaling(device="cpu").fit(logits, [[0]])


def test_invalid_n_bins_rejected():
    probs = to_probs(synthetic_logits((2, 3, 4, 4), seed=28)[0])
    labels = torch.zeros(2, 4, 4, dtype=torch.long)
    for n_bins in (0, True, 2.0, "3"):
        with pytest.raises(ValueError, match="n_bins"):
            reliability_curve(probs, labels, n_bins=n_bins)
    assert len(reliability_curve(probs, labels, n_bins=np.int64(4)).bin_counts) == 4


def test_two_channel_from_binary_rejects_bad_input_type():
    with pytest.raises(ValueError, match="input_type"):
        two_channel_from_binary(torch.randn(2, 1, 4, 4), input_type="scores")


def test_unrepresentable_initial_temperature_rejected():
    logits, labels = synthetic_logits((2, 3, 4, 4), seed=29)
    with pytest.raises(ValueError, match="representable in float32"):
        ArgmaxPreservingMatrixScaling(device="cpu", init_floor=1e-46, max_iter=1).fit(
            logits, labels
        )


def test_duplicate_registry_id_rejected():
    from fiducio.registry import register_calibrator

    class Other(TemperatureScaling):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_calibrator("temperature_scaling")(Other)


def test_half_precision_probabilities_accepted():
    logits, labels = synthetic_logits((2, 5, 64, 64), seed=3)
    for dtype in (torch.float16, torch.bfloat16):
        probs = torch.softmax(logits.to(dtype), dim=1)
        assert math.isfinite(expected_calibration_error(probs, labels))
        TemperatureScaling(input_type="probs", max_iter=5, device="cpu").fit(probs, labels)


def test_masked_out_padding_is_not_validated():
    logits, labels = synthetic_logits((2, 3, 6, 6), seed=5)
    probs = to_probs(logits)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)
    mask[:, :2] = False
    padded_logits = logits.clone()
    padded_logits[:, :, :2] = float("nan")
    padded_probs = probs.clone()
    padded_probs[:, :, :2] = float("nan")
    expected = expected_calibration_error(probs, labels, mask=mask)
    assert expected_calibration_error(padded_probs, labels, mask=mask) == expected
    for input_type, inputs in (("logits", padded_logits), ("probs", padded_probs)):
        cal = TemperatureScaling(input_type=input_type, max_iter=5, device="cpu")
        out = cal.fit(inputs, labels, mask=mask).transform(inputs, mask=mask)
        assert torch.equal(out[:, :, :2], torch.zeros_like(out[:, :, :2]))
        assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="non-finite"):
        expected_calibration_error(padded_probs, labels)


def test_float64_beyond_float32_range_rejected():
    logits, labels = synthetic_logits((1, 3, 4, 4), seed=6)
    logits = logits.double()
    cal = TemperatureScaling(max_iter=5, device="cpu").fit(logits, labels)
    logits[0, 0, 0, 0] = 1e39
    with pytest.raises(ValueError, match="float32 range"):
        cal.transform(logits)


def test_default_ignore_index_is_exported():
    from fiducio.utils import DEFAULT_IGNORE_INDEX

    assert DEFAULT_IGNORE_INDEX == -100


def test_max_iter_accepts_numpy_integers_only():
    assert TemperatureScaling(max_iter=np.int64(5)).max_iter == 5
    for bad in (True, 5.0, "5"):
        with pytest.raises(ValueError, match="must be an integer"):
            TemperatureScaling(max_iter=bad)
    with pytest.raises(ValueError, match="> 0"):
        TemperatureScaling(max_iter=-1)


def test_integer_probabilities_validate_like_floats():
    """Integer 0/1 probabilities are valid input; finfo is only for float dtypes."""
    from fiducio.utils import validate_predictions

    one_hot = torch.tensor([[1, 0], [0, 1]])
    assert validate_predictions(one_hot, input_type="probs") == 2
    assert validate_predictions(one_hot.to(torch.bool), input_type="probs") == 2
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_predictions(torch.tensor([[2, 0]]), input_type="probs")


def test_uint64_labels_beyond_int64_are_rejected_not_wrapped():
    from fiducio.utils.tensors import integer_targets

    huge = torch.tensor([2**63], dtype=torch.uint64)
    with pytest.raises(ValueError, match="representable in int64"):
        integer_targets(huge, device=torch.device("cpu"))
    # 2**64 - 100 wraps to -100, which a metric would silently drop as ignore_index.
    wrapped = torch.tensor([[2**64 - 100]], dtype=torch.uint64)
    with pytest.raises(ValueError, match="integer labels"):
        negative_log_likelihood(torch.tensor([[[1.0], [0.0]]]), wrapped)


def test_validate_targets_rejects_fractional_and_nonfinite_float_labels():
    from fiducio.utils import validate_targets

    probs = torch.tensor([[0.5, 0.5]])
    for bad in (1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="integer labels"):
            validate_targets(probs, torch.tensor([bad]), None, num_classes=2, ignore_index=-100)
    validate_targets(probs, torch.tensor([1.0]), None, num_classes=2, ignore_index=-100)
