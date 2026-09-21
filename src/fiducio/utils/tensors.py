"""Tensor validation, conversion and (un)flattening helpers.

Fiducio uses a single, explicit tensor convention so that the same calibration
math works for 2D, 3D and general n-D segmentation:

* **predictions** are channel-first with the class axis at dimension 1:
  ``(B, C, *spatial)`` (e.g. ``(B, C, H, W)`` in 2D, ``(B, C, D, H, W)`` in 3D).
  A plain ``(N, C)`` tensor (no spatial axes) is also accepted.
* **targets** are integer class indices of shape ``(B, *spatial)``.
* **mask**, when given, is a boolean tensor of shape ``(B, *spatial)`` where
  ``True`` marks valid positions.

The class axis is *always* dimension 1. Fiducio never tries to guess it.
"""

from __future__ import annotations

import math

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray

DEFAULT_IGNORE_INDEX = -100
_PROB_SUM_ATOL = 1e-3


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device specification to a concrete :class:`torch.device`.

    ``None`` selects CUDA when available, otherwise CPU.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_tensor(
    x: ArrayLike,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert array-like input to a tensor without unnecessary copies."""
    if isinstance(x, np.ndarray):
        tensor = torch.from_numpy(x)
    elif torch.is_tensor(x):
        tensor = x
    else:
        raise TypeError(f"Expected a torch.Tensor or numpy.ndarray, got {type(x)!r}")
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def safe_log(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable log of probabilities."""
    return torch.log(probs.clamp_min(eps))


def positive_finite(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate a scalar hyperparameter before constructing tensors."""
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        domain = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {domain}")
    return value


def inverse_softplus(value: float, device: torch.device) -> torch.Tensor:
    """Stable inverse for a positive, float32-representable softplus value."""
    value = positive_finite(value, "initial value")
    v = torch.tensor(value, dtype=torch.float32, device=device)
    if not torch.isfinite(v) or v <= 0:
        raise ValueError("initial value must be positive and representable in float32")
    return v + torch.log(-torch.expm1(-v))


def integer_targets(targets: ArrayLike, *, device: torch.device) -> torch.Tensor:
    """Reject fractional/non-finite labels before converting to int64."""
    y = to_tensor(targets, device=device)
    if y.is_complex() or not torch.isfinite(y).all():
        raise ValueError("targets must contain finite integer labels")
    if y.is_floating_point() and (not torch.equal(y, y.trunc()) or
                                 (y >= 2**63).any() or (y < -(2**63)).any()):
        raise ValueError("targets must contain integer labels representable in int64")
    return y.long()


def validate_mask(predictions: torch.Tensor, mask: torch.Tensor | None) -> None:
    """Require the exact spatial layout, never implicit broadcasting."""
    expected = (predictions.shape[0], *predictions.shape[2:])
    if mask is not None and tuple(mask.shape) != expected:
        raise ValueError(f"mask must have shape {expected}; got {tuple(mask.shape)}")


def validate_predictions(
    predictions: torch.Tensor,
    *,
    input_type: str,
    expected_num_classes: int | None = None,
) -> int:
    """Validate a predictions tensor and return the number of classes.

    Raises
    ------
    TypeError, ValueError
        If the tensor is malformed for the declared ``input_type``.
    """
    if not torch.is_tensor(predictions):
        raise TypeError("predictions must be a torch.Tensor")
    if predictions.ndim < 2:
        raise ValueError(
            "predictions must have shape (B, C, *spatial) with the class axis at "
            f"dimension 1; got a {predictions.ndim}-D tensor"
        )
    num_classes = int(predictions.shape[1])
    if num_classes < 2:
        raise ValueError(
            "the class axis (dimension 1) must contain at least 2 classes; "
            f"got C={num_classes}. Binary segmentation should be provided as 2 channels."
        )
    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ValueError(
            f"this calibrator was fitted for C={expected_num_classes} classes, "
            f"but received C={num_classes}"
        )
    if not torch.isfinite(predictions).all():
        raise ValueError("predictions contain non-finite values (nan/inf)")
    if input_type == "probs" and predictions.numel():
        pmin = float(predictions.min())
        pmax = float(predictions.max())
        if pmin < 0.0 or pmax > 1.0:
            raise ValueError(
                "input_type='probs' but values fall outside [0, 1]; "
                "pass input_type='logits' for unnormalised scores"
            )
        sums = predictions.sum(dim=1)
        if not torch.allclose(sums, torch.ones_like(sums), atol=_PROB_SUM_ATOL, rtol=0):
            raise ValueError(
                "input_type='probs' but probabilities do not sum to 1 along the "
                "class axis (dimension 1)"
            )
    return num_classes


def validate_targets(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    """Validate label and mask tensors against ``predictions``."""
    if not torch.is_tensor(targets):
        raise TypeError("targets must be a torch.Tensor")
    expected_shape = (predictions.shape[0],) + tuple(predictions.shape[2:])
    if tuple(targets.shape) != expected_shape:
        raise ValueError(
            f"targets must have shape {expected_shape} (predictions without the "
            f"class axis); got {tuple(targets.shape)}"
        )
    validate_mask(predictions, mask)
    valid = targets != ignore_index
    if mask is not None:
        valid = valid & mask.to(torch.bool)
    if valid.any():
        valid_labels = targets[valid]
        lo = int(valid_labels.min())
        hi = int(valid_labels.max())
        if lo < 0 or hi >= num_classes:
            raise ValueError(
                f"valid labels must lie in [0, {num_classes - 1}]; got [{lo}, {hi}]"
            )


def class_last_flatten(z: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten ``(B, C, *spatial)`` to ``(B*prod(spatial), C)``.

    Returns the flattened tensor and the original shape, which
    :func:`restore_class_first` uses to invert the operation.
    """
    original_shape = tuple(z.shape)
    num_classes = original_shape[1]
    flat = z.movedim(1, -1).reshape(-1, num_classes)
    return flat, original_shape


def restore_class_first(flat: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
    """Invert :func:`class_last_flatten`."""
    batch = original_shape[0]
    spatial = original_shape[2:]
    num_classes = flat.shape[1]
    reshaped = flat.reshape((batch, *spatial, num_classes))
    return reshaped.movedim(-1, 1)


def flatten_valid(
    z: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the valid ``(N, C)`` rows and ``(N,)`` labels for fitting.

    Positions whose label equals ``ignore_index`` or whose mask is ``False`` are
    dropped.
    """
    flat, _ = class_last_flatten(z)
    targets_flat = targets.reshape(-1)
    valid = targets_flat != ignore_index
    if mask is not None:
        valid = valid & mask.reshape(-1).to(torch.bool)
    return flat[valid], targets_flat[valid].long()


def apply_mask_to_probabilities(
    probs: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Zero out probabilities at masked-out (``False``) positions.

    ``mask`` has shape ``(B, *spatial)`` and is broadcast across the class axis.
    Masked-out voxels are set to 0 across all classes; valid voxels keep a
    proper distribution that sums to 1.
    """
    if mask is None:
        return probs
    validate_mask(probs, mask)
    mask_b = mask.to(dtype=probs.dtype).unsqueeze(1)
    return probs * mask_b


def two_channel_from_binary(
    scores: ArrayLike, *, input_type: str = "logits"
) -> torch.Tensor:
    """Convert single-channel binary outputs to the two-channel form Fiducio uses.

    Fiducio represents binary segmentation as two channels (``C = 2``). Use this
    helper to convert a single-channel sigmoid output, whose class axis at
    dimension 1 has size 1 (shape ``(B, 1, *spatial)``), into ``(B, 2, *spatial)``.
    Class 1 is the positive class.

    Parameters
    ----------
    scores:
        ``(B, 1, *spatial)`` score for the positive class.
    input_type:
        ``"logits"`` — a sigmoid logit ``z``; returns logits ``[0, z]`` whose
        softmax equals ``[1 - sigmoid(z), sigmoid(z)]``. ``"probs"`` — a
        probability ``p``; returns ``[1 - p, p]``.
    """
    x = to_tensor(scores, dtype=torch.float32)
    if x.ndim < 2 or x.shape[1] != 1:
        raise ValueError(
            "scores must have a singleton class axis at dimension 1, i.e. shape "
            f"(B, 1, *spatial); got {tuple(x.shape)}"
        )
    if input_type == "logits":
        return torch.cat([torch.zeros_like(x), x], dim=1)
    if input_type == "probs":
        return torch.cat([1.0 - x, x], dim=1)
    raise ValueError(f"input_type must be 'logits' or 'probs', got {input_type!r}")
