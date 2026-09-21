"""Early-stopping configuration shared by every gradient-fitted calibrator."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StoppingRule:
    """Validation-based stopping rule for the Adam optimizer.

    Attributes
    ----------
    patience:
        Stop after this many consecutive iterations without a validation-NLL
        improvement of more than ``min_delta``. ``None`` disables stopping (the
        learning-rate schedule below may still be active).
    min_delta:
        Minimum decrease of the validation NLL that counts as an improvement.
    lr_patience:
        If set, multiply the learning rate by ``lr_factor`` after this many
        iterations without improvement (``torch``'s ``ReduceLROnPlateau``).
    lr_factor:
        Learning-rate decay factor, strictly between 0 and 1.
    """

    patience: int | None = None
    min_delta: float = 0.0
    lr_patience: int | None = None
    lr_factor: float = 0.1


def resolve_stopping(
    optimizer: str,
    patience: int | None,
    min_delta: float,
    lr_patience: int | None,
    lr_factor: float,
) -> StoppingRule | None:
    """Validate early-stopping arguments and return the rule, or ``None`` if unused.

    Raises
    ------
    ValueError
        If an argument is out of range, or early stopping is requested with an
        optimizer other than ``"adam"`` (L-BFGS runs as a single optimizer call
        and cannot be interrupted between iterations).
    """
    if not math.isfinite(float(min_delta)) or float(min_delta) < 0:
        raise ValueError("min_delta must be >= 0 and finite")
    if not 0.0 < float(lr_factor) < 1.0:
        raise ValueError("lr_factor must be in (0, 1)")
    for name, value in (("patience", patience), ("lr_patience", lr_patience)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be > 0 and an integer")
    if patience is None and lr_patience is None:
        return None
    if optimizer != "adam":
        raise ValueError("early stopping (patience / lr_patience) requires optimizer='adam'")
    return StoppingRule(
        patience=None if patience is None else int(patience),
        min_delta=float(min_delta),
        lr_patience=None if lr_patience is None else int(lr_patience),
        lr_factor=float(lr_factor),
    )
