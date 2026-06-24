"""Internal utilities for Fiducio (logging and tensor handling)."""

from __future__ import annotations

from .logging import get_logger
from .tensors import (
    DEFAULT_IGNORE_INDEX,
    apply_mask_to_probabilities,
    class_last_flatten,
    flatten_valid,
    resolve_device,
    restore_class_first,
    safe_log,
    to_tensor,
    validate_predictions,
    validate_targets,
)

__all__ = [
    "DEFAULT_IGNORE_INDEX",
    "apply_mask_to_probabilities",
    "class_last_flatten",
    "flatten_valid",
    "get_logger",
    "resolve_device",
    "restore_class_first",
    "safe_log",
    "to_tensor",
    "validate_predictions",
    "validate_targets",
]
