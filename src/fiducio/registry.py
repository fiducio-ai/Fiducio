"""Controlled registry of calibrator classes.

Persistence stores a stable ``calibrator_id`` string rather than a Python module
path. Loading resolves that id through this registry, so Fiducio never imports an
arbitrary module named inside a saved file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .base import Calibrator

_REGISTRY: dict[str, type[Calibrator]] = {}

T = TypeVar("T", bound="type[Calibrator]")


def register_calibrator(calibrator_id: str) -> Callable[[T], T]:
    """Class decorator that registers a calibrator under a stable id.

    Parameters
    ----------
    calibrator_id:
        Stable, unique identifier persisted in saved files. Changing it breaks
        backward compatibility of previously saved calibrators.
    """

    def decorator(cls: T) -> T:
        if calibrator_id in _REGISTRY and _REGISTRY[calibrator_id] is not cls:
            raise ValueError(f"calibrator id {calibrator_id!r} is already registered")
        cls.calibrator_id = calibrator_id
        _REGISTRY[calibrator_id] = cls
        return cls

    return decorator


def get_calibrator_class(calibrator_id: str) -> type[Calibrator]:
    """Return the calibrator class registered under ``calibrator_id``."""
    try:
        return _REGISTRY[calibrator_id]
    except KeyError:
        raise KeyError(
            f"unknown calibrator id {calibrator_id!r}; known ids: "
            f"{sorted(_REGISTRY)}"
        ) from None


def registered_ids() -> list[str]:
    """Return the sorted list of registered calibrator ids."""
    return sorted(_REGISTRY)
