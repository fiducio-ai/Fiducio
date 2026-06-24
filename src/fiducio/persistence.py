"""Saving and loading calibrators via a controlled registry.

The on-disk format is a single :func:`torch.save` payload containing:

* ``format`` / ``format_version`` – format identification;
* ``fiducio_version`` – the Fiducio version that wrote the file;
* ``calibrator_id`` – a stable id resolved through the controlled registry;
* ``config`` – constructor keyword arguments (no device);
* ``num_classes`` / ``fitted`` – fit-time metadata;
* ``state`` – learned parameters as tensors / plain Python types.

Loading never imports an arbitrary module path stored in the file: the
``calibrator_id`` is looked up in :mod:`fiducio.registry`. By default the payload
is read with ``weights_only=True`` and onto CPU.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .registry import get_calibrator_class
from .utils import get_logger

FORMAT_NAME = "fiducio-calibrator"
FORMAT_VERSION = 1

PathLike = str | os.PathLike
MapLocation = str | torch.device

_logger = get_logger(__name__)


def _major(version_str: str | None) -> int | None:
    if not version_str:
        return None
    try:
        return int(str(version_str).split(".")[0])
    except ValueError:
        return None


def _fiducio_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("fiducio")
        except PackageNotFoundError:
            return "0.0.0+unknown"
    except Exception:  # pragma: no cover - importlib always present on 3.10+
        return "0.0.0+unknown"


def _to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


def save_calibrator(calibrator: Any, path: PathLike) -> None:
    """Serialize ``calibrator`` to ``path``.

    Parameters
    ----------
    calibrator:
        A fitted (or at least constructed) :class:`fiducio.Calibrator`.
    path:
        Destination file. The ``.pt`` extension is conventional.
    """
    if not getattr(calibrator, "calibrator_id", ""):
        raise ValueError(
            f"{type(calibrator).__name__} has no registered calibrator_id and "
            "cannot be saved"
        )
    directory = os.path.dirname(os.fspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    payload: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "fiducio_version": _fiducio_version(),
        "calibrator_id": calibrator.calibrator_id,
        "config": calibrator.get_config(),
        "num_classes": calibrator.num_classes,
        "fitted": calibrator.is_fitted,
        "state": _to_cpu(calibrator._get_state()),
    }
    torch.save(payload, os.fspath(path))


def load_calibrator(
    path: PathLike,
    map_location: MapLocation | None = "cpu",
) -> Any:
    """Load a calibrator previously written by :func:`save_calibrator`.

    Parameters
    ----------
    path:
        File to load.
    map_location:
        Device for the loaded tensors and the reconstructed calibrator. Defaults
        to ``"cpu"``; the device saved on the original calibrator is **not**
        forced. Pass ``"cuda"`` to load onto a GPU.

    Notes
    -----
    The payload is read with ``weights_only=True`` when supported, which restricts
    deserialization to tensors and basic Python types. Only load files you trust.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibrator file not found: {path}")

    location: MapLocation = "cpu" if map_location is None else map_location
    try:
        payload = torch.load(os.fspath(path), map_location=location, weights_only=True)
    except TypeError:  # pragma: no cover - very old torch without weights_only
        payload = torch.load(os.fspath(path), map_location=location)

    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(f"{path} is not a Fiducio calibrator file")

    saved_version = payload.get("fiducio_version")
    saved_major, current_major = _major(saved_version), _major(_fiducio_version())
    if saved_major is not None and current_major is not None and saved_major != current_major:
        _logger.warning(
            "calibrator was saved with fiducio %s but the installed version is %s; "
            "loading may not be fully compatible",
            saved_version,
            _fiducio_version(),
        )

    calibrator_id = payload["calibrator_id"]
    cls = get_calibrator_class(calibrator_id)

    config = dict(payload.get("config", {}))
    device = torch.device(location) if isinstance(location, (str, torch.device)) else None
    calibrator = cls(device=device, **config)
    calibrator._num_classes = payload.get("num_classes")
    calibrator._set_state(payload.get("state", {}))
    calibrator._fitted = bool(payload.get("fitted", True))
    return calibrator
