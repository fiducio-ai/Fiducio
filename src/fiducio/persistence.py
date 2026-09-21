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


def _validate_state(calibrator: Any, state: dict[str, Any], classes: int | None,
                    fitted: bool) -> None:
    """Validate built-in parameter layouts, including legacy full-matrix MSc."""
    cid = calibrator.calibrator_id
    expected: dict[str, tuple[int, ...]] = {}
    if cid in {"temperature_scaling", "ensemble_temperature_scaling"}:
        expected["temperature"] = ()
        if cid == "ensemble_temperature_scaling":
            expected["weights"] = (3,)
    elif classes is not None:
        if cid in {"vector_scaling", "matrix_scaling", "dirichlet_calibration",
                   "translation_invariant_matrix_scaling"}:
            shape = (classes,) if cid == "vector_scaling" else (classes, classes)
            if cid == "translation_invariant_matrix_scaling" and state.get("row_sum") is not None:
                shape = (classes, classes - 1)
                expected["row_sum"] = ()
            expected.update(weight=shape, bias=(classes,))
        elif cid in {"class_conditional_matrix_scaling", "argmax_preserving_matrix_scaling",
                     "order_preserving_matrix_scaling"}:
            k = classes if cid == "class_conditional_matrix_scaling" else classes - 1
            expected.update(raw_b=(classes, k, k), raw_mu=(classes, k))
    template = calibrator._get_state()
    if set(state) - set(template):
        raise ValueError("unexpected saved state fields")
    for key in template:
        # row_sum was absent in pre-release MSc checkpoints.
        if key not in state and not (key == "row_sum" and
                                    cid == "translation_invariant_matrix_scaling"):
            raise ValueError(f"missing saved parameter {key}")
    for name, value in state.items():
        if value is None:
            if name in expected or (fitted and name != "row_sum"):
                raise ValueError(f"missing saved parameter {name}")
            continue
        if name not in expected and not fitted and template.get(name) is None:
            raise ValueError(f"unexpected parameter {name} on an unfitted calibrator")
        if not torch.is_tensor(value) or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"saved parameter {name} must be a finite floating tensor")
        if name in expected and tuple(value.shape) != expected[name]:
            raise ValueError(f"invalid shape for saved parameter {name}")
        if name == "temperature" and value <= 0:
            raise ValueError("saved temperature must be positive")
        if name == "weights" and ((value < 0).any() or
                                  not torch.isclose(value.sum(), value.new_tensor(1.), atol=1e-6, rtol=0)):
            raise ValueError("saved mixture weights must be a probability distribution")


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
    The payload is read exclusively with ``weights_only=True``, which restricts
    deserialization to tensors and basic Python types. Only load files you trust.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"calibrator file not found: {path}")

    location: MapLocation = "cpu" if map_location is None else map_location
    payload = torch.load(os.fspath(path), map_location=location, weights_only=True)

    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ValueError(f"{path} is not a Fiducio calibrator file")

    if type(payload.get("format_version")) is not int or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported or missing Fiducio format_version")
    required = {"fiducio_version", "calibrator_id", "config", "num_classes", "fitted", "state"}
    if not required <= payload.keys():
        raise ValueError("missing required Fiducio payload fields")
    if not isinstance(payload["calibrator_id"], str) or not isinstance(payload["fiducio_version"], str):
        raise ValueError("invalid calibrator id or package version")
    if not isinstance(payload["config"], dict) or not isinstance(payload["state"], dict):
        raise ValueError("config and state must be dictionaries")
    fitted, classes = payload["fitted"], payload["num_classes"]
    if type(fitted) is not bool or (fitted and (type(classes) is not int or classes < 2)):
        raise ValueError("invalid fitted status or number of classes")
    if not fitted and classes is not None:
        raise ValueError("an unfitted calibrator must have num_classes=None")

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

    config = dict(payload["config"])
    if "device" in config:
        raise ValueError("saved config must not override map_location")
    device = torch.device(location) if isinstance(location, (str, torch.device)) else None
    try:
        calibrator = cls(device=device, **config)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid saved calibrator configuration: {exc}") from exc
    _validate_state(calibrator, payload["state"], classes, fitted)
    calibrator._num_classes = classes
    calibrator._set_state(payload["state"])
    calibrator._fitted = fitted
    return calibrator
