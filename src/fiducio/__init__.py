"""Fiducio: model-agnostic post-hoc calibration for 2D/3D semantic segmentation.

Fiducio provides a consistent ``fit`` / ``transform`` API for post-hoc
calibrators that operate on the logits or probabilities of any segmentation
model. See https://fiducio-ai.github.io/Fiducio/ for documentation.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .base import Calibrator, NotFittedError
from .calibrators import (
    CMS,
    CMSAP,
    CMSOP,
    ETS,
    MS,
    TS,
    VS,
    ArgmaxPreservingMatrixScaling,
    ClassConditionalMatrixScaling,
    DirichletCalibration,
    EnsembleTemperatureScaling,
    MatrixScaling,
    OrderPreservingMatrixScaling,
    TemperatureScaling,
    TranslationInvariantMatrixScaling,
    VectorScaling,
)
from .metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from .persistence import load_calibrator, save_calibrator
from .registry import registered_ids

try:
    __version__ = version("fiducio")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # core
    "Calibrator",
    "NotFittedError",
    # calibrators
    "TemperatureScaling",
    "EnsembleTemperatureScaling",
    "VectorScaling",
    "MatrixScaling",
    "TranslationInvariantMatrixScaling",
    "DirichletCalibration",
    "ClassConditionalMatrixScaling",
    "ArgmaxPreservingMatrixScaling",
    "OrderPreservingMatrixScaling",
    # aliases
    "TS",
    "ETS",
    "VS",
    "MS",
    "CMS",
    "CMSAP",
    "CMSOP",
    # persistence
    "save_calibrator",
    "load_calibrator",
    "registered_ids",
    # metrics
    "negative_log_likelihood",
    "expected_calibration_error",
    "brier_score",
]
