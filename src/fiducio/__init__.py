"""Fiducio: model-agnostic post-hoc calibration for 2D/3D semantic segmentation.

Fiducio provides a consistent ``fit`` / ``transform`` API for post-hoc
calibrators that operate on the logits or probabilities of any segmentation
model. See https://fiducio-ai.github.io/Fiducio/ for documentation.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import plots
from .base import Calibrator, NotFittedError
from .calibrators import (
    CDC,
    CMS,
    CMSAP,
    CMSOP,
    ETS,
    MS,
    TS,
    VS,
    ArgmaxPreservingMatrixScaling,
    ClassConditionalMatrixScaling,
    CMSap,
    CMSop,
    DirichletCalibration,
    EnsembleTemperatureScaling,
    MatrixScaling,
    MSc,
    OrderPreservingMatrixScaling,
    TemperatureScaling,
    TranslationInvariantMatrixScaling,
    VectorScaling,
)
from .metrics import (
    ReliabilityCurve,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)
from .persistence import load_calibrator, save_calibrator
from .registry import get_calibrator_class, registered_ids
from .utils import two_channel_from_binary

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
    "MSc",
    "CMS",
    "CMSAP",
    "CMSOP",
    # paper-shorthand aliases
    "CDC",
    "CMSap",
    "CMSop",
    # persistence & registry
    "save_calibrator",
    "load_calibrator",
    "registered_ids",
    "get_calibrator_class",
    # metrics
    "negative_log_likelihood",
    "expected_calibration_error",
    "brier_score",
    "reliability_curve",
    "ReliabilityCurve",
    # helpers
    "two_channel_from_binary",
    # plotting (optional, requires fiducio[plots])
    "plots",
]
