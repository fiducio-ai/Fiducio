"""Calibrator implementations.

Importing this package registers every calibrator id in the controlled registry
used by :func:`fiducio.load_calibrator`.
"""

from __future__ import annotations

from .class_conditional import (
    ArgmaxPreservingMatrixScaling,
    ClassConditionalMatrixScaling,
    OrderPreservingMatrixScaling,
)
from .dirichlet import DirichletCalibration
from .ensemble_temperature import EnsembleTemperatureScaling
from .matrix import MatrixScaling, TranslationInvariantMatrixScaling
from .temperature import TemperatureScaling
from .vector import VectorScaling

# Short aliases following common calibration-literature shorthands.
TS = TemperatureScaling
ETS = EnsembleTemperatureScaling
VS = VectorScaling
MS = MatrixScaling
CMS = ClassConditionalMatrixScaling
CMSAP = ArgmaxPreservingMatrixScaling
CMSOP = OrderPreservingMatrixScaling

__all__ = [
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
]
