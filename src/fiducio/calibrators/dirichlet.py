"""Dirichlet calibration."""

from __future__ import annotations

from ..registry import register_calibrator
from ._affine import _AffineCalibrator


@register_calibrator("dirichlet_calibration")
class DirichletCalibration(_AffineCalibrator):
    """Dirichlet calibration: ``softmax(W log p + b)``.

    A log-linear transform in probability space, equivalent to matrix scaling
    applied to log-probabilities. When ``input_type='logits'`` the inputs are
    converted to log-probabilities with ``log_softmax`` before calibration.
    Off-diagonal / bias L2 regularisation (ODIR) is recommended.

    Parameters
    ----------
    max_iter:
        Maximum L-BFGS iterations.
    lr:
        L-BFGS learning rate.
    lambda_reg:
        L2 penalty on the off-diagonal entries of ``W`` (ODIR).
    mu_reg:
        L2 penalty on the bias vector (ODIR).
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`.

    References
    ----------
    Kull et al. (2019), *Beyond temperature scaling: Obtaining well-calibrated
    multiclass probabilities with Dirichlet calibration*, NeurIPS.
    """

    _mode = "matrix"
    _input_space = "logprobs"
