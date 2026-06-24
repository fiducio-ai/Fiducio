"""Matrix scaling calibrators (full and translation-invariant)."""

from __future__ import annotations

from ..registry import register_calibrator
from ._affine import _AffineCalibrator


@register_calibrator("matrix_scaling")
class MatrixScaling(_AffineCalibrator):
    """Matrix scaling: ``softmax(W z + b)`` with a full ``C x C`` matrix.

    Optional off-diagonal / bias L2 regularisation (ODIR) keeps the matrix close
    to the identity, which is recommended when the number of classes is large
    relative to the calibration set.

    Parameters
    ----------
    optimizer:
        ``"adam"`` (default) or ``"lbfgs"``.
    lr:
        Learning rate. Defaults to ``0.1`` (Adam) or ``1.0`` (L-BFGS).
    max_iter:
        Maximum optimizer iterations. Defaults to ``200`` (Adam) or ``100``
        (L-BFGS).
    lambda_reg:
        L2 penalty on the off-diagonal entries of ``W`` (ODIR).
    mu_reg:
        L2 penalty on the bias vector (ODIR).
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`.

    References
    ----------
    Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML;
    Kull et al. (2019) for off-diagonal/intercept regularisation.
    """

    _mode = "matrix"
    _input_space = "logits"


@register_calibrator("translation_invariant_matrix_scaling")
class TranslationInvariantMatrixScaling(_AffineCalibrator):
    """Constrained matrix scaling that is invariant to logit translations.

    The matrix ``W`` is constrained so that every row sums to zero. Because
    ``softmax`` ignores constant shifts of its input, this makes the calibrated
    output invariant to adding the same constant to every input logit
    (``g(z + c·1) = g(z)``), removing the gauge redundancy of unconstrained
    matrix scaling.

    Parameters are identical to :class:`MatrixScaling`.
    """

    _mode = "matrix_ti"
    _input_space = "logits"
