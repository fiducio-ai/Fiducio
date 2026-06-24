"""Vector scaling calibrator."""

from __future__ import annotations

from ..registry import register_calibrator
from ._affine import _AffineCalibrator


@register_calibrator("vector_scaling")
class VectorScaling(_AffineCalibrator):
    """Vector scaling: ``softmax(diag(w) z + b)``.

    A per-class generalisation of temperature scaling with one scale ``w_c`` and
    one bias ``b_c`` per class. Operates on logits (or ``log`` of probabilities
    when ``input_type='probs'``).

    Parameters
    ----------
    max_iter:
        Maximum L-BFGS iterations.
    lr:
        L-BFGS learning rate.
    lambda_reg:
        L2 penalty pulling the scale vector towards 1.
    mu_reg:
        L2 penalty on the bias vector.
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`.

    References
    ----------
    Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML.
    """

    _mode = "diagonal"
    _input_space = "logits"
