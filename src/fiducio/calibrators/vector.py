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
    optimizer:
        ``"adam"`` (default) or ``"lbfgs"``.
    lr:
        Learning rate. Defaults to ``0.1`` (Adam) or ``1.0`` (L-BFGS).
    max_iter:
        Maximum optimizer iterations. Defaults to ``200`` (Adam) or ``100``
        (L-BFGS).
    lambda_reg:
        L2 penalty pulling the scale vector towards 1.
    mu_reg:
        L2 penalty on the bias vector.
    patience, min_delta, lr_patience, lr_factor:
        Optional validation-based early stopping (Adam only), see
        :class:`fiducio.Calibrator`. ``fit`` then requires ``val_predictions``
        and ``val_targets``.
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`.

    References
    ----------
    Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML.
    """

    _mode = "diagonal"
    _input_space = "logits"
