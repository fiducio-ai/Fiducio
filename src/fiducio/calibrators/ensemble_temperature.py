"""Ensemble Temperature Scaling (ETS)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..base import Calibrator, DeviceLike
from ..registry import register_calibrator
from ..utils import class_last_flatten, restore_class_first, safe_log
from ._optim import minimize, resolve_optimizer

_EPS = 1e-12
_T_EPS = 1e-6


@register_calibrator("ensemble_temperature_scaling")
class EnsembleTemperatureScaling(Calibrator):
    r"""Ensemble Temperature Scaling.

    Calibrated probabilities are a convex combination of a temperature-scaled
    distribution, the original distribution and the uniform distribution:

    .. math:: p = w_0\,\mathrm{softmax}(z/T) + w_1\,\mathrm{softmax}(z) + w_2\,u

    where :math:`w` lies on the 3-simplex and :math:`u` is uniform. Fitting is
    done in two stages: first the temperature ``T`` (NLL), then the mixture
    weights ``w`` with ``T`` fixed.

    Parameters
    ----------
    init_temperature:
        Initial temperature for stage 1.
    optimizer:
        ``"adam"`` (default) or ``"lbfgs"``.
    lr:
        Learning rate. Defaults to ``0.1`` (Adam) or ``1.0`` (L-BFGS).
    max_iter:
        Maximum optimizer iterations per stage. Defaults to ``200`` (Adam) or
        ``100`` (L-BFGS).
    patience, min_delta, lr_patience, lr_factor:
        Optional validation-based early stopping (Adam only), see
        :class:`fiducio.Calibrator`. ``fit`` then requires ``val_predictions``
        and ``val_targets``.
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`.

    References
    ----------
    Zhang et al. (2020), *Mix-n-Match: Ensemble and Compositional Methods for
    Uncertainty Calibration in Deep Learning*, ICML.
    """

    _input_space = "logits"

    def __init__(
        self,
        *,
        init_temperature: float = 1.0,
        optimizer: str = "adam",
        lr: float | None = None,
        max_iter: int | None = None,
        patience: int | None = None,
        min_delta: float = 0.0,
        lr_patience: int | None = None,
        lr_factor: float = 0.1,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(input_type=input_type, ignore_index=ignore_index, device=device)
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")
        self.init_temperature = float(init_temperature)
        self.optimizer, self.lr, self.max_iter = resolve_optimizer(
            optimizer, lr, max_iter,
            adam_lr=0.1, lbfgs_lr=1.0, adam_max_iter=200, lbfgs_max_iter=100,
        )
        self._init_stopping(self.optimizer, patience, min_delta, lr_patience, lr_factor)
        self.temperature: float = float(init_temperature)
        self.weights: torch.Tensor = torch.tensor([1.0, 0.0, 0.0], device=self.device)

    def _fit_temperature(self, z_flat: torch.Tensor, y_flat: torch.Tensor) -> None:
        init = max(self.init_temperature, _T_EPS)
        raw = torch.log(torch.expm1(torch.tensor(init, device=self.device)))
        raw_t = raw.clone().requires_grad_(True)

        def loss_fn() -> torch.Tensor:
            temperature = F.softplus(raw_t) + _T_EPS
            return F.cross_entropy(z_flat / temperature, y_flat)

        val_fn = None
        if self._val is not None:
            z_val, y_val = self._val

            def val_fn() -> torch.Tensor:
                return F.cross_entropy(z_val / (F.softplus(raw_t) + _T_EPS), y_val)

        minimize(
            self.optimizer, [raw_t], loss_fn, lr=self.lr, max_iter=self.max_iter,
            val_fn=val_fn, stopping=self._stopping,
        )
        self.temperature = float((F.softplus(raw_t) + _T_EPS).detach().cpu().item())

    def _fit_weights(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        temperature = max(self.temperature, _T_EPS)
        p0 = F.softmax(z_flat / temperature, dim=1).detach()
        p1 = F.softmax(z_flat, dim=1).detach()
        p2 = torch.full_like(p0, 1.0 / float(num_classes))
        raw_w = torch.tensor([1.0, 0.0, 0.0], device=self.device).requires_grad_(True)

        def loss_fn() -> torch.Tensor:
            w = F.softmax(raw_w, dim=0)
            p = (w[0] * p0 + w[1] * p1 + w[2] * p2).clamp_min(_EPS)
            return F.nll_loss(torch.log(p), y_flat)

        val_fn = None
        if self._val is not None:
            z_val, y_val = self._val
            v0 = F.softmax(z_val / temperature, dim=1)
            v1 = F.softmax(z_val, dim=1)
            v2 = torch.full_like(v0, 1.0 / float(num_classes))

            def val_fn() -> torch.Tensor:
                w = F.softmax(raw_w, dim=0)
                p = (w[0] * v0 + w[1] * v1 + w[2] * v2).clamp_min(_EPS)
                return F.nll_loss(torch.log(p), y_val)

        minimize(
            self.optimizer, [raw_w], loss_fn, lr=self.lr, max_iter=self.max_iter,
            val_fn=val_fn, stopping=self._stopping,
        )
        self.weights = F.softmax(raw_w, dim=0).detach()

    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        self._fit_temperature(z_flat, y_flat)
        self._fit_weights(z_flat, y_flat, num_classes)

    def _mixture_log_probs(self, z_flat: torch.Tensor, num_classes: int) -> torch.Tensor:
        temperature = max(self.temperature, _T_EPS)
        w = self.weights.to(z_flat.device, dtype=z_flat.dtype)
        p0 = F.softmax(z_flat / temperature, dim=1)
        p1 = F.softmax(z_flat, dim=1)
        p2 = torch.full_like(p0, 1.0 / float(num_classes))
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        return safe_log(p)

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        flat, shape = class_last_flatten(canonical)
        log_p = self._mixture_log_probs(flat, shape[1])
        return restore_class_first(log_p, shape)

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "init_temperature": self.init_temperature,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "max_iter": self.max_iter,
        }

    def _get_state(self) -> dict[str, Any]:
        return {
            "temperature": torch.tensor(float(self.temperature)),
            "weights": self.weights.detach().cpu(),
        }

    def _set_state(self, state: dict[str, Any]) -> None:
        value = state.get("temperature")
        if value is not None:
            self.temperature = float(torch.as_tensor(value).item())
        weights = state.get("weights")
        if weights is not None:
            self.weights = torch.as_tensor(weights, device=self.device).float()

    def __repr__(self) -> str:
        w = self.weights.tolist()
        return (
            f"EnsembleTemperatureScaling(temperature={self.temperature:.4f}, "
            f"weights=[{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}], device={self.device})"
        )
