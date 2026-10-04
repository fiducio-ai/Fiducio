"""Ensemble Temperature Scaling (ETS)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..base import Calibrator, DeviceLike
from ..registry import register_calibrator
from ..utils import class_last_flatten, restore_class_first, safe_log
from ..utils.tensors import inverse_softplus, positive_finite
from ._optim import minimize, resolve_optimizer

_EPS = 1e-12
_T_EPS = 1e-6
_W_INIT = 1e-3


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
    Adapted from Zhang et al. (2020), *Mix-n-Match: Ensemble and Compositional
    Methods for Uncertainty Calibration in Deep Learning*, ICML. Unlike the cited
    method, fitting here is sequential: stage 1 minimizes the NLL over ``T`` with
    Adam, stage 2 minimizes the cross-entropy over ``w`` with ``T`` fixed; the
    paper fits all parameters jointly.
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
        self.init_temperature = positive_finite(init_temperature, "init_temperature")
        self.optimizer, self.lr, self.max_iter = resolve_optimizer(
            optimizer,
            lr,
            max_iter,
            adam_lr=0.1,
            lbfgs_lr=1.0,
            adam_max_iter=200,
            lbfgs_max_iter=100,
        )
        self._init_stopping(self.optimizer, patience, min_delta, lr_patience, lr_factor)
        self.temperature: float = float(init_temperature)
        self.weights: torch.Tensor = torch.tensor([1.0, 0.0, 0.0], device=self.device)

    def _fit_temperature(self, z_flat: torch.Tensor, y_flat: torch.Tensor) -> None:
        init = max(self.init_temperature, _T_EPS)
        raw = inverse_softplus(init, self.device)
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
            self.optimizer,
            [raw_t],
            loss_fn,
            lr=self.lr,
            max_iter=self.max_iter,
            val_fn=val_fn,
            stopping=self._stopping,
        )
        self.temperature = float((F.softplus(raw_t) + _T_EPS).detach().cpu().item())

    def _fit_weights(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        temperature = max(self.temperature, _T_EPS)
        uniform = 1.0 / float(num_classes)

        def true_class_probs(z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # The NLL only needs each component's probability of the true class.
            # float64 keeps the small stage-2 loss changes above rounding noise, so
            # L-BFGS line searches do not stall on some platforms.
            index = y.unsqueeze(1)
            scaled = F.softmax(z / temperature, dim=1).gather(1, index).squeeze(1)
            original = F.softmax(z, dim=1).gather(1, index).squeeze(1)
            return scaled.double(), original.double()

        q0, q1 = true_class_probs(z_flat.detach(), y_flat)
        # Start next to the temperature-scaled distribution. The softmax gradient of
        # each weight scales with the weight itself, so the other weights must not
        # start at ~0 or the optimizer can never move them.
        init_w = torch.tensor(
            [1.0 - 2 * _W_INIT, _W_INIT, _W_INIT], dtype=torch.float64, device=self.device
        )
        raw_w = torch.log(init_w).requires_grad_(True)

        def mixture_nll(t0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
            w = F.softmax(raw_w, dim=0)
            return -torch.log((w[0] * t0 + w[1] * t1 + w[2] * uniform).clamp_min(_EPS)).mean()

        def loss_fn() -> torch.Tensor:
            return mixture_nll(q0, q1)

        val_fn = None
        if self._val is not None:
            v0, v1 = true_class_probs(*self._val)

            def val_fn() -> torch.Tensor:
                return mixture_nll(v0, v1)

        minimize(
            self.optimizer,
            [raw_w],
            loss_fn,
            lr=self.lr,
            max_iter=self.max_iter,
            val_fn=val_fn,
            stopping=self._stopping,
        )
        self.weights = F.softmax(raw_w, dim=0).detach().float()

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
