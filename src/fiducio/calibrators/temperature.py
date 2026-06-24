"""Temperature scaling calibrator."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..base import Calibrator, DeviceLike
from ..registry import register_calibrator
from ._optim import minimize, resolve_optimizer

_T_EPS = 1e-6


@register_calibrator("temperature_scaling")
class TemperatureScaling(Calibrator):
    """Temperature scaling: ``softmax(z / T)`` with a single scalar ``T > 0``.

    The simplest and most robust post-hoc calibrator. It cannot change the
    predicted class ordering; it only rescales confidence.

    Parameters
    ----------
    init_temperature:
        Initial temperature (must be positive).
    optimizer:
        ``"adam"`` (default) or ``"lbfgs"``.
    lr:
        Learning rate. Defaults to ``0.1`` (Adam) or ``1.0`` (L-BFGS).
    max_iter:
        Maximum optimizer iterations. Defaults to ``200`` (Adam) or ``100``
        (L-BFGS).
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`.

    References
    ----------
    Guo et al. (2017), *On Calibration of Modern Neural Networks*, ICML.
    """

    _input_space = "logits"

    def __init__(
        self,
        *,
        init_temperature: float = 1.0,
        optimizer: str = "adam",
        lr: float | None = None,
        max_iter: int | None = None,
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
        self.temperature: float = float(init_temperature)

    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        init = max(self.init_temperature, _T_EPS)
        raw = torch.log(torch.expm1(torch.tensor(init, device=self.device)))
        raw_t = raw.clone().requires_grad_(True)

        def loss_fn() -> torch.Tensor:
            temperature = F.softplus(raw_t) + _T_EPS
            return F.cross_entropy(z_flat / temperature, y_flat)

        minimize(self.optimizer, [raw_t], loss_fn, lr=self.lr, max_iter=self.max_iter)
        self.temperature = float((F.softplus(raw_t) + _T_EPS).detach().cpu().item())

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        temperature = max(self.temperature, _T_EPS)
        return canonical / temperature

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "init_temperature": self.init_temperature,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "max_iter": self.max_iter,
        }

    def _get_state(self) -> dict[str, Any]:
        return {"temperature": torch.tensor(float(self.temperature))}

    def _set_state(self, state: dict[str, Any]) -> None:
        value = state.get("temperature")
        if value is not None:
            self.temperature = float(torch.as_tensor(value).item())

    def __repr__(self) -> str:
        return f"TemperatureScaling(temperature={self.temperature:.4f}, device={self.device})"
