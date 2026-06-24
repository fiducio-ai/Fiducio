"""Temperature scaling calibrator."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..base import Calibrator, DeviceLike
from ..registry import register_calibrator

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
    max_iter:
        Maximum L-BFGS iterations.
    lr:
        L-BFGS learning rate.
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
        max_iter: int = 100,
        lr: float = 0.1,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(input_type=input_type, ignore_index=ignore_index, device=device)
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive")
        self.init_temperature = float(init_temperature)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.temperature: float = float(init_temperature)

    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        init = max(self.init_temperature, _T_EPS)
        raw = torch.log(torch.expm1(torch.tensor(init, device=self.device)))
        raw_t = raw.clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [raw_t], lr=self.lr, max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = F.softplus(raw_t) + _T_EPS
            loss = F.cross_entropy(z_flat / temperature, y_flat)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float((F.softplus(raw_t) + _T_EPS).detach().cpu().item())

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        temperature = max(self.temperature, _T_EPS)
        return canonical / temperature

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "init_temperature": self.init_temperature,
            "max_iter": self.max_iter,
            "lr": self.lr,
        }

    def _get_state(self) -> dict[str, Any]:
        return {"temperature": torch.tensor(float(self.temperature))}

    def _set_state(self, state: dict[str, Any]) -> None:
        value = state.get("temperature")
        if value is not None:
            self.temperature = float(torch.as_tensor(value).item())

    def __repr__(self) -> str:
        return f"TemperatureScaling(temperature={self.temperature:.4f}, device={self.device})"
