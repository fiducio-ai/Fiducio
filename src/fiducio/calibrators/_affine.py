"""Shared affine-map calibrator base (vector / matrix / Dirichlet scaling)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..base import Calibrator, DeviceLike
from ..utils import class_last_flatten, restore_class_first
from ._optim import minimize, resolve_optimizer


class _AffineCalibrator(Calibrator):
    """Affine calibration ``softmax(W z + b)`` on a canonical representation.

    Subclasses pick the parameterisation through ``_mode``:

    * ``"diagonal"`` – ``W = diag(w)`` (vector scaling);
    * ``"matrix"`` – full ``W`` (matrix scaling / Dirichlet calibration);
    * ``"matrix_ti"`` – full ``W`` constrained so each row sums to zero, which
      makes the map invariant to adding a constant to all input logits.

    ``z`` is the canonical representation produced by the base class (logits for
    scaling methods, log-probabilities for Dirichlet calibration).
    """

    _mode: str = "matrix"

    def __init__(
        self,
        *,
        optimizer: str = "adam",
        lr: float | None = None,
        max_iter: int | None = None,
        lambda_reg: float = 0.0,
        mu_reg: float = 0.0,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(input_type=input_type, ignore_index=ignore_index, device=device)
        self.optimizer, self.lr, self.max_iter = resolve_optimizer(
            optimizer, lr, max_iter,
            adam_lr=0.1, lbfgs_lr=1.0, adam_max_iter=200, lbfgs_max_iter=100,
        )
        self.lambda_reg = float(lambda_reg)
        self.mu_reg = float(mu_reg)
        self._weight: torch.Tensor | None = None  # (C, C) or (C,) for diagonal
        self._bias: torch.Tensor | None = None  # (C,)

    # --------------------------------------------------------------- internals

    def _init_params(self, num_classes: int) -> None:
        c = int(num_classes)
        if self._mode == "diagonal":
            self._weight = torch.ones(c, device=self.device)
        else:
            self._weight = torch.eye(c, device=self.device)
        self._bias = torch.zeros(c, device=self.device)

    def _effective_weight(self) -> torch.Tensor:
        assert self._weight is not None
        if self._mode == "matrix_ti":
            return self._weight - self._weight.mean(dim=1, keepdim=True)
        return self._weight

    def _apply_flat(self, z_flat: torch.Tensor) -> torch.Tensor:
        assert self._bias is not None
        if self._mode == "diagonal":
            assert self._weight is not None
            return z_flat * self._weight + self._bias
        weight = self._effective_weight()
        return z_flat.matmul(weight.t()) + self._bias

    def _regularization(self) -> torch.Tensor:
        assert self._weight is not None and self._bias is not None
        reg = self._weight.new_zeros(())
        if self._mode == "diagonal":
            if self.lambda_reg > 0:
                reg = reg + self.lambda_reg * (self._weight - 1.0).square().mean()
        else:
            weight = self._effective_weight()
            c = weight.shape[-1]
            if self.lambda_reg > 0 and c > 1:
                off = ~torch.eye(c, dtype=torch.bool, device=weight.device)
                reg = reg + self.lambda_reg * weight[off].square().mean()
        if self.mu_reg > 0:
            reg = reg + self.mu_reg * self._bias.square().mean()
        return reg

    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        self._init_params(num_classes)
        assert self._weight is not None and self._bias is not None
        params = [self._weight.requires_grad_(True), self._bias.requires_grad_(True)]

        def loss_fn() -> torch.Tensor:
            logits = self._apply_flat(z_flat)
            return F.cross_entropy(logits, y_flat) + self._regularization()

        minimize(self.optimizer, params, loss_fn, lr=self.lr, max_iter=self.max_iter)
        self._weight = self._weight.detach()
        self._bias = self._bias.detach()

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        flat, shape = class_last_flatten(canonical)
        out = self._apply_flat(flat)
        return restore_class_first(out, shape)

    # ------------------------------------------------------------- persistence

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "lambda_reg": self.lambda_reg,
            "mu_reg": self.mu_reg,
        }

    def _get_state(self) -> dict[str, Any]:
        return {"weight": self._weight, "bias": self._bias}

    def _set_state(self, state: dict[str, Any]) -> None:
        weight = state.get("weight")
        bias = state.get("bias")
        self._weight = None if weight is None else torch.as_tensor(weight, device=self.device)
        self._bias = None if bias is None else torch.as_tensor(bias, device=self.device)
