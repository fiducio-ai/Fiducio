"""Class-conditional matrix scaling: CMS, argmax- and order-preserving variants.

These calibrators route each voxel to an expert affine map selected by the
uncalibrated top class, and operate in log-probability space. Each expert is
optimized **independently** on the voxels routed to it, and regularization is
applied to the affine map induced in the common logit space (matrix-scaling
style off-diagonal / bias penalties).

* :class:`ClassConditionalMatrixScaling` (CMS) – an unconstrained affine map per
  expert.
* :class:`ArgmaxPreservingMatrixScaling` (CMSAP) – guarantees the calibrated
  argmax equals the uncalibrated argmax.
* :class:`OrderPreservingMatrixScaling` (CMSOP) – guarantees the full class
  ranking is preserved.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..base import Calibrator, DeviceLike
from ..registry import register_calibrator
from ..utils import class_last_flatten, restore_class_first


def _inv_softplus(value: float, device: torch.device) -> torch.Tensor:
    v = torch.tensor(float(value), dtype=torch.float32, device=device)
    return torch.log(torch.expm1(v.clamp_min(1e-12)))


class _ClassConditionalBase(Calibrator):
    """Shared implementation for class-conditional matrix scaling calibrators."""

    _input_space = "logprobs"
    #: whether expert parameters are mapped through softplus to be positive.
    _positive_params: bool = False

    def __init__(
        self,
        *,
        max_iter: int = 200,
        lr: float = 1e-2,
        lambda_reg: float = 0.0,
        mu_reg: float = 0.0,
        min_expert_voxels: int = 1,
        init_alpha: float = 1.0,
        init_floor: float = 1e-6,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(input_type=input_type, ignore_index=ignore_index, device=device)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.lambda_reg = float(lambda_reg)
        self.mu_reg = float(mu_reg)
        self.min_expert_voxels = max(1, int(min_expert_voxels))
        self.init_alpha = float(init_alpha)
        self.init_floor = float(init_floor)
        self._raw_b: torch.Tensor | None = None  # (C, K, K) or (C, C, C)
        self._raw_mu: torch.Tensor | None = None  # (C, K) or (C, C)

    # ------------------------------------------------------- parameter helpers

    def _matrix_dim(self, num_classes: int) -> int:
        """Per-expert square matrix size (``C-1`` for AP/OP, ``C`` for CMS)."""
        raise NotImplementedError

    def _init_params(self, num_classes: int) -> None:
        c = int(num_classes)
        k = self._matrix_dim(c)
        if self._positive_params:
            raw_floor = _inv_softplus(self.init_floor, self.device).item()
            raw_alpha = _inv_softplus(max(self.init_alpha, self.init_floor), self.device).item()
            raw_mu = torch.full((c, k), raw_floor, device=self.device)
            raw_b = torch.full((c, k, k), raw_floor, device=self.device)
            eye = torch.eye(k, dtype=torch.bool, device=self.device)
            raw_b[:, eye] = raw_alpha
        else:
            raw_mu = torch.zeros((c, k), device=self.device)
            raw_b = torch.eye(k, device=self.device).unsqueeze(0).repeat(c, 1, 1)
        self._raw_b = raw_b
        self._raw_mu = raw_mu

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(raw) if self._positive_params else raw

    @staticmethod
    def _competitor_indices(num_classes: int, device: torch.device) -> torch.Tensor:
        base = torch.arange(num_classes, device=device)
        return torch.stack([base[base != c] for c in range(num_classes)], dim=0)

    # ------------------------------------------------------ per-expert forward

    def _expert_logits(
        self, logp_rows: torch.Tensor, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> torch.Tensor:
        """Map log-probabilities routed to ``expert`` to calibrated logits."""
        raise NotImplementedError

    def _logit_affine(
        self, num_classes: int, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Affine map ``(A, b)`` induced in the common logit space."""
        raise NotImplementedError

    @staticmethod
    def _ms_regularization(
        matrix: torch.Tensor, bias: torch.Tensor, lambda_reg: float, mu_reg: float
    ) -> torch.Tensor:
        reg = matrix.new_zeros(())
        c = int(matrix.shape[-1])
        if lambda_reg > 0 and c > 1:
            off = ~torch.eye(c, dtype=torch.bool, device=matrix.device)
            reg = reg + lambda_reg * matrix[off].square().mean()
        if mu_reg > 0:
            reg = reg + mu_reg * bias.square().mean()
        return reg

    # --------------------------------------------------------------- fit / map

    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        self._init_params(num_classes)
        assert self._raw_b is not None and self._raw_mu is not None
        top = torch.argmax(z_flat, dim=1)
        for c in range(num_classes):
            sel = top == c
            count = int(sel.sum().item())
            if count < self.min_expert_voxels:
                continue
            rows = z_flat[sel]
            targets = y_flat[sel]
            b_c = self._raw_b[c].clone().requires_grad_(True)
            mu_c = self._raw_mu[c].clone().requires_grad_(True)
            optimizer = torch.optim.Adam([b_c, mu_c], lr=self.lr)
            best_loss = float("inf")
            best_b = b_c.detach().clone()
            best_mu = mu_c.detach().clone()
            for _ in range(self.max_iter):
                optimizer.zero_grad()
                bp = self._positive(b_c)
                mup = self._positive(mu_c)
                logits = self._expert_logits(rows, c, bp, mup)
                affine_a, affine_b = self._logit_affine(num_classes, c, bp, mup)
                loss = F.cross_entropy(logits, targets) + self._ms_regularization(
                    affine_a, affine_b, self.lambda_reg, self.mu_reg
                )
                loss.backward()
                optimizer.step()
                value = float(loss.detach().item())
                if value + 1e-9 < best_loss:
                    best_loss = value
                    best_b = b_c.detach().clone()
                    best_mu = mu_c.detach().clone()
            with torch.no_grad():
                self._raw_b[c] = best_b
                self._raw_mu[c] = best_mu

    def _route(self, logp_flat: torch.Tensor, num_classes: int) -> torch.Tensor:
        assert self._raw_b is not None and self._raw_mu is not None
        top = torch.argmax(logp_flat, dim=1)
        out = torch.empty_like(logp_flat)
        for c in range(num_classes):
            idx = torch.nonzero(top == c, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            rows = logp_flat.index_select(0, idx)
            bp = self._positive(self._raw_b[c])
            mup = self._positive(self._raw_mu[c])
            out.index_copy_(0, idx, self._expert_logits(rows, c, bp, mup))
        return out

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        flat, shape = class_last_flatten(canonical)
        out = self._route(flat, shape[1])
        return restore_class_first(out, shape)

    # ------------------------------------------------------------- persistence

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "lr": self.lr,
            "lambda_reg": self.lambda_reg,
            "mu_reg": self.mu_reg,
            "min_expert_voxels": self.min_expert_voxels,
            "init_alpha": self.init_alpha,
            "init_floor": self.init_floor,
        }

    def _get_state(self) -> dict[str, Any]:
        return {"raw_b": self._raw_b, "raw_mu": self._raw_mu}

    def _set_state(self, state: dict[str, Any]) -> None:
        raw_b = state.get("raw_b")
        raw_mu = state.get("raw_mu")
        self._raw_b = None if raw_b is None else torch.as_tensor(raw_b, device=self.device)
        self._raw_mu = None if raw_mu is None else torch.as_tensor(raw_mu, device=self.device)


@register_calibrator("class_conditional_matrix_scaling")
class ClassConditionalMatrixScaling(_ClassConditionalBase):
    """Class-conditional matrix scaling (CMS).

    One unconstrained affine map ``A_c log p + b_c`` per uncalibrated top class
    ``c``. Unlike the preserving variants it may change the argmax. Off-diagonal
    and bias L2 regularization (``lambda_reg`` / ``mu_reg``) keep each expert
    matrix close to the identity.
    """

    _positive_params = False

    def _matrix_dim(self, num_classes: int) -> int:
        return int(num_classes)

    def _expert_logits(
        self, logp_rows: torch.Tensor, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> torch.Tensor:
        return logp_rows.matmul(b_c) + mu_c

    def _logit_affine(
        self, num_classes: int, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return b_c, mu_c


@register_calibrator("argmax_preserving_matrix_scaling")
class ArgmaxPreservingMatrixScaling(_ClassConditionalBase):
    """Argmax-preserving class-conditional matrix scaling (CMSAP).

    Parameterised through non-negative margins between the top class and its
    competitors, which guarantees the calibrated argmax equals the uncalibrated
    argmax for every voxel.
    """

    _positive_params = True

    def _matrix_dim(self, num_classes: int) -> int:
        return int(num_classes) - 1

    def _expert_logits(
        self, logp_rows: torch.Tensor, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> torch.Tensor:
        c = int(logp_rows.shape[1])
        comp = self._competitor_indices(c, logp_rows.device)[expert]
        margins = (
            logp_rows[:, expert : expert + 1] - logp_rows.index_select(1, comp)
        ).clamp_min(0.0)
        tilde = margins.matmul(b_c.t()) + mu_c
        out = torch.zeros_like(logp_rows)
        out[:, comp] = -tilde
        return out

    def _logit_affine(
        self, num_classes: int, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = int(num_classes)
        dtype, device = b_c.dtype, b_c.device
        comp = self._competitor_indices(c, device)[expert]
        rows = torch.arange(c - 1, device=device)
        d_mat = torch.zeros((c - 1, c), dtype=dtype, device=device)
        d_mat[rows, expert] = 1.0
        d_mat[rows, comp] = -1.0
        s_mat = torch.zeros((c, c - 1), dtype=dtype, device=device)
        s_mat[comp, rows] = 1.0
        winner = F.one_hot(torch.tensor(expert, device=device), num_classes=c).to(dtype)
        a_col = torch.ones((c, 1), dtype=dtype, device=device).matmul(
            winner.unsqueeze(0)
        ) - s_mat.matmul(b_c).matmul(d_mat)
        bias = -s_mat.matmul(mu_c)
        return a_col.t(), bias


@register_calibrator("order_preserving_matrix_scaling")
class OrderPreservingMatrixScaling(_ClassConditionalBase):
    """Order-preserving class-conditional matrix scaling (CMSOP).

    Parameterised through non-negative gaps between consecutively ranked
    classes, which guarantees the full class ranking is preserved for every
    voxel.
    """

    _positive_params = True

    def _matrix_dim(self, num_classes: int) -> int:
        return int(num_classes) - 1

    def _expert_logits(
        self, logp_rows: torch.Tensor, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> torch.Tensor:
        sorted_logp, perm = torch.sort(logp_rows, dim=1, descending=True)
        margins = (sorted_logp[:, :-1] - sorted_logp[:, 1:]).clamp_min(0.0)
        tilde = margins.matmul(b_c.t()) + mu_c
        h_sorted = torch.zeros_like(logp_rows)
        h_sorted[:, 1:] = -torch.cumsum(tilde, dim=1)
        out = torch.zeros_like(h_sorted)
        out.scatter_(1, perm, h_sorted)
        return out

    def _logit_affine(
        self, num_classes: int, expert: int, b_c: torch.Tensor, mu_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = int(num_classes)
        dtype, device = b_c.dtype, b_c.device
        rows = torch.arange(c - 1, device=device)
        d_mat = torch.zeros((c - 1, c), dtype=dtype, device=device)
        d_mat[rows, rows] = 1.0
        d_mat[rows, rows + 1] = -1.0
        l_mat = torch.zeros((c, c - 1), dtype=dtype, device=device)
        for j in range(1, c):
            l_mat[j, :j] = 1.0
        winner = F.one_hot(torch.tensor(0, device=device), num_classes=c).to(dtype)
        a_mat = torch.ones((c, 1), dtype=dtype, device=device).matmul(
            winner.unsqueeze(0)
        ) - l_mat.matmul(b_c).matmul(d_mat)
        bias = -l_mat.matmul(mu_c)
        return a_mat.t(), bias
