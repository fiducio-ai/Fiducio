"""Class-conditional matrix scaling: CMS, argmax- and order-preserving variants.

These calibrators route each voxel to an expert affine map selected by the
uncalibrated top class, and operate in log-probability space. By default
(``independent_experts=False``, matching the paper) all experts are optimized
**jointly**: a single optimizer minimizes one cross-entropy loss over every
voxel at once (each voxel's contribution passing through its own expert's
affine map), plus a regularization term that is the *mean*, over all experts,
of the penalty on the affine map each expert induces in the common logit space
(matrix-scaling style off-diagonal / bias penalties). Setting
``independent_experts=True`` instead fits each expert in its own optimization
loop on only the voxels routed to it, which avoids experts with few routed
voxels being dominated by the joint loss, at the cost of ``C`` times the
optimizer work.

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
from ._optim import minimize, resolve_optimizer


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
        optimizer: str = "adam",
        lr: float | None = None,
        max_iter: int | None = None,
        lambda_reg: float = 0.0,
        mu_reg: float = 0.0,
        independent_experts: bool = False,
        init_alpha: float = 1.0,
        init_floor: float = 1e-6,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(input_type=input_type, ignore_index=ignore_index, device=device)
        self.optimizer, self.lr, self.max_iter = resolve_optimizer(
            optimizer, lr, max_iter,
            adam_lr=1e-2, lbfgs_lr=1.0, adam_max_iter=200, lbfgs_max_iter=100,
        )
        self.lambda_reg = float(lambda_reg)
        self.mu_reg = float(mu_reg)
        self.independent_experts = bool(independent_experts)
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
        if self.independent_experts:
            self._fit_independent(z_flat, y_flat, num_classes)
        else:
            self._fit_joint(z_flat, y_flat, num_classes)

    def _fit_joint(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        """Optimize all experts together in a single loss (the paper's method)."""
        assert self._raw_b is not None and self._raw_mu is not None
        raw_b = self._raw_b.clone().requires_grad_(True)
        raw_mu = self._raw_mu.clone().requires_grad_(True)

        def loss_fn(raw_b: torch.Tensor = raw_b, raw_mu: torch.Tensor = raw_mu) -> torch.Tensor:
            logits = self._route(z_flat, num_classes, raw_b=raw_b, raw_mu=raw_mu)
            data_loss = F.cross_entropy(logits, y_flat)
            reg_terms = []
            for c in range(num_classes):
                bp = self._positive(raw_b[c])
                mup = self._positive(raw_mu[c])
                affine_a, affine_b = self._logit_affine(num_classes, c, bp, mup)
                reg_terms.append(
                    self._ms_regularization(affine_a, affine_b, self.lambda_reg, self.mu_reg)
                )
            return data_loss + torch.stack(reg_terms).mean()

        minimize(self.optimizer, [raw_b, raw_mu], loss_fn, lr=self.lr, max_iter=self.max_iter)
        with torch.no_grad():
            self._raw_b = raw_b.detach()
            self._raw_mu = raw_mu.detach()

    def _fit_independent(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        """Optimize each expert in its own loop, on only the voxels routed to it."""
        assert self._raw_b is not None and self._raw_mu is not None
        top = torch.argmax(z_flat, dim=1)
        for c in range(num_classes):
            sel = top == c
            if not bool(sel.any()):
                continue
            rows = z_flat[sel]
            targets = y_flat[sel]
            b_c = self._raw_b[c].clone().requires_grad_(True)
            mu_c = self._raw_mu[c].clone().requires_grad_(True)

            def loss_fn(
                b_c: torch.Tensor = b_c,
                mu_c: torch.Tensor = mu_c,
                c: int = c,
                rows: torch.Tensor = rows,
                targets: torch.Tensor = targets,
            ) -> torch.Tensor:
                bp = self._positive(b_c)
                mup = self._positive(mu_c)
                logits = self._expert_logits(rows, c, bp, mup)
                affine_a, affine_b = self._logit_affine(num_classes, c, bp, mup)
                return F.cross_entropy(logits, targets) + self._ms_regularization(
                    affine_a, affine_b, self.lambda_reg, self.mu_reg
                )

            minimize(self.optimizer, [b_c, mu_c], loss_fn, lr=self.lr, max_iter=self.max_iter)
            with torch.no_grad():
                self._raw_b[c] = b_c.detach()
                self._raw_mu[c] = mu_c.detach()

    def _route(
        self,
        logp_flat: torch.Tensor,
        num_classes: int,
        *,
        raw_b: torch.Tensor | None = None,
        raw_mu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route each row to its top-class expert; ``raw_b``/``raw_mu`` default to fitted state."""
        raw_b = self._raw_b if raw_b is None else raw_b
        raw_mu = self._raw_mu if raw_mu is None else raw_mu
        assert raw_b is not None and raw_mu is not None
        top = torch.argmax(logp_flat, dim=1)
        out = torch.empty_like(logp_flat)
        for c in range(num_classes):
            idx = torch.nonzero(top == c, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            rows = logp_flat.index_select(0, idx)
            bp = self._positive(raw_b[c])
            mup = self._positive(raw_mu[c])
            out.index_copy_(0, idx, self._expert_logits(rows, c, bp, mup))
        return out

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        flat, shape = class_last_flatten(canonical)
        out = self._route(flat, shape[1])
        return restore_class_first(out, shape)

    # ------------------------------------------------------------- persistence

    def _constructor_config(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "lr": self.lr,
            "max_iter": self.max_iter,
            "lambda_reg": self.lambda_reg,
            "mu_reg": self.mu_reg,
            "independent_experts": self.independent_experts,
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
    matrix close to the identity. Experts are optimized jointly by default; set
    ``independent_experts=True`` to fit each one in its own optimization loop.
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
    argmax for every voxel. Experts are optimized jointly by default; set
    ``independent_experts=True`` to fit each one in its own optimization loop.
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
    voxel. Experts are optimized jointly by default; set
    ``independent_experts=True`` to fit each one in its own optimization loop.
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
