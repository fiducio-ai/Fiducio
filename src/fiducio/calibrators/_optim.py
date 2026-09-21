"""Shared optimizer handling for calibrators.

Calibrators expose an ``optimizer`` argument that selects between ``"adam"``
(the default, matching the reference implementation) and ``"lbfgs"`` (a fast,
deterministic choice for the convex scaling objectives). ``lr`` and ``max_iter``
default to sensible per-optimizer values when left as ``None``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..utils.stopping import StoppingRule
from ..utils.tensors import positive_finite

VALID_OPTIMIZERS = ("adam", "lbfgs")


def resolve_optimizer(
    optimizer: str,
    lr: float | None,
    max_iter: int | None,
    *,
    adam_lr: float,
    lbfgs_lr: float,
    adam_max_iter: int,
    lbfgs_max_iter: int,
) -> tuple[str, float, int]:
    """Validate ``optimizer`` and fill ``lr`` / ``max_iter`` defaults.

    Returns the lowercased optimizer name and the resolved ``lr`` and
    ``max_iter`` so that they can be stored and reproduced on reload.

    Raises
    ------
    ValueError
        If ``optimizer`` is unknown or the resolved ``max_iter`` is not
        strictly positive.
    """
    name = str(optimizer).lower()
    if name not in VALID_OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {VALID_OPTIMIZERS}, got {optimizer!r}")
    resolved_lr = (adam_lr if name == "adam" else lbfgs_lr) if lr is None else float(lr)
    resolved_iter = (
        (adam_max_iter if name == "adam" else lbfgs_max_iter)
        if max_iter is None
        else int(max_iter)
    )
    positive_finite(resolved_lr, "lr")
    if max_iter is not None and (isinstance(max_iter, bool) or resolved_iter != max_iter):
        raise ValueError("max_iter must be an integer")
    if resolved_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {resolved_iter}")
    return name, resolved_lr, resolved_iter


def minimize(
    optimizer: str,
    params: list[torch.Tensor],
    loss_fn: Callable[[], torch.Tensor],
    *,
    lr: float,
    max_iter: int,
    val_fn: Callable[[], torch.Tensor | float] | None = None,
    stopping: StoppingRule | None = None,
) -> None:
    """Minimize ``loss_fn`` over ``params`` in place.

    ``loss_fn`` must return a scalar loss and must **not** call ``backward``
    (this helper handles gradients). For Adam the best iterate is restored at
    the end so a noisy final step never degrades the result: the iterate with
    the lowest training loss by default, or, when early stopping is active, the
    one with the lowest validation loss.

    Early stopping is active when both ``stopping`` and ``val_fn`` are given
    (Adam only). ``val_fn`` returns the validation loss for the current
    parameters and is evaluated without gradients after every step. Optimization
    ends once it has not improved by more than ``stopping.min_delta`` for
    ``stopping.patience`` consecutive iterations, and the learning rate is
    decayed on plateaus when ``stopping.lr_patience`` is set.
    """
    def check_params(*, gradients: bool = False) -> None:
        for p in params:
            value = p.grad if gradients else p
            if value is not None and not torch.isfinite(value).all():
                raise ValueError("optimization produced non-finite parameters or gradients")

    def checked_loss() -> torch.Tensor:
        loss = loss_fn()
        if not torch.isfinite(loss):
            raise ValueError("optimization produced a non-finite loss")
        return loss

    check_params()
    if optimizer == "lbfgs":
        lbfgs = torch.optim.LBFGS(
            params, lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = checked_loss()
            loss.backward()
            check_params(gradients=True)
            return loss

        lbfgs.step(closure)
        check_params()
        with torch.no_grad():
            checked_loss()
        return

    rule = stopping if val_fn is not None else None
    adam = torch.optim.Adam(params, lr=lr)
    scheduler = None
    if rule is not None and rule.lr_patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            adam, mode="min", factor=rule.lr_factor, patience=rule.lr_patience
        )
    def monitor() -> float:
        with torch.no_grad():
            value = float(val_fn() if rule is not None and val_fn is not None else checked_loss())
        if not math.isfinite(value):
            raise ValueError("optimization produced a non-finite monitored loss")
        return value

    best_loss = monitor()
    patience_best = best_loss
    best_state = [p.detach().clone() for p in params]
    stale = 0
    for _ in range(max_iter):
        adam.zero_grad()
        loss = checked_loss()
        loss.backward()
        check_params(gradients=True)
        adam.step()
        check_params()
        value = monitor()
        if value < best_loss:
            best_loss = value
            best_state = [p.detach().clone() for p in params]
        if rule is not None:
            if value < patience_best - rule.min_delta:
                patience_best = value
                stale = 0
            else:
                stale += 1
            if scheduler is not None:
                scheduler.step(value)
            if rule.patience is not None and stale >= rule.patience:
                break
    with torch.no_grad():
        for p, best in zip(params, best_state, strict=True):
            p.copy_(best)
