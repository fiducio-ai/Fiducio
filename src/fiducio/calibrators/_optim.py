"""Shared optimizer handling for calibrators.

Calibrators expose an ``optimizer`` argument that selects between ``"adam"``
(the default, matching the reference implementation) and ``"lbfgs"`` (a fast,
deterministic choice for the convex scaling objectives). ``lr`` and ``max_iter``
default to sensible per-optimizer values when left as ``None``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

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
) -> None:
    """Minimize ``loss_fn`` over ``params`` in place.

    ``loss_fn`` must return a scalar loss and must **not** call ``backward``
    (this helper handles gradients). For Adam the best-loss iterate is restored
    at the end so a noisy final step never degrades the result.
    """
    if optimizer == "lbfgs":
        lbfgs = torch.optim.LBFGS(
            params, lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        lbfgs.step(closure)
        return

    adam = torch.optim.Adam(params, lr=lr)
    best_loss = float("inf")
    best_state = [p.detach().clone() for p in params]
    for _ in range(max_iter):
        adam.zero_grad()
        loss = loss_fn()
        loss.backward()
        adam.step()
        value = float(loss.detach().item())
        if value + 1e-9 < best_loss:
            best_loss = value
            best_state = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, best in zip(params, best_state, strict=True):
            p.copy_(best)
