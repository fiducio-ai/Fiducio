"""Synthetic voxels with a known oracle.

Labels are drawn from ``softmax(z)``; the model sees ``a_i * z + bias`` where
``a_i = a * exp(jitter * N(0, 1))`` varies per case, so voxels of one case share
their miscalibration. With ``jitter=0`` and no bias the optimal temperature is
exactly ``a``: that is the justness test of the temperature scalers.
"""

from __future__ import annotations

import torch

BIAS = torch.tensor([0.0, 0.5, -0.5, 1.0])
RARE_SHIFT = {2: 6.5, 4: 4.0}  # lowers the last class logit: it then covers about 1% of voxels

SCENARIOS = {
    "over": dict(a=2.5),
    "under": dict(a=0.5),
    "bias": dict(a=2.5, bias=True),
    "rare": dict(a=2.5, rare=True),
}


def simulate(scenario, C, n_cases, vox, seed, jitter=0.15):
    """Return ``(logits (N, C), labels (N,), oracle logits (N, C))`` with ``N = n_cases * vox``."""
    cfg = SCENARIOS[scenario]
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_cases, vox, C, generator=g) * 1.5
    if cfg.get("rare"):
        z[..., -1] -= RARE_SHIFT[C]
    y = torch.multinomial(z.reshape(-1, C).softmax(1), 1, generator=g).squeeze(1)
    a = cfg["a"] * torch.exp(jitter * torch.randn(n_cases, 1, 1, generator=g))
    s = a * z
    if cfg.get("bias"):
        s = s + BIAS[:C]
    return s.reshape(-1, C), y, z.reshape(-1, C)
