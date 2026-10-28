"""Rerun the simulation quoted in the ``fiducio.diagnose`` documentation.

Draws labels from the model's own probabilities, so each simulated model is calibrated
by construction, and counts how often (a) the 95% bootstrap interval of the ECE misses
the ECE estimate and (b) the bias line flags the model. Run from the repository root::

    python reproducibility/diagnose_interval_check.py
"""

from __future__ import annotations

import torch

import fiducio

N_MODELS = 100


def main() -> None:
    missed = flagged = 0
    for seed in range(N_MODELS):
        g = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, 3, (24, 16, 16), generator=g)
        onehot = torch.nn.functional.one_hot(labels, 3).permute(0, 3, 1, 2).float()
        logits = onehot * 1.5 + torch.randn(onehot.shape, generator=g)
        probs = torch.softmax(logits, 1).permute(0, 2, 3, 1)
        drawn = torch.multinomial(probs.reshape(-1, 3), 1, generator=g).view(labels.shape)
        report = fiducio.diagnose(logits, drawn, n_boot=200, min_class_voxels=10)
        value, lo, hi = report.metrics["ece"]
        missed += not lo <= value <= hi
        flagged += report.bias in ("overconfident", "underconfident")
    print(f"ECE estimate outside its interval: {missed}/{N_MODELS}")
    print(f"calibrated models flagged as biased: {flagged}/{N_MODELS} (5 expected)")


if __name__ == "__main__":
    main()
