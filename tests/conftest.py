"""Shared fixtures and helpers for the Fiducio test suite."""

from __future__ import annotations

import torch
import torch.nn.functional as F

ALL_CALIBRATOR_IDS = [
    "temperature_scaling",
    "ensemble_temperature_scaling",
    "vector_scaling",
    "matrix_scaling",
    "translation_invariant_matrix_scaling",
    "dirichlet_calibration",
    "class_conditional_matrix_scaling",
    "argmax_preserving_matrix_scaling",
    "order_preserving_matrix_scaling",
]


def make_calibrator(calibrator_id: str, **kwargs):
    """Construct a calibrator by id with small, fast defaults on CPU."""
    from fiducio.registry import get_calibrator_class

    cls = get_calibrator_class(calibrator_id)
    params = {"device": "cpu"}
    # Keep iterative fits short so the suite runs fast on CPU.
    if calibrator_id in {
        "class_conditional_matrix_scaling",
        "argmax_preserving_matrix_scaling",
        "order_preserving_matrix_scaling",
    }:
        params["max_iter"] = 60
    params.update(kwargs)
    return cls(**params)


def synthetic_logits(shape, *, seed=0, scale=3.0):
    """Return overconfident logits and matching argmax labels.

    ``shape`` is ``(B, C, *spatial)``. Labels are ``(B, *spatial)``.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(*shape, generator=g)
    labels = base.argmax(dim=1)
    logits = base * scale + torch.randn(*shape, generator=g) * 0.5
    return logits, labels


def to_probs(logits):
    return F.softmax(logits, dim=1)
