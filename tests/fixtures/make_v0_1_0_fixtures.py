"""Regenerate the fiducio 0.1.0 persistence fixtures.

Run with the *published* 0.1.0 wheel, never with the source tree:

    uv venv venv && uv pip install --python venv "torch==2.10.0+cpu" numpy \
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
    uv pip install --python venv --no-deps fiducio==0.1.0
    venv/bin/python tests/fixtures/make_v0_1_0_fixtures.py tests/fixtures/v0_1_0

The files were produced this way with fiducio 0.1.0 and torch 2.10.0+cpu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import fiducio
from fiducio.registry import get_calibrator_class

IDS = [
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


def main(out: Path) -> None:
    if fiducio.__version__ != "0.1.0":
        raise SystemExit(f"expected fiducio 0.1.0, got {fiducio.__version__}")
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(2026)
    fit_z = torch.randn(4, 3, 5, 5, generator=g) * 3
    fit_y = fit_z.argmax(1)
    flip = torch.rand(fit_y.shape, generator=g) < 0.2
    fit_y[flip] = (fit_y[flip] + 1) % 3
    eval_z = torch.randn(2, 3, 5, 5, generator=g) * 3
    expected = {}
    for cid in IDS:
        cls = get_calibrator_class(cid)
        kwargs = {"device": "cpu", "max_iter": 20}
        if cid in {"matrix_scaling", "order_preserving_matrix_scaling"}:
            kwargs.update(patience=5)  # records stopping settings in the config
        cls(**kwargs).save(out / f"{cid}_unfitted.pt")
        cal = cls(**kwargs)
        if "patience" in kwargs:
            cal.fit(fit_z, fit_y, val_predictions=fit_z, val_targets=fit_y)
        else:
            cal.fit(fit_z, fit_y)
        cal.save(out / f"{cid}_fitted.pt")
        expected[cid] = cal.transform(eval_z)
    probs_cal = get_calibrator_class("dirichlet_calibration")(
        device="cpu", input_type="probs", max_iter=20
    ).fit(fit_z.softmax(1), fit_y)
    probs_cal.save(out / "dirichlet_calibration_probs_fitted.pt")
    expected["dirichlet_calibration_probs"] = probs_cal.transform(eval_z.softmax(1))
    torch.save(
        {"fiducio_version": fiducio.__version__, "torch_version": str(torch.__version__),
         "eval_logits": eval_z, "expected": expected},
        out / "reference.pt",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
