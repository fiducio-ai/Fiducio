"""Numerical regression against synthetic outputs of the research checkout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import fiducio

REFERENCE = json.loads((Path(__file__).parent / "fixtures/research_reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: f"{c['method']}-{c['classes']}-{c['seed']}")
def test_research_numerical_reference(case, monkeypatch):
    c = case["classes"]
    method = case["method"]
    cal = getattr(fiducio, method)(
        device="cpu", lambda_reg=case["lambda_reg"], mu_reg=case["mu_reg"]
    )
    x = torch.tensor(case["inputs"])
    y = torch.tensor(case["targets"])
    params = [torch.tensor(p).requires_grad_() for p in case["params"]]
    if method in ("MS", "MSc"):
        # Research MS applies row vectors on the right; public MS stores W
        # in the column-vector convention. MSc already shares that convention.
        cal._weight = params[0].t() if method == "MS" else params[0]
        cal._bias = params[1]
        if method == "MSc":
            cal._row_sum = params[2]

        def evaluate():
            return cal._apply_flat(x), cal._regularization()
    else:
        cal._raw_b, cal._raw_mu = params

        def evaluate():
            reg = torch.stack([
                cal._ms_regularization(
                    *cal._logit_affine(c, k, cal._positive(params[0][k]), cal._positive(params[1][k])),
                    cal.lambda_reg, cal.mu_reg,
                ) for k in range(c)
            ]).mean()
            return cal._route(x.log_softmax(1), c), reg

    logits, reg = evaluate()
    loss = F.cross_entropy(logits, y) + reg
    gradients = torch.autograd.grad(loss, params)
    for actual, expected in [(logits, case["logits"]), (reg, case["regularization"]), (loss, case["loss"])]:
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=2e-6)
    for actual, expected in zip(gradients, case["gradients"], strict=True):
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-5, atol=2e-6)
    optimizer = torch.optim.Adam(params, lr=0.01)
    for p, grad in zip(params, gradients, strict=True):
        p.grad = grad
    optimizer.step()
    logits, reg = evaluate()
    torch.testing.assert_close(logits, torch.tensor(case["updated_logits"]), rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(reg, torch.tensor(case["updated_regularization"]), rtol=1e-5, atol=2e-6)

    # Exercise the production fit objective and optimizer as well as the map.
    initial = [torch.tensor(p) for p in case["params"]]
    if method in ("MS", "MSc"):
        cal._weight = initial[0].t().contiguous() if method == "MS" else initial[0]
        cal._bias = initial[1]
        if method == "MSc":
            cal._row_sum = initial[2]
    else:
        cal._raw_b, cal._raw_mu = initial
    monkeypatch.setattr(cal, "_init_params", lambda _: None)
    cal.optimizer, cal.lr, cal.max_iter = "adam", 0.01, 1
    canonical = x if method in ("MS", "MSc") else x.log_softmax(1)
    cal._fit_core(canonical, y, c)
    actual = cal._map_logits(canonical)
    torch.testing.assert_close(actual, torch.tensor(case["updated_logits"]), rtol=1e-5, atol=2e-6)


def test_msc_identity_initialization_has_zero_odir_penalty():
    cal = fiducio.MSc(device="cpu", lambda_reg=1, mu_reg=1)
    cal._init_params(4)
    torch.testing.assert_close(cal._effective_weight(), torch.eye(4))
    assert cal._regularization().item() == 0


def test_legacy_msc_checkpoint_keeps_predictions(tmp_path):
    g = torch.Generator().manual_seed(72)
    raw = torch.randn(4, 4, generator=g)
    bias = torch.randn(4, generator=g)
    x = torch.randn(7, 4, generator=g)
    cal = fiducio.MSc(device="cpu")
    cal._set_state({"weight": raw, "bias": bias})
    cal._num_classes = 4
    cal._fitted = True
    expected = (x @ (raw - raw.mean(1, keepdim=True)).t() + bias).softmax(1)
    path = tmp_path / "legacy.pt"
    cal.save(path)
    loaded = fiducio.load_calibrator(path)
    torch.testing.assert_close(loaded.transform(x), expected)
    # Re-fitting intentionally uses the corrected parameterization.
    loaded.max_iter = 1
    loaded.fit(x, x.argmax(1))
    assert loaded._row_sum is not None
