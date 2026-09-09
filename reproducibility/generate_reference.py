"""Generate synthetic numerical references from an explicit research checkout.

No data, checkpoints or research training loops are loaded. The output must be
a new file. Source hashes identify the actual working files, including edits
that are not represented by the checkout's HEAD commit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    root = args.research_root.resolve()
    # Give the research package a separate namespace. Avoid its broad top-level
    # __init__ (which imports unrelated datasets and optional calibrators).
    for name, path in (
        ("fiducio_reference", root / "src/fiducio"),
        ("fiducio_reference.calibrators", root / "src/fiducio/calibrators"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    conditional = importlib.import_module(
        "fiducio_reference.calibrators.order_preserving_dirichlet"
    )
    matrix = importlib.import_module("fiducio_reference.calibrators.matrix")
    torch.set_num_threads(1)
    cases = []
    for classes in (2, 4):
        for seed in (0, 1):
            g = torch.Generator().manual_seed(seed)
            x = torch.randn(12, classes, generator=g)
            x[:classes] += 3 * torch.eye(classes)
            y = torch.arange(12) % classes
            for method in ("CDC", "CMSap", "CMSop", "MSc", "MS"):
                lam, mu = 0.13, 0.07
                if method in ("MS", "MSc"):
                    ref = matrix.MatrixScaling(
                        device="cpu", enforce_row_sum_invariance=method == "MSc"
                    )
                    weight = torch.randn(classes, classes, generator=g) * 0.2
                    bias = torch.randn(classes, generator=g) * 0.1
                    ref.b = torch.nn.Parameter(bias)
                    if method == "MSc":
                        ref.W_free = torch.nn.Parameter(weight[:, :-1].clone())
                        ref.row_sum_target = torch.nn.Parameter(torch.tensor(0.8))
                        params = [ref.W_free, ref.b, ref.row_sum_target]
                    else:
                        ref.W = torch.nn.Parameter(weight)
                        params = [ref.W, ref.b]

                    def evaluate(ref=ref, classes=classes, lam=lam, mu=mu, x=x):
                        w, apply = ref._get_W_matrices()
                        off = ~torch.eye(classes, dtype=torch.bool)
                        # This penalty is local to research fit(), rather than
                        # exposed as a method; use its documented ODIR formula.
                        reg = lam * w[off].square().mean() + mu * ref.b.square().mean()
                        return x @ apply + ref.b, reg
                else:
                    name = {
                        "CDC": "ClassConditionalDirichletCalibrator",
                        "CMSap": "LogitRegularizedArgmaxPreservingCCDirichletCalibrator",
                        "CMSop": "LogitRegularizedOrderPreservingCCDirichletCalibrator",
                    }[method]
                    ref = getattr(conditional, name)(
                        device="cpu", num_classes=classes, lambda_reg=lam, mu_reg=mu
                    )
                    params = [ref.A, ref.b] if method == "CDC" else [ref.raw_B, ref.raw_mu]
                    with torch.no_grad():
                        for p in params:
                            p.copy_(torch.randn(p.shape, generator=g) * 0.3)

                    def evaluate(ref=ref, x=x):
                        return ref._logits_from_log_probs_flat(x.log_softmax(1)), ref._regularization()

                initial = [p.detach().tolist() for p in params]
                logits, reg = evaluate()
                loss = F.cross_entropy(logits, y) + reg
                gradients = torch.autograd.grad(loss, params)
                optimizer = torch.optim.Adam(params, lr=0.01)
                for p, grad in zip(params, gradients, strict=True):
                    p.grad = grad
                optimizer.step()
                updated_logits, updated_reg = evaluate()
                cases.append({
                    "method": method, "classes": classes, "seed": seed,
                    "inputs": x.tolist(), "targets": y.tolist(), "params": initial,
                    "lambda_reg": lam, "mu_reg": mu, "logits": logits.detach().tolist(),
                    "regularization": reg.item(), "loss": loss.item(),
                    "gradients": [p.tolist() for p in gradients],
                    "updated_logits": updated_logits.detach().tolist(),
                    "updated_regularization": updated_reg.item(),
                })
    sources = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name.startswith("fiducio_reference") and file:
            path = Path(file).resolve()
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "scope": "Synthetic fixed-parameter maps, ODIR penalties, joint CE gradients and one Adam step; not paper training reproduction.",
        "research_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "research_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
        "source_sha256": sources,
        "torch_version": torch.__version__,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {len(cases)} synthetic reference cases to {args.output}")


if __name__ == "__main__":
    main()
