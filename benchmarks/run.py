"""Benchmark Fiducio against netcal, torch-uncertainty and dirichletcal on synthetic voxels.

    python run.py calibration [--seeds 10] [--out results/calibration.csv]
    python run.py scaling [--sizes 100000 1000000] [--timeout 600] [--out results/scaling.csv]
    python run.py report [--out ../docs/benchmark.md]

Competitors are installed from ``requirements.txt`` (not an extra of the package: their
Python ranges do not fit Fiducio's). A method that fails is recorded with its full traceback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import adapters
import data
import plots
import torch

HERE = Path(__file__).parent
VOX = 5000  # voxels per case
CASES = 20  # calibration and test cases per split in the calibration axis


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(
            ln.split(":")[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    return dict(
        cpu=cpu, cores=os.cpu_count(), threads=torch.get_num_threads(), torch=torch.__version__
    )


def scores(p, y, C):
    from fiducio.metrics import (
        brier_score,
        expected_calibration_error,
        negative_log_likelihood,
        reliability_curve,
    )

    return dict(
        nll=negative_log_likelihood(p, y),
        ece=expected_calibration_error(p, y, n_bins=15),
        ace=reliability_curve(p, y, n_bins=15).ace,
        brier=brier_score(p, y),
        ece_rare=expected_calibration_error(p, y, n_bins=15, class_index=C - 1),
    )


def independent_scores(p, y):
    """ECE and NLL from netcal and scikit-learn on the same probabilities (cross-check)."""
    try:
        from netcal.metrics import ECE
        from sklearn.metrics import log_loss

        pn, yn = p.double().numpy(), y.numpy()
        out = dict(nll_indep=float(log_loss(yn, pn, labels=list(range(p.shape[1])))))
        if p.shape[1] > 2:  # netcal's binary ECE measures P(class 1), not top-label confidence
            out["ece_indep"] = float(ECE(bins=15).measure(pn, yn))
        return out
    except Exception:  # noqa: BLE001
        return {}


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def calibration(a):
    hw, rows = hardware(), []
    for scenario in data.SCENARIOS:
        for C in (2, 4):
            for seed in range(a.seeds):
                zc, yc, _ = data.simulate(scenario, C, CASES, VOX, 2 * seed)
                zt, yt, oracle = data.simulate(scenario, C, CASES, VOX, 2 * seed + 1)
                base = dict(
                    scenario=scenario,
                    C=C,
                    seed=seed,
                    n_cal=len(yc),
                    rare_prev=(yt == C - 1).float().mean().item(),
                    **hw,
                )
                rows.append(
                    dict(base, method="oracle", lib="none", **scores(oracle.softmax(1), yt, C))
                )
                for name, (cls, _) in adapters.ADAPTERS.items():
                    row = dict(
                        base,
                        method=name,
                        lib=cls.lib,
                        dtype=cls.dtype,
                        lib_version=adapters.version(cls.lib),
                    )
                    try:
                        m = cls(C)
                        t = time.perf_counter()
                        m.fit(zc, yc)
                        row["fit_s"] = time.perf_counter() - t
                        p = m.predict(zt)
                        row.update(scores(p, yt, C), **independent_scores(p, yt))
                        row["argmax_kept"] = (p.argmax(1) == zt.argmax(1)).float().mean().item()
                    except Exception:  # noqa: BLE001
                        row["error"] = traceback.format_exc()
                    rows.append(row)
                print(scenario, C, seed, flush=True)
    write(a.out, rows)


def child(a):
    """One (method, N) measurement in a fresh process: median fit and transform times, peak RSS."""
    torch.set_num_threads(a.threads)
    cls = adapters.ADAPTERS[a.method][0]
    zc, yc, _ = data.simulate("over", 4, max(1, a.n // VOX), VOX, 0)
    zt, _, _ = data.simulate("over", 4, max(1, a.n // VOX), VOX, 1)
    cls(4).fit(zc[:2000], yc[:2000])  # warm-up
    fit, pred = [], []
    for _ in range(a.reps):
        m = cls(4)
        t = time.perf_counter()
        m.fit(zc, yc)
        fit.append(time.perf_counter() - t)
        t = time.perf_counter()
        m.predict(zt)
        pred.append(time.perf_counter() - t)
    print(
        json.dumps(
            dict(
                fit_s=statistics.median(fit),
                transform_s=statistics.median(pred),
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            )
        )
    )


def scaling(a):
    hw, rows = hardware(), []
    hw["threads"] = a.threads
    for n in a.sizes:
        for name, (cls, _) in adapters.ADAPTERS.items():
            row = dict(
                method=name,
                lib=cls.lib,
                lib_version=adapters.version(cls.lib),
                n=n,
                reps=a.reps,
                **hw,
            )
            cmd = [
                sys.executable,
                __file__,
                "child",
                "--method",
                name,
                "--n",
                str(n),
                "--reps",
                str(a.reps),
                "--threads",
                str(a.threads),
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, cwd=HERE)
                if r.returncode == 0:
                    row.update(status="ok", **json.loads(r.stdout.strip().splitlines()[-1]))
                else:
                    row.update(
                        status="oom" if r.returncode == -9 else "error", error=r.stderr[-2000:]
                    )
            except subprocess.TimeoutExpired:
                row["status"] = "timeout"
            rows.append(row)
            print(n, name, row["status"], flush=True)
    write(a.out, rows)


FEATURES = """| | Fiducio | netcal | torch-uncertainty | dirichletcal |
|---|:-:|:-:|:-:|:-:|
| Native `(B, C, *spatial)` input, 2D and 3D | yes | no (flatten to `(N, C)`) | not evaluated | no (flatten to `(N, C)`) |
| `mask` and `ignore_index` | yes | no (index before `fit`) | not evaluated | no (index before `fit`) |
| Logits as input | yes | no, probabilities only | yes | no, probabilities only |
| Save and reload a fitted calibrator | yes | yes (`save_model`) | not evaluated | not evaluated |
| Class-conditional calibrators (CDC, CMSap, CMSop) | yes | none among the methods tested | none among the methods tested | none among the methods tested |

"Flatten" means the adapter used here, not a limitation proven in the library; "not evaluated"
means the capability was not tested for this page.
"""

NOTES = """## Reading these results

* **Synthetic data only.** Labels are drawn from `softmax(z)`; the model sees `a_i * z` (plus a class
  bias in `bias`), with `a_i` varying per case. A scenario with a known oracle lets the
  temperature scalers be checked against the exact optimum. It says nothing about nnU-Net logits.
* **One metric implementation.** NLL, ECE and ACE come from `fiducio.metrics` (15 uniform bins,
  pooled over voxels) for every method. ACE here is the unweighted mean over non-empty bins,
  not the adaptive ACE of other work. Cross-check against netcal's ECE (4 classes only: its binary
  ECE measures P(class 1), a different quantity) and scikit-learn's log loss on the same probabilities: max |ΔECE| = {dece:.2e}, max |ΔNLL| = {dnll:.2e}.
* **Differences are paired**: same seed, same cases. The interval is 1.96 standard errors over
  seeds; each seed redraws all cases, so it covers case-to-case variance.
* **Input contract.** Fiducio and torch-uncertainty receive logits; netcal and dirichletcal receive
  `softmax(z)` in float64. dirichletcal runs on JAX, which computes in float32 unless x64 is
  enabled (probabilities differ by about 2e-6 between the two modes).
* **netcal pitfall.** With float32 probabilities `TemperatureScaling.fit` leaves T = 1 without a
  warning; the adapter casts to float64. Its `T` multiplies the logits, Fiducio's divides them.
* **Defaults.** Each library runs with its default optimiser budget; no per-library tuning.
* **Timings** are medians of fit and transform measured separately, each (method, N) in a fresh
  process with a timeout. `timeout`, `oom` and `error` are results, not omissions.
"""


def ci(s):
    return 1.96 * s.std() / len(s) ** 0.5 if len(s) > 1 else float("nan")


def figure(name, alt):
    """Material swaps the two images with the colour scheme (``#only-light`` / ``#only-dark``)."""
    base = f"assets/benchmark/{name}"
    return f"![{alt}]({base}-light.png#only-light)\n![{alt}]({base}-dark.png#only-dark)\n"


def fmt(mean, ci, p=4):
    return f"{mean:.{p}f} ± {ci:.{p}f}"


def report(a):
    import pandas as pd

    cal = pd.read_csv(HERE / "results/calibration.csv")
    sc = pd.read_csv(HERE / "results/scaling.csv")
    ok = cal[cal["error"].isna()] if "error" in cal else cal
    plots.draw_all(ok, sc, HERE.parent / "docs/assets/benchmark")
    out = [
        "# Benchmark",
        "",
        "Fiducio against netcal, torch-uncertainty and dirichletcal on synthetic voxels. "
        "Generated by `benchmarks/run.py report`; do not edit by hand.",
        "",
    ]
    v = {
        r.lib: r.lib_version
        for r in cal.drop_duplicates("lib").itertuples()
        if isinstance(r.lib_version, str)
    }
    hw = cal.iloc[0]
    out += [
        "Versions: " + ", ".join(f"{k} {x}" for k, x in sorted(v.items())) + f", torch {hw.torch}. "
        f"Hardware: {hw.cpu}, {hw.cores} cores, {hw.threads} torch threads.",
        "",
    ]
    dece = (ok.ece - ok.ece_indep).abs().max()
    dnll = (ok.nll - ok.nll_indep).abs().max()
    out += [NOTES.format(dece=dece, dnll=dnll)]
    out += [
        "## Calibration",
        "",
        figure(
            "calibration",
            "Dot plots of NLL excess over the oracle and ECE for each scenario with 4 classes",
        ),
        "NLL, ECE and ECE of the last class (the rare one in `rare`), mean ± 95% CI over seeds. "
        "Δ NLL is paired against the matching Fiducio method.",
        "",
    ]
    for scenario in data.SCENARIOS:
        out += [
            f"### {scenario}",
            "",
            "| C | method | NLL | ECE | ECE last class | Δ NLL vs Fiducio |",
            "|--:|---|--:|--:|--:|--:|",
        ]
        for C in (2, 4):
            d = ok[(ok.scenario == scenario) & (ok.C == C)]
            for name in ["oracle", *adapters.ADAPTERS]:
                g = d[d.method == name].set_index("seed")
                if g.empty:
                    out.append(f"| {C} | {name} | failed | | | |")
                    continue
                ref = adapters.ADAPTERS.get(name, (None, None))[1]
                delta = ""
                if ref:
                    diff = (g.nll - d[d.method == ref].set_index("seed").nll).dropna()
                    delta = fmt(diff.mean(), 1.96 * diff.std() / len(diff) ** 0.5, 5)
                out.append(
                    f"| {C} | {name} | {fmt(g.nll.mean(), ci(g.nll))} | {fmt(g.ece.mean(), ci(g.ece))} | {fmt(g.ece_rare.mean(), ci(g.ece_rare))} | {delta} |"
                )
        out.append("")
    bad = cal[cal["error"].notna()] if "error" in cal else cal.iloc[:0]
    if len(bad):
        out += ["Failures: " + ", ".join(sorted(set(bad.method))) + " (tracebacks in the CSV).", ""]
    out += [
        "## Scaling",
        "",
        figure("tradeoff", "Fit time against peak memory at one million voxels"),
        figure("scaling", "Fit time and peak memory from 100 thousand to one million voxels"),
        "Scenario `over`, 4 classes, one process per row.",
        "",
        "| N | method | status | fit (s) | transform (s) | peak RSS (MB) |",
        "|--:|---|---|--:|--:|--:|",
    ]
    for r in sc.itertuples():
        f = lambda x: "" if pd.isna(x) else f"{x:.2f}"  # noqa: E731
        out.append(
            f"| {r.n} | {r.method} | {r.status} | {f(r.fit_s)} | {f(r.transform_s)} | {f(r.peak_rss_mb)} |"
        )
    out += [
        "",
        "## Features",
        "",
        FEATURES,
        "## Reproduce",
        "",
        "```bash",
        "pip install -r benchmarks/requirements.txt  # Python 3.12 or 3.13",
        "python benchmarks/run.py calibration && python benchmarks/run.py scaling && python benchmarks/run.py report",
        "```",
        "",
    ]
    Path(a.out).write_text("\n".join(out))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibration")
    c.add_argument("--seeds", type=int, default=10)
    c.add_argument("--out", default=HERE / "results/calibration.csv")
    s = sub.add_parser("scaling")
    s.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    s.add_argument("--timeout", type=int, default=600)
    s.add_argument("--reps", type=int, default=3)
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--out", default=HERE / "results/scaling.csv")
    k = sub.add_parser("child")
    k.add_argument("--method", required=True)
    k.add_argument("--n", type=int, required=True)
    k.add_argument("--reps", type=int, default=3)
    k.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--out", default=HERE.parent / "docs/benchmark.md")
    a = p.parse_args()
    {"calibration": calibration, "scaling": scaling, "child": child, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
