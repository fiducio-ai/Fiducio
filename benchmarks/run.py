"""Benchmark Fiducio against netcal, torch-uncertainty and dirichletcal on synthetic voxels.

    python run.py calibration [--seeds 10] [--out results/calibration.csv]
    python run.py scaling [--sizes 100000 1000000] [--timeout 600] [--device cpu] [--out results/scaling.csv]
    python run.py downstream [--seeds 10] [--out results/downstream.csv]
    python run.py robustness [--seeds 10] [--cases ...] [--out results/robustness.csv]
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
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import adapters
import data
import plots
import torch

HERE = Path(__file__).parent
VOX = 5000  # voxels per case
CASES = 20  # calibration and test cases per split in the calibration axis
DOWNSTREAM_SCENARIOS = ("bias", "rare")
ROBUST_CASES = ("saturated", "tiny", "class_absent", "float32", "float64")


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as handle:
            cpu = next(ln.split(":")[1].strip() for ln in handle if ln.startswith("model name"))
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


def _sync(device):
    """Wait for queued CUDA work, so a timing measures execution and not launch."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def fit_timed(cls, C, zc, yc):
    """Construct and fit an adapter, returning it with the fit duration in seconds."""
    m = cls(C)
    _sync(adapters.DEVICE)
    t = time.perf_counter()
    m.fit(zc, yc)
    _sync(adapters.DEVICE)
    return m, time.perf_counter() - t


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
                        m, row["fit_s"] = fit_timed(cls, C, zc, yc)
                        p = m.predict(zt)
                        row.update(scores(p, yt, C), **independent_scores(p, yt))
                        row["argmax_kept"] = (p.argmax(1) == zt.argmax(1)).float().mean().item()
                    except Exception:  # noqa: BLE001
                        row["error"] = traceback.format_exc()
                    rows.append(row)
                print(scenario, C, seed, flush=True)
    write(a.out, rows)


def dice_score(p, y, C):
    """Macro Dice over classes present in the labels."""
    pred = p.argmax(1)
    scores = []
    for c in range(C):
        present = y == c
        if bool(present.any()):
            inter = ((pred == c) & present).sum().item()
            denom = (pred == c).sum().item() + present.sum().item()
            scores.append(2 * inter / max(denom, 1))
    return sum(scores) / len(scores) if scores else float("nan")


def downstream_scores(p, y, C, reference_argmax):
    from fiducio.metrics import expected_calibration_error, negative_log_likelihood

    pred = p.argmax(1)
    out = dict(
        nll=negative_log_likelihood(p, y),
        ece=expected_calibration_error(p, y, n_bins=15),
        dice=dice_score(p, y, C),
        argmax_changed=(pred != reference_argmax).float().mean().item(),
    )
    for c in range(C):
        out[f"ece_c{c}"] = expected_calibration_error(p, y, n_bins=15, class_index=c)
        out[f"volume_bias_c{c}"] = (pred == c).float().mean().item() - (
            y == c
        ).float().mean().item()
    return out


def downstream(a):
    """Axis 4: Fiducio only, identity as the reference decision."""
    hw, rows, C = hardware(), [], 4
    for scenario in DOWNSTREAM_SCENARIOS:
        for seed in range(a.seeds):
            zc, yc, _ = data.simulate(scenario, C, a.cases, VOX, 2 * seed)
            zt, yt, _ = data.simulate(scenario, C, a.cases, VOX, 2 * seed + 1)
            base = dict(
                scenario=scenario,
                C=C,
                seed=seed,
                n_cal=len(yc),
                n_test=len(yt),
                rare_prev=(yt == C - 1).float().mean().item(),
                **hw,
            )
            reference = zt.argmax(1)
            for name, (cls, _) in adapters.DOWNSTREAM_ADAPTERS.items():
                row = dict(
                    base,
                    method=name,
                    lib=cls.lib,
                    dtype=cls.dtype,
                    lib_version=adapters.version(cls.lib),
                )
                try:
                    m, row["fit_s"] = fit_timed(cls, C, zc, yc)
                    p = m.predict(zt)
                    row.update(downstream_scores(p, yt, C, reference))
                except Exception:  # noqa: BLE001
                    row["error"] = traceback.format_exc()
                rows.append(row)
            print("downstream", scenario, seed, flush=True)
    write(a.out, rows)


@contextmanager
def deadline(seconds):
    """Raise ``TimeoutError`` after ``seconds`` (POSIX only; benchmarks run on Linux)."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def stop(signum, frame):
        raise TimeoutError(f"exceeded {seconds}s")

    old = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def robust_data(case, seed):
    """One robustness case: ``(calibration logits, labels, test logits, labels)``."""
    C = 4
    cases, vox = (1, 8) if case == "tiny" else (2, 1000)
    zc, yc, _ = data.simulate("over", C, cases, vox, 2 * seed)
    zt, yt, _ = data.simulate("over", C, cases, vox, 2 * seed + 1)
    if case == "saturated":
        zc, zt = zc * 1e4, zt * 1e4  # softmax underflows to exact 0/1
    elif case == "class_absent":
        keep = yc != C - 1
        zc, yc = zc[keep], yc[keep]
    elif case == "float64":
        zc, zt = zc.double(), zt.double()
    return zc, yc, zt, yt


def robustness(a):
    """Axis 5: status per (case, method, seed), including failures."""
    from fiducio.metrics import expected_calibration_error, negative_log_likelihood

    hw, rows, C = hardware(), [], 4
    for case in a.cases:
        for seed in range(a.seeds):
            zc, yc, zt, yt = robust_data(case, seed)
            base = dict(
                case=case,
                seed=seed,
                C=C,
                n_cal=len(yc),
                n_test=len(yt),
                dtype="float64" if case == "float64" else "float32",
                **hw,
            )
            for name, (cls, _) in adapters.ROBUSTNESS_ADAPTERS.items():
                if a.methods and name not in a.methods:
                    continue
                row = dict(
                    base,
                    method=name,
                    lib=cls.lib,
                    lib_version=adapters.version(cls.lib),
                )
                try:
                    with deadline(a.timeout):
                        m, row["fit_s"] = fit_timed(cls, C, zc, yc)
                        p = m.predict(zt)
                        row["nll"] = negative_log_likelihood(p, yt)
                        row["ece"] = expected_calibration_error(p, yt, n_bins=15)
                    row["status"] = "ok"
                except TimeoutError:
                    row["status"] = "timeout"
                except (MemoryError, torch.OutOfMemoryError):
                    row["status"] = "oom"
                except Exception:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = traceback.format_exc()
                rows.append(row)
                print("robustness", case, seed, name, row["status"], flush=True)
    write(a.out, rows)


def child(a):
    """One (method, N) measurement in a fresh process: median fit and transform times, peak RSS."""
    torch.set_num_threads(a.threads)
    adapters.use_device(a.device)
    cls = adapters.ADAPTERS[a.method][0]
    zc, yc, _ = data.simulate("over", 4, max(1, a.n // VOX), VOX, 0)
    zt, _, _ = data.simulate("over", 4, max(1, a.n // VOX), VOX, 1)
    cls(4).fit(zc[:2000], yc[:2000])  # warm-up
    _sync(a.device)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    fit, pred = [], []
    for _ in range(a.reps):
        m, dt = fit_timed(cls, 4, zc, yc)
        fit.append(dt)
        _sync(a.device)
        t = time.perf_counter()
        m.predict(zt)
        _sync(a.device)
        pred.append(time.perf_counter() - t)
    out = dict(
        fit_s=statistics.median(fit),
        transform_s=statistics.median(pred),
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    if a.device.startswith("cuda"):
        out["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps(out))


def scaling(a):
    hw, rows = hardware(), []
    hw["threads"] = a.threads
    if a.device.startswith("cuda"):
        hw["gpu"] = torch.cuda.get_device_name(0)
    for n in a.sizes:
        for name, (cls, _) in adapters.ADAPTERS.items():
            # netcal and dirichletcal are numpy/JAX and stay on the host CPU.
            device = a.device if cls.lib in {"fiducio", "torch-uncertainty"} else "cpu"
            row = dict(
                method=name,
                lib=cls.lib,
                lib_version=adapters.version(cls.lib),
                n=n,
                reps=a.reps,
                device=device,
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
                "--device",
                device,
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
| Native `(B, C, *spatial)` input, 2D and 3D | yes | no (flatten to `(N, C)`) | temperature only (other scalers need `(N, C)`) | no (flatten to `(N, C)`) |
| `mask` and `ignore_index` | yes | no (index before `fit`) | no mask (`-100` ignored by the loss) | no (index before `fit`) |
| Logits as input | yes | no, probabilities only | yes | no, probabilities only |
| Save and reload a fitted calibrator | yes | yes (`save_model`) | manual (`state_dict` + `set_model`) | not evaluated |
| Per-class bias in vector scaling | yes | n/a | no | n/a |
| Class-conditional calibrators (CDC, CMSap, CMSop) | yes | none among the methods tested | none among the methods tested | none among the methods tested |

"Flatten" means the adapter used here, not a limitation proven in the library; "not evaluated"
means the capability was not tested for this page; "n/a" means no vector-scaling method was
tested for that library.
"""

NOTES = """## Reading these results

* **Synthetic data only.** Labels are drawn from `softmax(z)`; the model sees `a_i * z` (plus a class
  bias in `bias`), with `a_i` varying per case. A scenario with a known oracle lets the
  temperature scalers be checked against a Bayes oracle (a lower bound); with `jitter=0`, TS
  recovers `T* = a` up to finite-sample error. It says nothing about nnU-Net logits.
* **One metric implementation.** NLL, ECE and ACE come from `fiducio.metrics` (15 uniform bins,
  pooled over voxels) for every method. ACE here is the unweighted mean over non-empty bins,
  not the adaptive ACE of other work. Cross-check against netcal's ECE (4 classes only: its binary
  ECE measures P(class 1), a different quantity) and scikit-learn's log loss on the same probabilities: max |ΔECE| = {dece:.2e}, max |ΔNLL| = {dnll:.2e}.
* **Differences are paired**: same seed, same cases. The interval is 1.96 standard errors over
  seeds (a normal approximation; with 10 seeds the exact t multiplier is 2.26); each seed
  redraws all cases, so it covers case-to-case variance.
* **Input contract.** Fiducio and torch-uncertainty receive logits; netcal and dirichletcal receive
  `softmax(z)` in float64. dirichletcal runs on JAX, which computes in float32 unless x64 is
  enabled (probabilities differ by a few 1e-6 between the two modes).
* **netcal pitfall.** With float32 probabilities `TemperatureScaling.fit` leaves T = 1 without a
  warning; the adapter casts to float64. Its `T` multiplies the logits, Fiducio's divides them.
* **Defaults.** Each library runs with its default optimiser budget; no per-library tuning.
"""


def ci(s):
    return 1.96 * s.std() / len(s) ** 0.5 if len(s) > 1 else float("nan")


def figure(name, alt):
    """Material swaps the two images with the colour scheme (``#only-light`` / ``#only-dark``)."""
    base = f"assets/benchmark/{name}"
    return f"![{alt}]({base}-light.png#only-light)\n![{alt}]({base}-dark.png#only-dark)\n"


def fmt(mean, ci, p=4):
    def one(x):
        s = f"{x:.{p}f}"
        return s[1:] if s.startswith("-") and abs(x) < 0.5 * 10**-p else s

    return f"{one(mean)} ± {one(ci)}"


def report(a):
    import pandas as pd

    cal = pd.read_csv(HERE / "results/calibration.csv")
    sc = pd.read_csv(HERE / "results/scaling.csv")
    ds = pd.read_csv(HERE / "results/downstream.csv")
    rb = pd.read_csv(HERE / "results/robustness.csv")
    ok = cal[cal["error"].isna()] if "error" in cal else cal
    dsok = ds[ds["error"].isna()] if "error" in ds else ds
    plots.draw_all(ok, sc, HERE.parent / "docs/assets/benchmark")
    out = [
        "# Benchmark",
        "",
        "Fiducio against netcal, torch-uncertainty and dirichletcal on synthetic voxels. "
        "Generated by `benchmarks/run.py report`; do not edit by hand.",
        "",
    ]
    v = {}
    for frame in (cal, ds, rb):
        if "lib_version" not in frame:
            continue
        for r in frame.drop_duplicates("lib").itertuples():
            if isinstance(r.lib_version, str) and r.lib_version:
                v.setdefault(r.lib, r.lib_version)
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
        "The figure shows NLL excess and ECE; the tables also report NLL, ECE and ECE of the last "
        "class (the rare one in `rare`), mean ± 95% CI over seeds. Δ NLL is paired against the "
        "matching Fiducio method.",
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
    out += [
        "Δ NLL compares each method with the matching Fiducio calibrator. In `bias`, Fiducio's "
        "vector scaling fits a per-class bias (regularised by `mu_reg`) while torch-uncertainty's "
        "`VectorScaler` has none, so its Δ NLL in that scenario measures capacity, not "
        "implementation quality.",
        "",
    ]
    bad = cal[cal["error"].notna()] if "error" in cal else cal.iloc[:0]
    if len(bad):
        out += ["Failures: " + ", ".join(sorted(set(bad.method))) + " (tracebacks in the CSV).", ""]
    out += [
        "## Downstream (Fiducio only)",
        "",
        "Same paired seeds as the calibration axis, 4 classes, scenarios with a per-class bias "
        "(`bias`) and a rare class (`rare`), and identity as the reference decision. None of the "
        "competitors' tested methods is class-conditional, so this axis is Fiducio only.",
        "",
        "Per-class ECE uses the class-specific metric (`class_index`), and the **volume bias** of a "
        "class is its predicted voxel share minus its label share (signed: negative means "
        "under-segmentation). Δ Dice is paired against identity over seeds. `argmax changed` is the "
        "share of test voxels whose argmax differs from the uncalibrated one.",
        "",
    ]
    for scenario in DOWNSTREAM_SCENARIOS:
        d = dsok[dsok.scenario == scenario]
        header = (
            "| method | NLL | ECE | "
            + " | ".join(f"ECE c{c}" for c in range(4))
            + " | Dice | Δ Dice | argmax changed | "
            + " | ".join(f"bias c{c}" for c in range(4))
            + " |"
        )
        align = "|---|--:|--:|" + "--:|" * 4 + "--:|--:|--:|" + "--:|" * 4
        out += [f"### {scenario}", "", header, align]
        for name in adapters.DOWNSTREAM_ADAPTERS:
            g = d[d.method == name].set_index("seed")
            if g.empty:
                out.append("| " + " | ".join([name, "failed", *([""] * 12)]) + " |")
                continue
            diff = (g.dice - d[d.method == "identity"].set_index("seed").dice).dropna()
            cells = [fmt(g.nll.mean(), ci(g.nll)), fmt(g.ece.mean(), ci(g.ece))]
            cells += [fmt(g[f"ece_c{c}"].mean(), ci(g[f"ece_c{c}"])) for c in range(4)]
            cells += [
                fmt(g.dice.mean(), ci(g.dice)),
                fmt(diff.mean(), ci(diff), 4) if len(diff) else "",
                fmt(g.argmax_changed.mean(), ci(g.argmax_changed), 4),
            ]
            cells += [
                fmt(g[f"volume_bias_c{c}"].mean(), ci(g[f"volume_bias_c{c}"]), 4) for c in range(4)
            ]
            out.append("| " + " | ".join([name, *cells]) + " |")
        out.append("")
    out += [
        "TS, CMSap and CMSop preserve the argmax in exact arithmetic (CMSop the full class order), "
        "so their Dice and argmax change are zero by construction, up to floating-point ties; "
        "VS, MS and DC can change the decision, so Dice is informative for them. In these runs the "
        "preserving methods improve ECE but keep identity's volume bias, while VS, MS and DC change "
        "the decision and reduce it; in `rare`, the decision-changing methods also lower Dice "
        "slightly.",
        "",
    ]
    dsbad = ds[ds["error"].notna()] if "error" in ds else ds.iloc[:0]
    if len(dsbad):
        out += [
            "Failures: " + ", ".join(sorted(set(dsbad.method))) + " (tracebacks in the CSV).",
            "",
        ]

    robust_notes = {
        "saturated": "logits scaled by 10⁴ so that `softmax` underflows to exact 0/1; netcal and "
        "dirichletcal receive those probabilities, Fiducio and torch-uncertainty the logits, and "
        "the `(probs)` variants exercise Fiducio's `log(p)` path.",
        "tiny": "one case and 8 voxels per split.",
        "class_absent": "class 3 removed from the calibration split; the test split keeps it.",
        "float32": "inputs in float32; `netcal TS (raw dtype)` skips the adapter's float64 cast.",
        "float64": "inputs in float64.",
    }
    out += [
        "## Robustness",
        "",
        "Every (case, method, seed) is a row in `results/robustness.csv` and appears below; "
        "`timeout`, `oom` and `error` are results, not omissions. Status is counted over seeds; "
        "NLL and ECE are medians over the successful seeds. Each fit-and-transform is bounded by "
        "an in-process SIGALRM deadline (not a fresh process, unlike the scaling axis), so a "
        "timeout is recorded per row. An `error` is the adapter/library combination, not "
        "necessarily a library defect: torch-uncertainty's matrix scaler does not accept float64 "
        "through this adapter, and dirichletcal rejects a missing class.",
        "",
    ]
    for case in ROBUST_CASES:
        d = rb[rb.case == case]
        out += [
            f"### {case}",
            "",
            robust_notes.get(case, ""),
            "",
            "| method | status | NLL (median) | ECE (median) |",
            "|---|---|--:|--:|",
        ]
        for name in adapters.ROBUSTNESS_ADAPTERS:
            g = d[d.method == name]
            if g.empty:
                continue
            counts = Counter(g.status)
            status = ", ".join(f"{k} ×{n}" for k, n in sorted(counts.items()))
            good = g[g.status == "ok"].dropna(subset=["nll", "ece"])
            nll = f"{good.nll.median():.4f}" if len(good) else ""
            ece = f"{good.ece.median():.4f}" if len(good) else ""
            out.append(f"| {name} | {status} | {nll} | {ece} |")
        out.append("")
    rb_totals = Counter(rb.status)
    out += [
        "Status totals over every robustness row: "
        + ", ".join(f"{k} {n}" for k, n in sorted(rb_totals.items()))
        + ".",
        "",
    ]
    raw = rb[(rb.method == "netcal TS (raw dtype)") & (rb.status == "ok")]
    f32 = raw[raw.case == "float32"].nll.median()
    f64 = raw[raw.case == "float64"].nll.median()
    ident = rb[(rb.method == "identity") & (rb.case == "float32")].nll.median()
    if pd.notna(f32) and pd.notna(f64):
        out += [
            f"With raw float32 probabilities netcal leaves `T = 1`: its median NLL is {f32:.4f}, "
            f"the same as identity ({ident:.4f}), against {f64:.4f} in float64. The main adapter "
            "avoids this by casting to float64.",
            "",
        ]

    def scaling_table(frame):
        vram = "peak_vram_mb" in frame.columns
        has_dev = "device" in frame.columns
        rows = [
            "| N | method |"
            + (" device |" if has_dev else "")
            + " status | fit (s) | transform (s) | peak RSS (MB) |"
            + (" peak VRAM (MB) |" if vram else ""),
            "|--:|---|"
            + (":-:|" if has_dev else "")
            + "---|--:|--:|--:|"
            + ("--:|" if vram else ""),
        ]
        f = lambda x: "" if pd.isna(x) else f"{x:.2f}"  # noqa: E731
        tf = lambda x: "" if pd.isna(x) else f"{x:.4f}" if vram else f(x)  # noqa: E731
        for r in frame.itertuples():
            device = "" if not has_dev or pd.isna(r.device) else f" {r.device} |"
            row = (
                f"| {r.n} | {r.method} |{device} {r.status} | {f(r.fit_s)} | "
                f"{tf(r.transform_s)} | {f(r.peak_rss_mb)} |"
            )
            if vram:
                row += f" {f(r.peak_vram_mb)} |"
            rows.append(row)
        return rows

    out += [
        "## Scaling",
        "",
        figure("tradeoff", "Fit time against peak memory at one million voxels"),
        figure("scaling", "Fit time and peak memory from 100 thousand to one million voxels"),
        "Scenario `over`, 4 classes, one process per row.",
        "",
        "Fit times reflect each library's default optimiser budget: Fiducio's temperature scaling "
        "uses Adam with `max_iter=200`, which dominates its fit time at 1M voxels, while netcal's "
        "default is much cheaper.",
        "",
        f"The table reports medians over {int(sc.reps.iloc[0])} repetitions; the figures show no spread.",
        "",
        *scaling_table(sc),
        "",
    ]
    gpu_path = HERE / "results/scaling-gpu.csv"
    if gpu_path.exists():
        gpu = pd.read_csv(gpu_path)
        out += [
            "## Scaling (GPU)",
            "",
            f"Same scenario and process, on {gpu.iloc[0].gpu} (torch {gpu.iloc[0].torch}). "
            "Fiducio and torch-uncertainty run on the GPU; netcal and dirichletcal are numpy/JAX "
            "and stay on the CPU, so only the CUDA rows are comparable with each other, and only "
            "on this machine.",
            "",
            figure(
                "scaling-gpu",
                f"CUDA torch methods on {gpu.iloc[0].gpu}: median fit time, transform time, and peak VRAM over input sizes",
            ),
            f"The graph shows CUDA rows only; the table also includes CPU baselines. Medians over "
            f"{int(gpu.reps.iloc[0])} repetitions; no spread is shown.",
            "",
            *scaling_table(gpu),
            "",
        ]
    out += [
        "## Features",
        "",
        FEATURES,
        "## Reproduce",
        "",
        "```bash",
        "cd benchmarks",
        "pip install -r requirements.txt  # Python 3.12 or 3.13",
        "python run.py calibration && python run.py scaling && "
        "python run.py downstream && python run.py robustness && python run.py report",
        "# with a CUDA build of torch, for the GPU table:",
        "python run.py scaling --device cuda --out results/scaling-gpu.csv && python run.py report",
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
    s.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    s.add_argument("--out", default=HERE / "results/scaling.csv")
    d = sub.add_parser("downstream")
    d.add_argument("--seeds", type=int, default=10)
    d.add_argument("--cases", type=int, default=CASES)
    d.add_argument("--out", default=HERE / "results/downstream.csv")
    b = sub.add_parser("robustness")
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--cases", nargs="+", default=list(ROBUST_CASES))
    b.add_argument("--methods", nargs="+", default=None)
    b.add_argument("--timeout", type=int, default=120)
    b.add_argument("--out", default=HERE / "results/robustness.csv")
    k = sub.add_parser("child")
    k.add_argument("--method", required=True)
    k.add_argument("--n", type=int, required=True)
    k.add_argument("--reps", type=int, default=3)
    k.add_argument("--threads", type=int, default=4)
    k.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    r = sub.add_parser("report")
    r.add_argument("--out", default=HERE.parent / "docs/benchmark.md")
    a = p.parse_args()
    {
        "calibration": calibration,
        "scaling": scaling,
        "child": child,
        "report": report,
        "downstream": downstream,
        "robustness": robustness,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
