"""Run with ``pytest benchmarks/tests`` from the repository root (not part of the package tests)."""

import argparse
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import adapters  # noqa: E402
import data  # noqa: E402
import run  # noqa: E402

# the repository's pytest config turns warnings into errors; competitors emit their own
pytestmark = pytest.mark.filterwarnings("ignore")


def rows_of(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def test_temperature_scaling_recovers_oracle_temperature():
    z, y, _ = data.simulate("over", 4, 1, 200_000, 0, jitter=0.0)
    m = adapters.Fiducio(4)
    m.fit(z, y)
    assert m.m.temperature == pytest.approx(2.5, rel=0.05)


@pytest.mark.parametrize("name", [n for n in adapters.ADAPTERS if not n.startswith("fiducio")])
def test_adapter_runs(name):
    cls = adapters.ADAPTERS[name][0]
    pytest.importorskip({"none": "torch"}.get(cls.lib, cls.lib.replace("-", "_")))
    z, y, _ = data.simulate("over", 4, 2, 1000, 0)
    m = cls(4)
    m.fit(z, y)
    p = m.predict(z)
    assert p.shape == z.shape and bool(((p.sum(1) - 1).abs() < 1e-4).all())


def test_child_records_device(capsys):
    run.child(argparse.Namespace(method="fiducio TS", n=1000, reps=1, threads=1, device="cpu"))
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert {"fit_s", "transform_s", "peak_rss_mb"} <= row.keys()
    assert "peak_vram_mb" not in row  # VRAM is only meaningful on a CUDA run


def test_sync_only_waits_for_cuda_work(monkeypatch):
    calls = []
    monkeypatch.setattr(run.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run.torch.cuda, "synchronize", lambda: calls.append(True))
    run._sync("cpu")
    assert calls == []
    run._sync("cuda")
    assert calls == [True]


def test_gpu_scaling_plot_renders_both_themes(tmp_path):
    import plots

    rows = rows_of(Path(__file__).parents[1] / "results/scaling-gpu.csv")
    for mode in ("light", "dark"):
        path = tmp_path / f"scaling-gpu-{mode}.png"
        plots.scaling_gpu(rows, mode, path)
        assert path.stat().st_size > 10_000


def test_gpu_scaling_plot_removes_stale_image_when_all_rows_fail(tmp_path):
    import plots

    path = tmp_path / "scaling-gpu-light.png"
    path.write_bytes(b"stale")
    plots.scaling_gpu([{"device": "cuda", "status": "error"}], "light", path)
    assert not path.exists()


def test_report_omits_gpu_graph_when_cuda_rows_fail(tmp_path, monkeypatch):
    import shutil

    pytest.importorskip("pandas")
    source = Path(__file__).parents[1] / "results"
    results = tmp_path / "benchmarks" / "results"
    results.mkdir(parents=True)
    for name in ("calibration.csv", "scaling.csv", "downstream.csv", "robustness.csv"):
        shutil.copyfile(source / name, results / name)
    with (source / "scaling-gpu.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["device"] == "cuda":
            row["status"] = "error"
    with (results / "scaling-gpu.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    monkeypatch.setattr(run, "HERE", tmp_path / "benchmarks")
    monkeypatch.setattr(run.plots, "draw_all", lambda *_: None)
    out = tmp_path / "benchmark.md"
    run.report(argparse.Namespace(out=out))
    report = out.read_text()
    assert "No successful CUDA rows; graph omitted." in report
    assert "scaling-gpu-light.png" not in report


def test_downstream_rows_parseable(tmp_path):
    out = tmp_path / "downstream.csv"
    run.downstream(argparse.Namespace(seeds=1, cases=1, out=out))
    rows = rows_of(out)
    assert rows
    assert {"scenario", "seed", "method", "ece", "dice", "argmax_changed"} <= rows[0].keys()
    assert {r["method"] for r in rows} == set(adapters.DOWNSTREAM_ADAPTERS)
    assert all(r["method"] for r in rows)


def test_robustness_rows_parseable(tmp_path):
    out = tmp_path / "robustness.csv"
    methods = ["identity", "fiducio TS"]
    run.robustness(
        argparse.Namespace(seeds=1, cases=["saturated"], methods=methods, timeout=120, out=out)
    )
    rows = rows_of(out)
    assert len(rows) == len(methods)
    assert {r["method"] for r in rows} == set(methods)
    assert all(r["status"] in {"ok", "timeout", "oom", "error"} for r in rows)


@pytest.mark.parametrize(
    "name",
    [n for n in adapters.ROBUSTNESS_ADAPTERS if not n.startswith(("fiducio", "identity"))],
)
def test_robustness_competitor_rows(name, tmp_path):
    cls = adapters.ROBUSTNESS_ADAPTERS[name][0]
    pytest.importorskip({"none": "torch"}.get(cls.lib, cls.lib.replace("-", "_")))
    out = tmp_path / "robustness.csv"
    run.robustness(
        argparse.Namespace(seeds=1, cases=["float32"], methods=[name], timeout=120, out=out)
    )
    rows = rows_of(out)
    assert len(rows) == 1 and rows[0]["method"] == name
    assert rows[0]["status"] in {"ok", "timeout", "oom", "error"}
