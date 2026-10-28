"""Run with ``pytest benchmarks/tests`` from the repository root (not part of the package tests)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import adapters  # noqa: E402
import data  # noqa: E402

# the repository's pytest config turns warnings into errors; competitors emit their own
pytestmark = pytest.mark.filterwarnings("ignore")


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
