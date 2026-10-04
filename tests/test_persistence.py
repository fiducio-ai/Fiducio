"""Round-trip and safety tests for save / load."""

from __future__ import annotations

import pytest
import torch

from conftest import ALL_CALIBRATOR_IDS, make_calibrator, synthetic_logits
from fiducio import MatrixScaling, load_calibrator, save_calibrator
from fiducio.persistence import FORMAT_NAME


@pytest.mark.parametrize("calibrator_id", ALL_CALIBRATOR_IDS)
def test_save_load_roundtrip(tmp_path, calibrator_id):
    logits, labels = synthetic_logits((3, 3, 6, 6), seed=30)
    cal = make_calibrator(calibrator_id).fit(logits, labels)
    expected = cal.transform(logits)

    path = tmp_path / "calibrator.pt"
    cal.save(path)
    loaded = load_calibrator(path)

    assert loaded.calibrator_id == calibrator_id
    assert loaded.num_classes == 3
    assert loaded.is_fitted
    assert str(loaded.device) == "cpu"
    got = loaded.transform(logits)
    assert torch.allclose(expected, got, atol=1e-6)


def test_save_load_function_api(tmp_path):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=31)
    cal = make_calibrator("matrix_scaling").fit(logits, labels)
    path = tmp_path / "ms.pt"
    save_calibrator(cal, path)
    loaded = load_calibrator(path)
    assert torch.allclose(cal.transform(logits), loaded.transform(logits), atol=1e-6)


def test_payload_contains_metadata(tmp_path):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=32)
    cal = make_calibrator("temperature_scaling").fit(logits, labels)
    path = tmp_path / "ts.pt"
    cal.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format"] == FORMAT_NAME
    assert payload["calibrator_id"] == "temperature_scaling"
    assert "fiducio_version" in payload
    assert payload["config"]["input_type"] == "logits"
    # no module path is stored in the file
    flat = str(payload)
    assert "fiducio.calibrators" not in flat


def test_map_location_default_is_cpu(tmp_path):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=33)
    cal = make_calibrator("vector_scaling").fit(logits, labels)
    path = tmp_path / "vs.pt"
    cal.save(path)
    loaded = load_calibrator(path)  # default map_location='cpu'
    assert str(loaded.device) == "cpu"


def test_load_rejects_non_fiducio_file(tmp_path):
    path = tmp_path / "bogus.pt"
    torch.save({"hello": "world"}, path)
    with pytest.raises(ValueError, match="not a Fiducio calibrator file"):
        load_calibrator(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibrator(tmp_path / "does_not_exist.pt")


def test_unknown_calibrator_id_raises():
    from fiducio.registry import get_calibrator_class

    with pytest.raises(KeyError, match="unknown calibrator id"):
        get_calibrator_class("not_a_real_calibrator")


def test_unknown_calibrator_id_in_payload_raises_value_error(tmp_path):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=34)
    cal = MatrixScaling(device="cpu").fit(logits, labels)
    path = tmp_path / "unknown_id.pt"
    cal.save(path)
    payload = torch.load(path, weights_only=True)
    payload["calibrator_id"] = "not_a_real_calibrator"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported calibrator id"):
        load_calibrator(path)


def test_float64_saved_state_is_cast_on_load(tmp_path):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=35)
    cal = MatrixScaling(device="cpu").fit(logits, labels)
    expected = cal.transform(logits)
    path = tmp_path / "float64.pt"
    cal.save(path)
    payload = torch.load(path, weights_only=True)
    payload["state"] = {
        key: value.double() if torch.is_tensor(value) else value
        for key, value in payload["state"].items()
    }
    torch.save(payload, path)
    loaded = load_calibrator(path)
    assert torch.allclose(expected, loaded.transform(logits), atol=1e-6)


def test_junk_file_raises_value_error(tmp_path):
    path = tmp_path / "junk.pt"
    path.write_bytes(b"\x00\x01\x02not a torch file")
    with pytest.raises(ValueError):
        load_calibrator(path)


def test_failed_save_keeps_existing_file(tmp_path, monkeypatch):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=36)
    cal = MatrixScaling(device="cpu").fit(logits, labels)
    path = tmp_path / "cal.pt"
    cal.save(path)
    before = path.read_bytes()

    def fail_save(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr("fiducio.persistence.torch.save", fail_save)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        cal.save(path)
    assert path.read_bytes() == before
    assert [f.name for f in tmp_path.iterdir()] == ["cal.pt"]


def test_save_ignores_a_stale_fixed_name_temporary(tmp_path):
    cal = MatrixScaling(device="cpu")
    path = tmp_path / "cal.pt"
    (tmp_path / "cal.pt.tmp").mkdir()  # 0.1.1 always wrote <path>.tmp
    cal.save(path)
    assert load_calibrator(path).calibrator_id == "matrix_scaling"


@pytest.mark.filterwarnings("ignore:Validating sparse tensor invariants:UserWarning")
def test_sparse_saved_state_raises_value_error(tmp_path):
    logits, labels = synthetic_logits((2, 3, 5, 5), seed=37)
    path = tmp_path / "sparse.pt"
    MatrixScaling(device="cpu").fit(logits, labels).save(path)
    payload = torch.load(path, weights_only=True)
    payload["state"]["weight"] = payload["state"]["weight"].to_sparse()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite floating tensor"):
        load_calibrator(path)


def test_invalid_map_location_rejected(tmp_path):
    path = tmp_path / "cal.pt"
    MatrixScaling(device="cpu").save(path)
    with pytest.raises(TypeError, match="map_location"):
        load_calibrator(path, map_location={"cuda:0": "cpu"})
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            load_calibrator(path, map_location="cuda")
