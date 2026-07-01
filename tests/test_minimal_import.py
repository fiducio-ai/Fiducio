"""Importing fiducio must not require any optional dependency."""

from __future__ import annotations

import subprocess
import sys


def test_import_does_not_pull_optional_dependencies():
    # Run in a fresh interpreter so we observe a clean module table.
    code = (
        "import sys; import fiducio; "
        "leaked = [m for m in ('matplotlib', 'monai', 'nibabel') if m in sys.modules]; "
        "assert not leaked, leaked; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_api_is_importable():
    import fiducio

    for name in [
        "Calibrator",
        "TemperatureScaling",
        "EnsembleTemperatureScaling",
        "VectorScaling",
        "MatrixScaling",
        "TranslationInvariantMatrixScaling",
        "DirichletCalibration",
        "ClassConditionalMatrixScaling",
        "ArgmaxPreservingMatrixScaling",
        "OrderPreservingMatrixScaling",
        "save_calibrator",
        "load_calibrator",
        "negative_log_likelihood",
        "expected_calibration_error",
        "brier_score",
    ]:
        assert hasattr(fiducio, name), name


def test_aliases_point_to_classes():
    import fiducio

    assert fiducio.TS is fiducio.TemperatureScaling
    assert fiducio.ETS is fiducio.EnsembleTemperatureScaling
    assert fiducio.VS is fiducio.VectorScaling
    assert fiducio.MS is fiducio.MatrixScaling
    assert fiducio.MSc is fiducio.TranslationInvariantMatrixScaling
    assert fiducio.CMS is fiducio.ClassConditionalMatrixScaling
    assert fiducio.CMSAP is fiducio.ArgmaxPreservingMatrixScaling
    assert fiducio.CMSOP is fiducio.OrderPreservingMatrixScaling
    # Paper-shorthand aliases for the class-conditional family.
    assert fiducio.CDC is fiducio.ClassConditionalMatrixScaling
    assert fiducio.CMSap is fiducio.ArgmaxPreservingMatrixScaling
    assert fiducio.CMSop is fiducio.OrderPreservingMatrixScaling
