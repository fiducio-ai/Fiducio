"""Release metadata consistency checks."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_citation_version_matches_pyproject():
    pyproject = (ROOT / "pyproject.toml").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    cited = re.search(r'^version: "([^"]+)"', citation, re.M).group(1)
    assert cited == version


def test_changelog_has_dated_entry_for_released_version():
    pyproject = (ROOT / "pyproject.toml").read_text()
    changelog = (ROOT / "CHANGELOG.md").read_text()
    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.M)
