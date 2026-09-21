"""Reject publishing from branches, version-mismatched tags or commits outside main."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import tomllib


def check(ref: str) -> None:
    with Path("pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    if not re.fullmatch(r"refs/tags/v\d+\.\d+\.\d+", ref) or ref != f"refs/tags/v{version}":
        raise ValueError("publishing requires a vX.Y.Z tag matching pyproject.toml")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(["git", "rev-parse", f"{ref}^{{commit}}"], text=True).strip()
    if head != tag:
        raise ValueError("checkout does not match the release tag")
    subprocess.run(["git", "merge-base", "--is-ancestor", head, "origin/main"], check=True)
    print(f"Validated release {version} at {head}")


if __name__ == "__main__":
    check(os.environ.get("GITHUB_REF", ""))
