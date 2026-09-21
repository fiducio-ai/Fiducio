"""Validate source parity, wheel records and release contents before publication."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import tarfile
import zipfile
from pathlib import Path


def check(dist: Path) -> None:
    wheels, sources = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as wheel, tarfile.open(sources[0]) as sdist:
        files = {m.name.split("/", 1)[1]: sdist.extractfile(m).read()
                 for m in sdist.getmembers() if m.isfile()}
        names = wheel.namelist()
        if "fiducio/py.typed" not in names or not any(n.endswith("/licenses/LICENSE") for n in names):
            raise ValueError("wheel is missing typing marker or license")
        for name in names:
            if name.startswith("fiducio/") and files.get("src/" + name) != wheel.read(name):
                raise ValueError(f"wheel/sdist source mismatch: {name}")
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        for name, digest, size in csv.reader(io.StringIO(wheel.read(record).decode())):
            if digest:
                content = wheel.read(name)
                expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
                if digest != expected or int(size) != len(content):
                    raise ValueError(f"invalid wheel record: {name}")
        for required in ["docs/paper/index.html", "docs/paper/assets/page.js",
                         "docs/paper/assets/page.css", "docs/paper/assets/fonts/OFL-Newsreader.txt",
                         "docs/paper/assets/fonts/OFL-IBMPlexSans.txt",
                         "docs/paper/assets/fonts/OFL-IBMPlexMono.txt"]:
            if required not in files:
                raise ValueError(f"sdist missing {required}")
        for path in Path("docs/paper").rglob("*"):
            if path.is_file() and files.get(path.as_posix()) != path.read_bytes():
                raise ValueError(f"sdist missing or changed page asset: {path}")
        if any("/paper/" in name for name in names):
            raise ValueError("project page must not be bundled in the runtime wheel")
    print("Distribution contents, RECORD and source parity verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, nargs="?", default=Path("dist"))
    check(parser.parse_args().dist)
