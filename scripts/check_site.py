"""Check the built paper page, its relative assets and byte-for-byte copy."""
from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if value and key in {"src", "href"}:
                self.refs.append(value)


def check(source: Path, site: Path) -> None:
    page = site / "paper/index.html"
    parser = References()
    parser.feed(page.read_text())
    for ref in parser.refs:
        url = urlsplit(ref)
        if url.scheme or url.netloc or not url.path:
            continue
        target = page.parent / unquote(url.path)
        if not target.exists():
            raise ValueError(f"missing built page resource: {ref}")
    css = (site / "paper/assets/page.css").read_text()
    for ref in re.findall(r"url\(['\"]?([^)'\"]+)", css):
        if not urlsplit(ref).scheme and not (site / "paper/assets" / ref).exists():
            raise ValueError(f"missing stylesheet resource: {ref}")
    for path in source.rglob("*"):
        if path.is_file():
            built = site / "paper" / path.relative_to(source)
            if not built.is_file() or path.read_bytes() != built.read_bytes():
                raise ValueError(f"paper asset differs or is missing: {path}")
    if not (site / "index.html").is_file():
        raise ValueError("documentation homepage is missing")
    print("Paper page, assets and documentation links verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/paper"))
    parser.add_argument("--site", type=Path, default=Path("site"))
    args = parser.parse_args()
    check(args.source, args.site)
