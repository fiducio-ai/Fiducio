# Contributing to Fiducio

Thanks for your interest in improving Fiducio! This document describes how to set
up a development environment and the conventions we follow.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) version
`0.12.3`, matching `tool.uv.required-version` and CI. The checkout defaults to
Python 3.12; CI also checks Python 3.10 and 3.11.

```bash
git clone https://github.com/fiducio-ai/Fiducio.git
cd Fiducio
uv sync --locked --extra docs --group build
```

## Running the checks

```bash
uv run --no-sync ruff check .
uv run --no-sync mypy
uv run --no-sync pytest
uv run --no-sync mkdocs build --strict
uv build --no-build-isolation
uv run --no-sync twine check dist/*
```

`uv sync` creates `.venv` and installs the project plus the default `dev` group.
The command above also installs documentation extras and locked build tools.
`--locked` rejects an out-of-date lockfile. After syncing, `--no-sync` keeps
commands from changing that explicitly selected environment.

The repository selects CPU PyTorch on Linux/Windows and PyPI PyTorch on macOS.
This is a development/CI choice, not a restriction on the installed library:
wheel metadata still requires only `torch>=2.10`. GPU applications can install
Fiducio into their own environment with their compatible PyTorch build.

Commit `uv.lock` whenever dependencies change. Run `uv lock` after editing
dependency declarations; use `uv lock --upgrade-package PACKAGE` for a deliberate
update, then sync and repeat the checks. Do not hand-edit the lockfile.
The lock identifies compatible versions per Python/platform; it does not imply
every platform uses the same package versions or reproduces a paper experiment.

These checks should pass before opening a pull request. New behaviour should come
with tests, and public API changes should be reflected in the documentation
under `docs/` and in `CHANGELOG.md`.

## Adding a calibrator

Calibrators subclass `fiducio.Calibrator` (or a shared helper base) and must:

1. implement the calibration math on flattened `(N, C)` tensors so the same code
   works for 2D, 3D and general n-D inputs;
2. register a **stable** `calibrator_id` via `@register_calibrator(...)` so that
   `save` / `load_calibrator` round-trips work;
3. implement `get_config`, `_get_state` and `_set_state` for persistence;
4. ship type annotations and Google-style docstrings;
5. come with tests covering 2D/3D, logits/probabilities, masks, `ignore_index`,
   save/load and any invariants the calibrator claims (e.g. argmax preservation).

## Conventions

- Public API is fully type-annotated; `py.typed` ships with the package.
- No `print` statements in library code — use the `logging` module.
- Keep heavy or domain-specific dependencies optional and imported lazily.
- Tensor layout is channel-first with the class axis at dimension 1:
  `(B, C, *spatial)` for predictions and `(B, *spatial)` for labels.

## Code of conduct

Be respectful and constructive. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/).

Release procedure: [docs/releasing.md](docs/releasing.md).
