# Contributing to Fiducio

Thanks for your interest in improving Fiducio! This document describes how to set
up a development environment and the conventions we follow.

## Development setup

```bash
git clone https://github.com/fiducio-ai/Fiducio.git
cd Fiducio
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

```bash
ruff check .          # lint
mypy                  # type check (src/fiducio)
pytest                # tests
```

All three should pass before opening a pull request. New behaviour should come
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
