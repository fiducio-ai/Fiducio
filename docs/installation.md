# Installation

Fiducio requires **Python ≥ 3.10** and **PyTorch ≥ 2.0**.

## From PyPI

```bash
pip install fiducio
```

The core install depends only on `numpy` and `torch`.

## From GitHub

Until the first PyPI release you can install directly from the repository:

```bash
pip install "git+https://github.com/fiducio-ai/Fiducio.git"
```

## Optional extras

Heavy or domain-specific dependencies are kept out of the core install and
loaded only when you ask for them:

| Extra | Installs | For |
|-------|----------|-----|
| `fiducio[plots]` | `matplotlib` | reliability-diagram plots |
| `fiducio[docs]` | MkDocs Material + mkdocstrings | building the docs |
| `fiducio[dev]` | pytest, ruff, mypy, build, matplotlib | development |
| `fiducio[all]` | all of the above | everything |

```bash
pip install "fiducio[plots]"
```

A minimal install can `import fiducio` and use every calibrator and metric
without matplotlib present; matplotlib is only needed for the optional
plotting helpers.

## Verifying the install

```python
import fiducio
print(fiducio.__version__)
print(fiducio.registered_ids())
```
