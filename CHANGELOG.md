# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Locked uv development, documentation and build environments, a default
  Python 3.12 selection, and uv-based CI for Python 3.10–3.12. Repository
  tooling selects CPU PyTorch; published runtime dependencies remain portable.
- Synthetic research references for MS, MSc, CDC, CMSap and CMSop, testing
  mapped logits, regularization, gradients and one production Adam step.
- Ensemble pooling example and paper implementation/reproducibility guide.
- Paper-shorthand aliases for the class-conditional family: `CDC` (alias of
  `ClassConditionalMatrixScaling`), `CMSap` (`ArgmaxPreservingMatrixScaling`)
  and `CMSop` (`OrderPreservingMatrixScaling`), plus `MSc` for
  `TranslationInvariantMatrixScaling` and `DC` for `DirichletCalibration`.
- `independent_experts` constructor option on the class-conditional calibrators
  (CDC/CMSap/CMSop): `False` (default) jointly optimizes all experts in a
  single loss, matching the paper; `True` fits each expert in its own
  optimization loop on only the voxels routed to it.
- `average_calibration_error` (ACE): the same uniform confidence bins as
  `expected_calibration_error`, but the per-bin gap is averaged **unweighted**
  over non-empty bins instead of weighted by bin population, so a sparsely
  populated bin counts as much as a densely populated one.
  `reliability_curve`'s `ReliabilityCurve` now also exposes an `ace` field, and
  `fiducio.plots.reliability_diagram` accepts `metric="ece"|"ace"|None`
  (replacing the old `show_ece: bool` argument) to choose which one annotates
  the panel.

### Fixed

- MSc now uses the research common-row-sum parameterization and applies ODIR
  to the reconstructed matrix. This changes regularized fits compared with
  the pre-release zero-row-sum implementation; old saved predictions remain
  readable, while refitting uses the new parameterization.
- Mypy targets each CI interpreter instead of forcing Python 3.10 syntax on
  newer dependency stubs. Packaging requires a PEP 639-compatible setuptools.
- GitHub Pages deployment is opt-in through `PAGES_ENABLED=true`; strict
  documentation builds still run before Pages is configured.
- README preservation guarantees now depend on the selected calibrator.
- `max_iter` is now validated to be strictly positive on every calibrator; a
  non-positive value (e.g. `max_iter=0`) previously fit silently without
  raising or optimizing.
- `ClassConditionalMatrixScaling` / `ArgmaxPreservingMatrixScaling` /
  `OrderPreservingMatrixScaling` (CDC/CMSap/CMSop) now optimize all experts
  **jointly** (a single optimizer minimizing one cross-entropy loss over every
  voxel at once, regularization averaged across experts), matching the paper's
  method. They previously optimized each expert independently in its own
  optimization loop, which is a different training procedure.

### Removed

- `min_expert_voxels` constructor argument on the class-conditional calibrators
  (CDC/CMSap/CMSop): meaningless now that all experts are optimized jointly in
  a single loss.

## [0.1.0] - 2026-06-24

First public beta release.

### Added

- Model-agnostic `Calibrator` API with a consistent
  `fit` / `transform` / `predict_proba` / `fit_transform` / `decision_function` /
  `save` interface and a top-level `load_calibrator` loader. `transform` and
  `decision_function` run under `torch.no_grad()`.
- Support for 2D `(B, C, H, W)`, 3D `(B, C, D, H, W)` and general
  `(B, C, *spatial)` inputs, with logits **or** probabilities, masks and
  `ignore_index`.
- Calibrators:
  - `TemperatureScaling` (`TS`)
  - `EnsembleTemperatureScaling` (`ETS`)
  - `VectorScaling` (`VS`)
  - `MatrixScaling` (`MS`)
  - `TranslationInvariantMatrixScaling`
  - `DirichletCalibration`
  - `ClassConditionalMatrixScaling` (`CMS`)
  - `ArgmaxPreservingMatrixScaling` (`CMSAP`)
  - `OrderPreservingMatrixScaling` (`CMSOP`)
- Configurable optimizer on every calibrator: `optimizer="adam"` (default) or
  `optimizer="lbfgs"`, with sensible per-optimizer `lr` / `max_iter` defaults.
- Calibration metrics: negative log-likelihood, expected calibration error
  (ECE), multiclass Brier score and `reliability_curve` / `ReliabilityCurve`.
- Optional plotting: `fiducio.plots.reliability_diagram` (matplotlib,
  lazy-imported, `[plots]` extra).
- Helpers: `two_channel_from_binary` (single-channel sigmoid → two channels) and
  `get_calibrator_class` (build a calibrator from its registered id).
- Registry-backed persistence (`save` / `load_calibrator`) that loads on CPU by
  default, accepts `map_location`, avoids importing arbitrary module paths, and
  warns on a major-version mismatch.
- Documentation (MkDocs Material), runnable CPU examples and a CI matrix for
  Python 3.10, 3.11 and 3.12.

[Unreleased]: https://github.com/fiducio-ai/Fiducio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fiducio-ai/Fiducio/releases/tag/v0.1.0
