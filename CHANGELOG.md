# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
