# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-21

First public beta release. It accompanies the paper *Rethinking Post-Hoc
Calibration in Semantic Segmentation* (Transactions on Machine Learning Research).

### Added

- Model-agnostic `Calibrator` API with a consistent
  `fit` / `transform` / `predict_proba` / `fit_transform` / `decision_function` /
  `save` interface and a top-level `load_calibrator` loader. `transform` and
  `decision_function` run under `torch.no_grad()`.
- Support for 2D `(B, C, H, W)`, 3D `(B, C, D, H, W)` and general
  `(B, C, *spatial)` inputs, with logits **or** probabilities, masks and
  `ignore_index`.
- Calibrators, with the paper's shorthand as aliases:
  - `TemperatureScaling` (`TS`), `EnsembleTemperatureScaling` (`ETS`),
    `VectorScaling` (`VS`), `MatrixScaling` (`MS`), `DirichletCalibration` (`DC`);
  - `TranslationInvariantMatrixScaling` (`MSc`): optimizes `C-1` free columns and
    a common row sum, reconstructs the last column and regularizes the
    reconstructed matrix;
  - `ClassConditionalMatrixScaling` (`CMS`, `CDC`), `ArgmaxPreservingMatrixScaling`
    (`CMSAP`, `CMSap`) and `OrderPreservingMatrixScaling` (`CMSOP`, `CMSop`):
    one affine map per uncalibrated top class. All experts are optimized
    **jointly** by default (one optimizer, one cross-entropy loss over every
    voxel, regularization averaged across experts in the common logit space, as
    in the paper); `independent_experts=True` fits each expert in its own loop.
- Configurable optimizer on every calibrator: `optimizer="adam"` (default) or
  `optimizer="lbfgs"`, with per-optimizer `lr` / `max_iter` defaults; a
  non-positive `max_iter` is rejected.
- Validation-based early stopping (`patience`, `min_delta`, `lr_patience`,
  `lr_factor`; Adam only) with `fit(..., val_predictions=, val_targets=,
  val_mask=)`: the iterate with the best validation NLL is kept, and the
  learning rate can decay on plateaus.
- Calibration metrics: negative log-likelihood, expected calibration error
  (ECE), average calibration error (ACE, unweighted over non-empty bins),
  multiclass Brier score and `reliability_curve` / `ReliabilityCurve`.
- Optional plotting: `fiducio.plots.reliability_diagram` (matplotlib,
  lazy-imported, `[plots]` extra; `metric="ece"|"ace"|None`).
- Helpers: `two_channel_from_binary` (single-channel sigmoid → two channels) and
  `get_calibrator_class` (build a calibrator from its registered id).
- Registry-backed persistence (`save` / `load_calibrator`) that loads on CPU by
  default, accepts `map_location`, avoids importing arbitrary module paths, and
  warns on a major-version mismatch.
- Synthetic numerical references for MS, MSc, CDC, CMSap and CMSop against the
  research implementation, an ensemble-pooling example and a paper
  implementation guide (`reproducibility/`).
- Documentation (MkDocs Material), runnable CPU examples, a README with a
  reliability figure (`examples/readme_figure.py`), locked uv development
  environments and a CI matrix for Python 3.10, 3.11 and 3.12.

[Unreleased]: https://github.com/fiducio-ai/Fiducio/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fiducio-ai/Fiducio/releases/tag/v0.1.0
