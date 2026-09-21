# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - Unreleased

Patch release. The public API (classes, functions and signatures) is unchanged.
Some invalid inputs that 0.1.0 accepted silently now raise `ValueError`.

### Changed

- **Minimum PyTorch is now 2.10** (was 2.0). Python support is unchanged (3.10–3.12).
- `ReliabilityCurve` statistics are float64 and `bin_counts` is int64, all on
  the predictions' device (previously float32 on CPU). ECE/ACE are still Python floats,
  the bin convention is unchanged, and metrics without valid voxels still return NaN.

### Fixed

- Adam fitting now records each loss against the state it was computed on, and
  treats the initial state as a candidate. It restores the true best state. `min_delta`
  now only drives the early-stopping patience counter.
- ECE/ACE accumulate in float64 with int64 counts, so they no longer drift on large
  voxel populations. Labels and masks are moved to the predictions' device.
- Stricter validation:
  - masks in `transform` and in the metrics must have the exact expected shape
    (no implicit broadcasting);
  - label layouts must be exact;
  - labels must be finite integers (integral floats are still accepted);
  - probabilities must be finite, within [0, 1] and sum to 1 (tolerance 1e-3).
- `fit` detaches model outputs from autograd and is transactional: a failed refit
  leaves the previously fitted state untouched.
- The inverse softplus used to initialise TS, ETS and class-conditional parameters
  is now numerically stable. Non-finite or out-of-range hyperparameters are rejected.
  Fits that produce a non-finite loss or non-finite parameters raise instead of
  returning a broken calibrator.

### Security

- `load_calibrator` always uses `torch.load(..., weights_only=True)` and never falls
  back to unrestricted unpickling. It validates the format version, calibrator id,
  configuration, class count, parameter shapes and finiteness, and rejects unknown
  versions and inconsistent files. Files written by the published 0.1.0 wheel
  (fitted and unfitted), and legacy full-matrix MSc checkpoints, still load; the
  tests check this against real 0.1.0 files.
- The PyTorch floor excludes versions affected by CVE-2025-32434 and CVE-2026-24747.
- Release hardening:
  - publishing requires a `vX.Y.Z` tag that matches the package version and
    points to a commit on `main`;
  - the tested wheel/sdist pair is the one uploaded, via Trusted Publishing with attestations;
  - workflow actions are pinned by commit SHA;
  - OIDC permissions are limited to the deploy jobs.

### Documentation and packaging

- Project page at <https://fiducio-ai.github.io/Fiducio/paper/>, linked from the
  README, the documentation and the PyPI metadata ("Project page"). Its assets and
  font licenses ship in the sdist, not in the wheel.
- The quickstart and the reliability-diagram example evaluate on data held out
  from fitting.
- CI runs:
  - the full suite and the examples against the installed wheel, including one
    run with PyTorch 2.10;
  - a rebuild of the documentation and the wheel from the sdist.
- Release procedure and required repository settings: `docs/releasing.md`.

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

[Unreleased]: https://github.com/fiducio-ai/Fiducio/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/fiducio-ai/Fiducio/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/fiducio-ai/Fiducio/releases/tag/v0.1.0
