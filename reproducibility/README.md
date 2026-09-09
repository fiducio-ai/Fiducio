# Paper implementation and reproducibility

Fiducio contains reusable calibrators. The numerical references in this release
validate selected calibration maps and objectives; they do not reproduce the
paper's fitted models, hyperparameter searches or benchmark tables.

## Method mapping

| Paper method | Public class | Research reference |
| --- | --- | --- |
| TS | `TemperatureScaling` | `calibrators/temperature.py:TemperatureScaling` |
| MS | `MatrixScaling` | `calibrators/matrix.py:MatrixScaling` |
| MSc | `TranslationInvariantMatrixScaling` | `MatrixScaling(enforce_row_sum_invariance=True)` |
| CDC | `ClassConditionalMatrixScaling` | `ClassConditionalDirichletCalibrator` |
| CMSap | `ArgmaxPreservingMatrixScaling` | `LogitRegularizedArgmaxPreservingCCDirichletCalibrator` |
| CMSop | `OrderPreservingMatrixScaling` | `LogitRegularizedOrderPreservingCCDirichletCalibrator` |
| LTS | Not included in this release | Separate research implementation; not reproduced here |

The conditional reference classes are in `calibrators/order_preserving_dirichlet.py`.
CMS uses joint optimization by default. `independent_experts=True` is a separate
optimization choice, not an interchangeable setting for paper results.

MSc learns the first `C-1` columns and a common row sum, reconstructs the final
column, and regularizes the resulting matrix. Older pre-release builds centered
each row to zero. Their saved predictions remain readable, but refitting now
uses the research parameterization; regularized fits can therefore change.

## Numerical regression

`tests/fixtures/research_reference.json` contains only synthetic tensors and
reference outputs. It records the research HEAD, whether its checkout was dirty,
the SHA-256 hashes of the imported source files, and the PyTorch version.
The hashes, not HEAD alone, identify the source used to generate the references.

The standard test suite checks MS, MSc, CDC, CMSap and CMSop for two class counts
and two seeds: mapped logits, regularization, joint unweighted cross-entropy,
parameter gradients and outputs after one Adam step. Matrix conventions are
mapped explicitly. The MS/MSc penalty is a documented formula because the
research implementation defines it inside `fit`; conditional penalties call
the research implementation directly. TS, ETS, VS and DC use the ordinary unit
tests and do not yet have research-generated numerical references.

To regenerate into a **new** file using a separately available research checkout:

```bash
python reproducibility/generate_reference.py \
  --research-root /path/to/research/Fiducio \
  --output /path/to/new-reference.json
```

The generator loads only calibration source modules, not datasets, checkpoints
or training pipelines. Run it only against a trusted checkout. Review differences
before replacing the checked-in reference. Standard CI needs no research checkout.

## Inputs, fitting and guarantees

- Models supply `(B, C, *spatial)` logits or probabilities; `(N, C)` voxel tables
  are also accepted. Classes always occupy dimension 1.
- Pool ensemble members consistently before calibration. Mean probabilities and
  mean logits are distinct inputs; see `examples/ensemble_pooling.py`.
- Fit only on a held-out calibration set. Never fit or choose hyperparameters
  using the test cases.
- `fit` materializes valid voxels in memory and performs full-batch optimization.
  For large 3D sets, prepare a reproducibly sampled `(N, C)` voxel table, recording
  seed, sampling unit and class weighting. Sampling changes the fitting objective.
  `transform` can be called on separate cases or chunks with the same fitted model.
- Preservation claims concern valid, unmasked voxels and resolved class orders.
  CMSap uses the input argmax to route experts; CMSop uses sorted ranks. Exact ties
  follow PyTorch's selection/sorting behavior, and finite precision can collapse
  very small probability differences. Do not promise preservation of all tie sets.
- The public full-batch optimizer does not reproduce the research case batching,
  schedules, clipping, class weighting, early stopping or hyperparameter selection.

## Before linking the camera-ready version

1. Identify the final manuscript and freeze the precise method configurations,
   splits, pooling, random seeds, optimization protocol and metric conventions.
2. Record which research source and checkpoints generated each reported result.
   Use dataset-pooled versus case-averaged metrics exactly as specified in that
   manuscript; the included generic metrics are not a benchmark reproduction script.
3. Run CI on the final commit, including the installed wheel. Review source
   distributions, notebooks and Git history before making the repository public.
4. Configure GitHub Pages with GitHub Actions as its source, then set repository
   variable `PAGES_ENABLED=true`. Documentation builds remain checked while
   deployment is disabled. A skipped deployment is not a published site.
5. Publish an approved tagged release and link that version from the manuscript.
   Update `CITATION.cff` with the final bibliographic information. PyPI is optional.

Items 1–2 require the final paper's experiment evidence. No dataset-specific
configurations or historical run provenance are inferred from library defaults.
