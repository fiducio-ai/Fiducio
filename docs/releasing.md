# Releasing Fiducio

Releases are manual and gated. No workflow uploads to PyPI on its own.

## What the workflows enforce

- `validate.yml` (reusable) builds **one** wheel/sdist pair and runs every check
  against exactly those archives:
    - Twine and a content check (`scripts/check_distribution.py`): license,
      `py.typed`, identical sources in the wheel and sdist, the project page shipped
      in the sdist and absent from the wheel;
    - ruff and mypy on Python 3.10, 3.11 and 3.12;
    - the full test suite and the examples against the *installed wheel*, on
      Python 3.10–3.12 with the locked PyTorch, plus one run with the minimum
      supported PyTorch 2.10;
    - `mkdocs build --strict` from the checkout and from the unpacked sdist,
      the project-page check (`scripts/check_site.py`), and a wheel rebuilt from
      the sdist that must match the tested wheel.
- `tests.yml` runs it on pull requests and on `main`; `package.yml` reruns it manually.
- `publish-pypi.yml` (manual, with `confirm: publish`) first runs
  `scripts/check_release.py`. It fails unless the run was started from a
  `vX.Y.Z` tag equal to `project.version` and the tagged commit is on `main`.
  It then runs the validation and uploads the validated artifacts with
  Trusted Publishing and attestations, once the `pypi` environment has been approved.
- The OIDC `id-token: write` permission is granted only to the PyPI publish job
  and the Pages deploy job. Third-party actions are pinned by commit SHA.

## Procedure

1. Merge the release PR once `Tests` is green. Update `CHANGELOG.md`, the version
   in `pyproject.toml` and `uv.lock` (`uv lock`) in that PR.
2. From an up-to-date `main`, create and push an annotated tag:
   `git tag -a vX.Y.Z -m "Fiducio X.Y.Z" && git push origin vX.Y.Z`.
3. Actions → *Publish to PyPI (manual)* → *Run workflow*. Pick the **tag** as the
   ref and type `publish`. Approve the `pypi` environment when validation has passed.
4. Download the published files from PyPI. Check their SHA-256 against the
   workflow artifact and verify the attestations (`pypi-attestations verify pypi`
   or the PyPI UI).
5. In a fresh environment: `pip install fiducio==X.Y.Z`, then fit, transform,
   save and reload a calibrator.
6. Publish the GitHub release for the tag with the changelog entry.

## Rollback

- Failure before the upload: stop. Nothing has been published. Fix, then retag
  only if the tag was never used for an upload.
- After an upload: never replace or delete published archives. Release a new
  patch version, and if needed yank the broken version on PyPI.
- Pages: revert only the offending commit on `main`. The docs workflow redeploys
  the previous site, and the existing documentation is preserved.

## Repository settings (not encoded in the repository)

These settings are applied by a maintainer in the GitHub UI.

| Setting | Before (2026-09-21) | After |
|---|---|---|
| `main` protection | none (no branch protection, no rulesets) | ruleset on `main`: block deletion and force-push, require a pull request, require the status check `Tests / validation / Release checks` |
| Tag protection | none | ruleset on `refs/tags/v*`: block deletion, update and non-fast-forward. Creation is limited to maintainers |
| `pypi` environment | required reviewers, no deployment-ref policy | keep required reviewers; deployment policy restricted to tags matching `v*.*.*` |
| `github-pages` environment | custom branch policy | unchanged (`main`) |
| Private vulnerability reporting | disabled | enabled (needed by `SECURITY.md`) |
| Immutable releases | disabled | enabled for new releases; existing `v0.1.0` is not modified |
| PyPI Trusted Publisher | configured for `publish-pypi.yml`, environment `pypi` | unchanged; check the workflow filename and environment still match |
