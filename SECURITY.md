# Security Policy

## Supported versions

Fiducio is in beta (`0.x`). Security fixes are applied to the latest released
version on the `main` branch.

## Reporting a vulnerability

Please report suspected vulnerabilities privately using GitHub's
["Report a vulnerability"](https://github.com/fiducio-ai/Fiducio/security/advisories/new)
workflow rather than opening a public issue. We aim to acknowledge reports
within a few business days.

## Loading saved calibrators

`fiducio.load_calibrator` reads files written by `Calibrator.save`. These files
are deserialized with PyTorch (`torch.load`). By default Fiducio loads with
`weights_only=True`, which restricts deserialization to plain tensors and basic
Python types and does **not** execute arbitrary code. Even so:

- Only load calibrator files from sources you trust.
- The on-disk format records a stable calibrator id that is resolved through an
  internal, controlled registry. Fiducio never imports an arbitrary module path
  stored inside a file.

If you must load a legacy file that requires `weights_only=False`, do so only for
trusted files and be aware that pickle-based loading can execute arbitrary code.
