# Security Policy

## Supported versions

Fiducio is in beta (`0.x`). Security fixes are applied to the latest released
version on the `main` branch.

## Reporting a vulnerability

Please report suspected vulnerabilities privately using GitHub's
["Report a vulnerability"](https://github.com/fiducio-ai/Fiducio/security/advisories/new)
workflow (GitHub private vulnerability reporting) rather than opening a public
issue. We aim to acknowledge reports within a few business days.

## Loading saved calibrators

`fiducio.load_calibrator` reads files written by `Calibrator.save`. These files
are deserialized with PyTorch (`torch.load`). Fiducio always loads with
`weights_only=True`, which restricts deserialization to tensors and basic Python types. This is
defense in depth, not a guarantee that untrusted files are safe. Fiducio requires
PyTorch >= 2.10, which includes fixes for CVE-2025-32434 and CVE-2026-24747.
Keep PyTorch updated as new security fixes become available.

- Only load calibrator files from sources you trust.
- The on-disk format records a stable calibrator id that is resolved through an
  internal, controlled registry. Fiducio never imports an arbitrary module path
  stored inside a file.

Fiducio never retries with unrestricted pickle loading. Files with an unknown
format version or inconsistent parameters are rejected. Unsupported legacy
formats must be converted separately in a trusted environment.
