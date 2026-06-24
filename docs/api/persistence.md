# API: Persistence

Fiducio saves a single `torch.save` payload containing format metadata, the
Fiducio version, a stable calibrator id, the constructor configuration and the
learned state. Loading resolves the id through an internal **controlled
registry** and never imports an arbitrary module path stored in the file. By
default files are read with `weights_only=True` and onto CPU.

!!! warning
    Only load calibrator files from sources you trust. See the project
    [security policy](https://github.com/fiducio-ai/Fiducio/blob/main/SECURITY.md).

::: fiducio.save_calibrator

::: fiducio.load_calibrator

::: fiducio.registered_ids
