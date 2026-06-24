---
name: Bug report
about: Report something that does not work as documented
title: "[bug] "
labels: bug
---

**Describe the bug**
A clear and concise description of what the bug is.

**To reproduce**
A minimal, self-contained snippet (synthetic tensors, no private data):

```python
import torch
from fiducio import TemperatureScaling
# ...
```

**Expected behaviour**
What you expected to happen.

**Environment**
- Fiducio version: (`python -c "import fiducio; print(fiducio.__version__)"`)
- PyTorch version:
- Python version:
- OS / device (CPU/CUDA):

**Additional context**
Tensor shapes, `input_type`, whether you used a mask / `ignore_index`, full traceback.
