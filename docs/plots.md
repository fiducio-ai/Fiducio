# Plotting

Fiducio ships an optional reliability-diagram helper. It needs matplotlib, which
is part of the `plots` extra:

```bash
pip install "fiducio[plots]"
```

Matplotlib is imported lazily, so `import fiducio` never requires it — only
calling a plotting function does.

## Reliability diagram

A reliability diagram bins predictions by top-1 confidence and compares the mean
confidence with the observed accuracy in each bin. A perfectly calibrated model
lies on the diagonal; bars below the diagonal indicate over-confidence.

```python
import matplotlib.pyplot as plt
import torch
from fiducio import TemperatureScaling
from fiducio.plots import reliability_diagram

logits = torch.randn(8, 4, 64, 64) * 4
labels = logits.argmax(dim=1)

calibrator = TemperatureScaling().fit(logits, labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
reliability_diagram(torch.softmax(logits, dim=1), labels, ax=ax1, title="Before")
reliability_diagram(calibrator.transform(logits), labels, ax=ax2, title="After")
fig.tight_layout()
fig.savefig("reliability.png", dpi=150)
```

`reliability_diagram` returns the matplotlib `Axes`, so you can compose it into
larger figures. It accepts the same `mask` / `ignore_index` arguments as the
metrics.

## Custom plots

If you want to build your own figure, [`fiducio.reliability_curve`](api/metrics.md)
returns the raw per-bin statistics (`bin_edges`, `bin_confidence`,
`bin_accuracy`, `bin_counts`, `ece`) as tensors.
