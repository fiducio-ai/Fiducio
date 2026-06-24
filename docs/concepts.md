# Concepts

## Post-hoc calibration

A trained segmentation model outputs a probability distribution over classes for
each voxel. **Calibration** asks whether those probabilities are trustworthy: of
all voxels predicted with confidence 0.9, are about 90% actually correct?

*Post-hoc* calibration leaves the trained model untouched. It fits a small
transform `g` on top of the frozen model so that `g(model output)` is
better-calibrated. Because the segmentation map (the argmax) is often unchanged —
and is *guaranteed* unchanged by some calibrators — you can calibrate a deployed
model without affecting its Dice/IoU.

## You need a separate, labelled calibration set

Calibration must be fit on data the model did **not** train on, with ground-truth
labels. Fitting on the training set gives over-optimistic, useless calibration,
because the model is already over-confident there. The usual recipe:

1. Train the model on the training split.
2. Hold out a **calibration split** with labels (a slice of validation data).
3. Collect the model's logits and the labels on that split.
4. `fit` a Fiducio calibrator on them.
5. Evaluate calibration on a separate **test split**.

A few dozen labelled volumes are often enough for the simpler calibrators.

## Logits versus probabilities

- **Logits** are the raw, unnormalised scores before `softmax`. Pass
  `input_type="logits"` (the default).
- **Probabilities** sum to 1 along the class axis. Pass `input_type="probs"`.

Most calibrators are mathematically defined on logits; Fiducio converts
internally as needed, so you only have to declare what you pass in. If you have a
choice, prefer logits — they carry strictly more information than probabilities.

## Tensor convention

Fiducio is channel-first with the **class axis fixed at dimension 1**:

- `predictions`: `(B, C, *spatial)`
- `targets`: `(B, *spatial)`, integer class indices
- `mask` (optional): `(B, *spatial)`, `True` = valid

The class axis is never guessed from the shape. 2D, 3D and general n-D inputs are
all handled by the same code, which flattens every spatial position into a
`(N, C)` table internally.

## Binary versus multiclass

Multiclass segmentation with `C` classes is the default case. **Binary**
(foreground/background) segmentation should be expressed as **two channels**
(`C = 2`) — the same softmax machinery then applies. If your model produces a
single-channel sigmoid output, convert it first with
[`fiducio.two_channel_from_binary`](api/metrics.md):

```python
from fiducio import two_channel_from_binary
two_channel = two_channel_from_binary(sigmoid_logits, input_type="logits")  # (B, 2, *)
```

## Masks and `ignore_index`

Two complementary ways to exclude voxels from fitting:

- **`mask`** — a boolean tensor; only `True` positions are used.
- **`ignore_index`** — any label equal to this value is dropped (default
  `-100`, matching PyTorch's cross-entropy convention).

In `transform`, a `mask` zeroes out masked positions in the output so they carry
no probability mass.

## Device and dtype

- Computation runs on the calibrator's `device` (`cpu`/`cuda`); `None` selects
  CUDA when available.
- Inputs are converted to `float32` internally; labels to `int64`.
- Saved calibrators **load on CPU by default**; pass `map_location="cuda"` to
  `load_calibrator` to load onto a GPU. The device used when saving is not forced.

## Optimizer

Every calibrator is fitted by gradient descent and exposes an `optimizer`
argument:

- `optimizer="adam"` (default) — Adam, matching the common reference
  implementations; robust and the safe choice.
- `optimizer="lbfgs"` — L-BFGS with strong-Wolfe line search; fast and
  deterministic for the convex scaling objectives (temperature, vector, matrix,
  Dirichlet).

`lr` and `max_iter` default to per-optimizer values (Adam: `lr=0.1`,
`max_iter=200`; L-BFGS: `lr=1.0`, `max_iter=100`; the class-conditional
calibrators keep `lr=0.01` for Adam), and can be overridden explicitly.

## Calibrated probabilities vs logits

`transform` (and its alias `predict_proba`) return calibrated **probabilities**.
If you need the calibrated **logits** instead — for example to feed another loss
— use `decision_function`, which returns pre-softmax scores of the same shape;
`softmax(decision_function(x), dim=1)` equals `transform(x)`. Both run under
`torch.no_grad()` and never build an autograd graph.

## Memory and large volumes

Fitting loads **all valid voxels** of the calibration set into a single
`(N, C)` tensor in memory. This is fast and simple for typical calibration sets
(a few dozen volumes), but for very large 3D datasets the flattened tensor can
become large. If you hit memory limits, fit on a representative subset of cases
or crop to a region of interest with a `mask`; calibration parameters are low
dimensional and rarely need the full dataset. (Streaming/batched fitting is on
the roadmap.) `transform` itself is applied volume by volume and is not memory
bound in the same way.

## Behaviour before `fit`

Calling `transform` before `fit` raises `NotFittedError`. After fitting, passing
predictions with a different number of classes than seen at fit time raises a
`ValueError`.

## Integrating with U-Net, nnU-Net and SegFormer

Fiducio is deliberately model-agnostic. The integration is always the same:

1. Run inference with your model to obtain **logits** of shape `(B, C, *spatial)`.
2. Gather the matching **labels** for a calibration split.
3. `fit` a calibrator and `save` it.
4. At deployment, apply `transform` to fresh logits.

For **U-Net** the logits are the network output before softmax. For **nnU-Net**
they are the per-voxel class scores (you can calibrate before the usual argmax).
For **SegFormer** they are the segmentation head logits, upsampled to the label
resolution. In every case Fiducio only sees `(B, C, *spatial)` tensors and never
needs to know the architecture.
