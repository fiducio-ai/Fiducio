"""Binary isotonic calibration to labels or mean human responses."""

from __future__ import annotations

import operator
from typing import Any

import torch

from ..base import Calibrator, DeviceLike, NotFittedError
from ..registry import register_calibrator
from ..utils import (
    apply_mask_to_probabilities,
    class_last_flatten,
    restore_class_first,
    validate_predictions,
)


@register_calibrator("binary_isotonic_calibration")
class BinaryIsotonicCalibration(Calibrator):
    """Monotone binary calibration by equal-mass bins and weighted PAVA.

    Fits class 1 against hard labels or soft distributions, including the mean
    human response (MHR). Inspired by Kirscher et al., *TwinTrack: Post-hoc
    Multi-Rater Calibration for Medical Image Segmentation*, MIDL 2026 (short
    paper), https://arxiv.org/abs/2604.15950v2.

    Equal scores are never split across bins. Approximate equal-mass binning,
    linear interpolation between fitted points and constant extrapolation are
    explicit Fiducio conventions, not a claim of exact TwinTrack reproduction.
    Weighted PAVA minimizes squared error on the bins. The map is nondecreasing
    across voxels and may have plateaus or change a segmentation at threshold 0.5.

    Parameters
    ----------
    n_bins:
        Maximum number of fitting bins (default 250). Ties can reduce this count.
    input_type, ignore_index, device:
        See :class:`fiducio.Calibrator`. Only two classes are supported.

    Notes
    -----
    Fit sorts all valid voxels: O(N log N) time and O(N) memory, not streaming.
    No optimizer or validation-based early stopping is used. Learned coordinates
    are stored in float64; public probabilities are float32 and preserve 0/1.
    ``decision_function`` returns log-probabilities, including -inf at zeros.
    """

    def __init__(
        self,
        *,
        n_bins: int = 250,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        super().__init__(input_type=input_type, ignore_index=ignore_index, device=device)
        try:
            bins = 0 if isinstance(n_bins, bool) else operator.index(n_bins)
        except TypeError:
            bins = 0
        if bins < 1:
            raise ValueError("n_bins must be an integer >= 1")
        self.n_bins = bins
        self.x: torch.Tensor | None = None
        self.y: torch.Tensor | None = None

    def _refresh_stopping(self) -> None:
        if self.patience is not None or self.lr_patience is not None:
            raise ValueError("binary isotonic calibration does not support early stopping")

    def _to_canonical(self, predictions: torch.Tensor) -> torch.Tensor:
        p = predictions.float()
        return p if self.input_type == "probs" else p.softmax(dim=1)

    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        if num_classes != 2:
            raise ValueError("BinaryIsotonicCalibration requires exactly 2 classes")
        scores, order = z_flat[:, 1].double().sort()
        targets = (y_flat[:, 1] if y_flat.ndim == 2 else y_flat).double()[order]
        _, counts = torch.unique_consecutive(scores, return_counts=True)
        group_ends = counts.cumsum(0)
        n, k = scores.numel(), min(self.n_bins, scores.numel())
        ranks = (torch.arange(1, k, device=scores.device) * n + k - 1) // k
        ends = group_ends[torch.searchsorted(group_ends, ranks)]
        ends = torch.cat([ends, group_ends[-1:]]).unique_consecutive()
        starts = torch.cat([ends.new_zeros(1), ends[:-1]])
        weights = ends - starts
        score_sum = torch.cat([scores.new_zeros(1), scores.cumsum(0)])
        target_sum = torch.cat([targets.new_zeros(1), targets.cumsum(0)])
        x = (score_sum[ends] - score_sum[starts]) / weights
        means = (target_sum[ends] - target_sum[starts]) / weights
        # PAVA touches at most n_bins points; CPU lists avoid a GPU sync per merge.
        blocks: list[tuple[float, int, int]] = []
        for value, weight in zip(means.cpu().tolist(), weights.cpu().tolist(), strict=True):
            blocks.append((value, weight, 1))
            while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                v1, w1, size1 = blocks.pop()
                v0, w0, size0 = blocks.pop()
                blocks.append(((v0 * w0 + v1 * w1) / (w0 + w1), w0 + w1, size0 + size1))
        fitted = [value for value, _, size in blocks for _ in range(size)]
        self.x = x
        self.y = torch.tensor(fitted, dtype=torch.float64, device=self.device).clamp_(0, 1)

    def _apply(self, canonical: torch.Tensor) -> torch.Tensor:
        assert self.x is not None and self.y is not None
        flat, shape = class_last_flatten(canonical)
        scores = flat[:, 1].double().contiguous()
        if self.x.numel() == 1:
            positive = self.y[0].expand_as(scores)
        else:
            right = torch.searchsorted(self.x, scores).clamp(1, self.x.numel() - 1)
            left = right - 1
            fraction = ((scores - self.x[left]) / (self.x[right] - self.x[left])).clamp(0, 1)
            positive = torch.lerp(self.y[left], self.y[right], fraction)
        positive = positive.float()
        return restore_class_first(torch.stack([1 - positive, positive], dim=1), shape)

    @torch.no_grad()
    def transform(self, predictions: Any, mask: Any | None = None) -> torch.Tensor:
        """Return calibrated probabilities, preserving exact zeros and ones."""
        if not self._fitted:
            raise NotFittedError("call fit() before transform()")
        preds, _, msk = self._prepare(predictions, None, mask, with_targets=False)
        validate_predictions(preds, input_type=self.input_type, expected_num_classes=2, mask=msk)
        # Neutralize invalid padding before search/interpolation, not by multiplication.
        if msk is not None:
            preds = preds.masked_fill(~msk.unsqueeze(1), 0)
        out = self._apply(self._to_canonical(preds))
        return apply_mask_to_probabilities(out, msk)

    def decision_function(self, predictions: Any) -> torch.Tensor:
        """Return log-probabilities; zero probabilities map to mathematical -inf."""
        return self.transform(predictions).log()

    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        return self._apply(canonical).log()

    def _constructor_config(self) -> dict[str, Any]:
        return {"n_bins": self.n_bins}

    def _get_state(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}

    def _set_state(self, state: dict[str, Any]) -> None:
        self.x = None if state["x"] is None else state["x"].to(self.device, dtype=torch.float64)
        self.y = None if state["y"] is None else state["y"].to(self.device, dtype=torch.float64)
