"""Base class and shared machinery for all Fiducio calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

from .utils import (
    apply_mask_to_probabilities,
    flatten_valid,
    get_logger,
    resolve_device,
    safe_log,
    to_tensor,
    validate_predictions,
    validate_targets,
)

DeviceLike = str | torch.device


class NotFittedError(RuntimeError):
    """Raised when ``transform`` is called before ``fit``."""


class Calibrator(ABC):
    """Abstract base class for post-hoc segmentation calibrators.

    All calibrators share the same interface and tensor convention:

    * **predictions** are channel-first, class axis at dimension 1
      (``(B, C, *spatial)``; ``(N, C)`` is also accepted);
    * **targets** are integer labels of shape ``(B, *spatial)``;
    * **mask**, when given, is ``(B, *spatial)`` with ``True`` for valid voxels.

    Inputs may be raw ``logits`` or normalised ``probs`` (see ``input_type``).
    ``transform`` and ``predict_proba`` always return calibrated *probabilities*
    of the same shape as the input.

    Parameters
    ----------
    input_type:
        ``"logits"`` (default) if predictions are unnormalised scores, or
        ``"probs"`` if they are probabilities that sum to 1 along the class axis.
    ignore_index:
        Label value excluded from fitting (default ``-100``).
    device:
        Computation device. ``None`` selects CUDA when available, else CPU.
    """

    #: Stable id assigned by :func:`fiducio.registry.register_calibrator`.
    calibrator_id: str = ""
    #: Representation the calibration math consumes: ``"logits"`` or ``"logprobs"``.
    _input_space: str = "logits"

    def __init__(
        self,
        *,
        input_type: str = "logits",
        ignore_index: int = -100,
        device: DeviceLike | None = None,
    ) -> None:
        if input_type not in {"logits", "probs"}:
            raise ValueError(f"input_type must be 'logits' or 'probs', got {input_type!r}")
        self.input_type = input_type
        self.ignore_index = int(ignore_index)
        self.device = resolve_device(device)
        self._num_classes: int | None = None
        self._fitted: bool = False
        self._logger = get_logger(type(self).__name__)

    # ------------------------------------------------------------------ public

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    @property
    def num_classes(self) -> int | None:
        """Number of classes seen at fit time, or ``None`` before fitting."""
        return self._num_classes

    def fit(
        self,
        predictions: Any,
        targets: Any,
        mask: Any | None = None,
    ) -> Calibrator:
        """Fit the calibrator on a labelled calibration set.

        Parameters
        ----------
        predictions:
            ``(B, C, *spatial)`` logits or probabilities (see ``input_type``).
        targets:
            ``(B, *spatial)`` integer labels.
        mask:
            Optional ``(B, *spatial)`` boolean mask; ``True`` marks valid voxels.
        """
        preds, tgts, msk = self._prepare(predictions, targets, mask, with_targets=True)
        assert tgts is not None  # guaranteed by with_targets=True
        num_classes = validate_predictions(preds, input_type=self.input_type)
        validate_targets(
            preds, tgts, msk, num_classes=num_classes, ignore_index=self.ignore_index
        )
        self._num_classes = num_classes
        canonical = self._to_canonical(preds)
        z_flat, y_flat = flatten_valid(canonical, tgts, msk, self.ignore_index)
        if z_flat.shape[0] == 0:
            raise ValueError(
                "no valid voxels to fit on (all positions are masked out or equal "
                "ignore_index)"
            )
        self._fit_core(z_flat, y_flat, num_classes)
        self._fitted = True
        return self

    def transform(self, predictions: Any, mask: Any | None = None) -> torch.Tensor:
        """Apply calibration and return probabilities of the input shape.

        Parameters
        ----------
        predictions:
            ``(B, C, *spatial)`` logits or probabilities.
        mask:
            Optional ``(B, *spatial)`` boolean mask. Masked-out voxels are set to
            0 across all classes in the output.
        """
        if not self._fitted:
            raise NotFittedError("call fit() before transform()")
        preds, _, msk = self._prepare(predictions, None, mask, with_targets=False)
        validate_predictions(
            preds, input_type=self.input_type, expected_num_classes=self._num_classes
        )
        canonical = self._to_canonical(preds)
        calibrated_logits = self._map_logits(canonical)
        probs = F.softmax(calibrated_logits, dim=1)
        return apply_mask_to_probabilities(probs, msk)

    def predict_proba(self, predictions: Any, mask: Any | None = None) -> torch.Tensor:
        """Alias for :meth:`transform`; returns calibrated probabilities."""
        return self.transform(predictions, mask=mask)

    def fit_transform(
        self,
        predictions: Any,
        targets: Any,
        mask: Any | None = None,
    ) -> torch.Tensor:
        """Fit on the calibration set, then transform the same predictions."""
        self.fit(predictions, targets, mask=mask)
        return self.transform(predictions, mask=mask)

    def save(self, path: Any) -> None:
        """Save this calibrator to ``path`` (see :func:`fiducio.save_calibrator`)."""
        from .persistence import save_calibrator

        save_calibrator(self, path)

    def get_config(self) -> dict[str, Any]:
        """Return constructor keyword arguments (excluding ``device``)."""
        config: dict[str, Any] = {
            "input_type": self.input_type,
            "ignore_index": self.ignore_index,
        }
        config.update(self._constructor_config())
        return config

    # ------------------------------------------------------------- subclass API

    @abstractmethod
    def _fit_core(self, z_flat: torch.Tensor, y_flat: torch.Tensor, num_classes: int) -> None:
        """Fit learned parameters from flattened ``(N, C)`` data."""

    @abstractmethod
    def _map_logits(self, canonical: torch.Tensor) -> torch.Tensor:
        """Map canonical ``(B, C, *spatial)`` scores to calibrated logits."""

    @abstractmethod
    def _get_state(self) -> dict[str, Any]:
        """Return learned parameters as tensors/plain types for persistence."""

    @abstractmethod
    def _set_state(self, state: dict[str, Any]) -> None:
        """Load learned parameters produced by :meth:`_get_state`."""

    def _constructor_config(self) -> dict[str, Any]:
        """Return subclass-specific constructor kwargs. Override as needed."""
        return {}

    # --------------------------------------------------------------- internals

    def _prepare(
        self,
        predictions: Any,
        targets: Any | None,
        mask: Any | None,
        *,
        with_targets: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        preds = to_tensor(predictions, dtype=torch.float32, device=self.device)
        tgts: torch.Tensor | None = None
        if with_targets:
            if targets is None:
                raise ValueError("targets are required for fit()")
            tgts = to_tensor(targets, device=self.device).long()
        msk: torch.Tensor | None = None
        if mask is not None:
            msk = to_tensor(mask, device=self.device).to(torch.bool)
        return preds, tgts, msk

    def _to_canonical(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert validated predictions to the calibrator's canonical space."""
        if self._input_space == "logits":
            if self.input_type == "logits":
                return predictions
            return safe_log(predictions)
        # log-probability space
        if self.input_type == "probs":
            return safe_log(predictions)
        return F.log_softmax(predictions, dim=1)
