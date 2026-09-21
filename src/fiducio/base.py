"""Base class and shared machinery for all Fiducio calibrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
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
from .utils.stopping import StoppingRule, resolve_stopping
from .utils.tensors import integer_targets, validate_mask

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

    Notes
    -----
    Calibrators fitted by gradient descent (all of them) also accept, as
    keyword-only constructor arguments, an optional validation-based early
    stopping rule (Adam only):

    * ``patience`` -- stop after this many iterations without the validation
      NLL improving by more than ``min_delta`` (default ``0.0``);
    * ``lr_patience`` / ``lr_factor`` -- multiply the learning rate by
      ``lr_factor`` (default ``0.1``) after ``lr_patience`` iterations without
      improvement (``ReduceLROnPlateau``).

    Setting either requires ``val_predictions`` / ``val_targets`` in :meth:`fit`;
    the iterate with the best validation NLL is kept. With ``max_iter`` acting as
    an upper bound, this reproduces the "Adam + early stopping on validation
    NLL" recipe used in the paper.
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
        self._val: tuple[torch.Tensor, torch.Tensor] | None = None
        self._logger = get_logger(type(self).__name__)

    # ------------------------------------------------------------------ public

    #: early-stopping settings (see :meth:`_init_stopping`); disabled by default.
    patience: int | None = None
    min_delta: float = 0.0
    lr_patience: int | None = None
    lr_factor: float = 0.1
    _stopping: StoppingRule | None = None

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
        *,
        val_predictions: Any | None = None,
        val_targets: Any | None = None,
        val_mask: Any | None = None,
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
        val_predictions, val_targets, val_mask:
            Optional held-out validation set, in the same format as
            ``predictions`` / ``targets`` / ``mask``. Required when early
            stopping is configured (``patience`` or ``lr_patience``), and
            rejected otherwise. The validation NLL (without regularization) is
            monitored after every Adam step and the best iterate is kept.

        Notes
        -----
        Calling ``fit`` again on an already-fitted instance re-fits from
        scratch: all learned parameters are reinitialised and overwritten, and
        a new ``num_classes`` (which may differ from the previous fit) is
        recorded. No state from the previous fit is reused. If fitting fails,
        the previous state is preserved. Model outputs are detached from autograd.
        """
        preds, tgts, msk = self._prepare(predictions, targets, mask, with_targets=True)
        assert tgts is not None  # guaranteed by with_targets=True
        num_classes = validate_predictions(preds, input_type=self.input_type)
        validate_targets(
            preds, tgts, msk, num_classes=num_classes, ignore_index=self.ignore_index
        )
        canonical = self._to_canonical(preds)
        z_flat, y_flat = flatten_valid(canonical, tgts, msk, self.ignore_index)
        if z_flat.shape[0] == 0:
            raise ValueError(
                "no valid voxels to fit on (all positions are masked out or equal "
                "ignore_index)"
            )
        val_data = self._prepare_validation(
            val_predictions, val_targets, val_mask, num_classes=num_classes
        )
        candidate = deepcopy(self)
        candidate._num_classes = num_classes
        candidate._val = val_data
        try:
            candidate._fit_core(z_flat, y_flat, num_classes)
        finally:
            candidate._val = None
        candidate._fitted = True
        self.__dict__.update(candidate.__dict__)
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
        logits, msk = self._calibrated_logits(predictions, mask)
        probs = F.softmax(logits, dim=1)
        return apply_mask_to_probabilities(probs, msk)

    def predict_proba(self, predictions: Any, mask: Any | None = None) -> torch.Tensor:
        """Alias for :meth:`transform`; returns calibrated probabilities."""
        return self.transform(predictions, mask=mask)

    def decision_function(self, predictions: Any) -> torch.Tensor:
        """Return calibrated **logits** (pre-softmax) of the input shape.

        Unlike :meth:`transform`, no mask is applied — a logit of 0 is a
        meaningful value, so masking calibrated logits is left to the caller.
        ``softmax`` of the result along dimension 1 equals :meth:`transform`.
        """
        if not self._fitted:
            raise NotFittedError("call fit() before decision_function()")
        logits, _ = self._calibrated_logits(predictions, None)
        return logits

    def fit_transform(
        self,
        predictions: Any,
        targets: Any,
        mask: Any | None = None,
        *,
        val_predictions: Any | None = None,
        val_targets: Any | None = None,
        val_mask: Any | None = None,
    ) -> torch.Tensor:
        """Fit on the calibration set, then transform the same predictions."""
        self.fit(
            predictions,
            targets,
            mask=mask,
            val_predictions=val_predictions,
            val_targets=val_targets,
            val_mask=val_mask,
        )
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
        if self._stopping is not None:
            config.update(
                patience=self.patience,
                min_delta=self.min_delta,
                lr_patience=self.lr_patience,
                lr_factor=self.lr_factor,
            )
        return config

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "unfitted"
        return (
            f"{type(self).__name__}(input_type={self.input_type!r}, "
            f"{status}, device={self.device})"
        )

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

    def _init_stopping(
        self,
        optimizer: str,
        patience: int | None,
        min_delta: float,
        lr_patience: int | None,
        lr_factor: float,
    ) -> None:
        """Validate and store early-stopping settings (call after ``optimizer`` is set)."""
        self._stopping = resolve_stopping(optimizer, patience, min_delta, lr_patience, lr_factor)
        if self._stopping is not None:
            self.patience = self._stopping.patience
            self.min_delta = self._stopping.min_delta
            self.lr_patience = self._stopping.lr_patience
            self.lr_factor = self._stopping.lr_factor

    def _prepare_validation(
        self,
        val_predictions: Any | None,
        val_targets: Any | None,
        val_mask: Any | None,
        *,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Flatten the optional validation set into canonical ``(N, C)`` / ``(N,)`` tensors."""
        if val_predictions is None and val_targets is None:
            if val_mask is not None:
                raise ValueError("val_mask given without val_predictions and val_targets")
            if self._stopping is not None:
                raise ValueError(
                    "early stopping (patience / lr_patience) requires val_predictions "
                    "and val_targets in fit()"
                )
            return None
        if val_predictions is None or val_targets is None:
            raise ValueError("val_predictions and val_targets must be given together")
        if self._stopping is None:
            raise ValueError(
                "validation data was given but early stopping is not configured; set "
                "patience and/or lr_patience on the calibrator"
            )
        preds, tgts, msk = self._prepare(val_predictions, val_targets, val_mask, with_targets=True)
        assert tgts is not None
        val_classes = validate_predictions(preds, input_type=self.input_type)
        if val_classes != num_classes:
            raise ValueError(
                f"validation predictions have {val_classes} classes, expected {num_classes}"
            )
        validate_targets(preds, tgts, msk, num_classes=num_classes, ignore_index=self.ignore_index)
        z_val, y_val = flatten_valid(self._to_canonical(preds), tgts, msk, self.ignore_index)
        if z_val.shape[0] == 0:
            raise ValueError("no valid validation voxels (all masked out or equal ignore_index)")
        return z_val, y_val

    def _prepare(
        self,
        predictions: Any,
        targets: Any | None,
        mask: Any | None,
        *,
        with_targets: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        preds = to_tensor(predictions, dtype=torch.float32, device=self.device).detach()
        tgts: torch.Tensor | None = None
        if with_targets:
            if targets is None:
                raise ValueError("targets are required for fit()")
            tgts = integer_targets(targets, device=self.device)
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

    def _calibrated_logits(
        self, predictions: Any, mask: Any | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Validate inputs and return calibrated logits (no autograd graph)."""
        preds, _, msk = self._prepare(predictions, None, mask, with_targets=False)
        validate_predictions(
            preds, input_type=self.input_type, expected_num_classes=self._num_classes
        )
        validate_mask(preds, msk)
        with torch.no_grad():
            canonical = self._to_canonical(preds)
            logits = self._map_logits(canonical)
        return logits, msk
