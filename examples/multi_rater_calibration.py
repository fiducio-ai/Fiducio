"""Calibrate against empirical rater proportions on separate synthetic cases.

MHR/isotonic approach: Kirscher et al., TwinTrack, MIDL 2026 (short paper),
https://arxiv.org/abs/2604.15950v2. Fiducio also extends parametric fits to MHR.
Run from the repository: uv run python examples/multi_rater_calibration.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from fiducio import (
    BinaryIsotonicCalibration,
    TemperatureScaling,
    brier_score,
    expected_calibration_error,
    mean_human_response,
    negative_log_likelihood,
)
from fiducio.plots import reliability_diagram


def synthetic_cases() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate pooled model probabilities and five noisy votes per case."""
    g = torch.Generator().manual_seed(2026)
    latent = torch.randn(12, 1, 16, 16, generator=g)
    probability = latent.sigmoid()
    votes = (torch.rand(12, 5, 16, 16, generator=g) < probability).long()
    positive = (2.5 * latent[:, 0] + 0.4).sigmoid()
    return torch.stack([1 - positive, positive], dim=1), votes


def report(name: str, p: torch.Tensor, votes: torch.Tensor) -> None:
    q, annotated = mean_human_response(votes, num_classes=2)
    errors = []
    evaluated = 0
    for rater in range(votes.shape[1]):
        valid = votes[:, rater] != -100
        if bool(valid.any()):
            errors.append(
                expected_calibration_error(p, votes[:, rater], mask=valid, class_index=1, n_bins=50)
            )
            evaluated += int(valid.sum())
    mean_rater_ece = sum(errors) / len(errors) if errors else float("nan")
    print(
        f"{name:18s} NLL-MHR={negative_log_likelihood(p, q, mask=annotated):.4f} "
        f"Brier-MHR={brier_score(p, q, mask=annotated):.4f} "
        f"ECE-MHR={expected_calibration_error(p, q, mask=annotated, class_index=1, n_bins=50):.4f} "
        f"mean-rater-ECE={mean_rater_ece:.4f}"
    )
    print(
        f"  evaluable raters={len(errors)}/{votes.shape[1]}, votes={evaluated}, "
        f"missing={votes.numel() - evaluated}"
    )


def main() -> None:
    predictions, annotations = synthetic_cases()
    # Split cases first. Predictions represent an already fixed ensemble pooling and ROI.
    p_cal, p_val, p_test = predictions[:6], predictions[6:9], predictions[9:]
    a_cal, a_val, a_test = annotations[:6], annotations[6:9], annotations[9:]
    q_cal, mask_cal = mean_human_response(a_cal, num_classes=2)
    q_val, mask_val = mean_human_response(a_val, num_classes=2)
    q_test, mask_test = mean_human_response(a_test, num_classes=2)
    report("uncalibrated", p_test, a_test)
    calibrated = None
    for name, cls in [("TS", TemperatureScaling), ("isotonic", BinaryIsotonicCalibration)]:
        for target in ["single-rater", "MHR"]:
            cal = cls(input_type="probs", device="cpu")
            kwargs = {}
            if name == "TS":
                cal.patience = 15
                kwargs = dict(
                    val_predictions=p_val,
                    val_targets=q_val if target == "MHR" else a_val[:, 0],
                    val_mask=mask_val,
                )
            cal.fit(p_cal, q_cal if target == "MHR" else a_cal[:, 0], mask=mask_cal, **kwargs)
            result = cal.transform(p_test)
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result.sum(1), torch.ones_like(result[:, 0]))
            report(f"{name}/{target}", result, a_test)
            if name == "isotonic" and target == "MHR":
                calibrated = result
    assert calibrated is not None
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, p, title in zip(
        axes, [p_test, calibrated], ["Uncalibrated", "Isotonic to MHR"], strict=True
    ):
        reliability_diagram(p, q_test, mask=mask_test, n_bins=50, class_index=1, ax=ax, title=title)
    fig.tight_layout()
    path = Path("outputs/multi_rater_reliability.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Class reliability diagram saved to {path}")
    print(
        "Metrics pool test voxels; ECE-MHR differs from mean-rater-ECE. "
        "This synthetic example does not reproduce the TwinTrack benchmark."
    )


if __name__ == "__main__":
    main()
