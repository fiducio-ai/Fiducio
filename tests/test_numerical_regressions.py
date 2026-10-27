"""Numerical limits that must not silently produce invalid results."""

import pytest
import torch

from fiducio import TemperatureScaling, brier_score


def test_transform_rejects_overflow_but_ignores_masked_padding():
    cal = TemperatureScaling(device="cpu", init_temperature=0.1, max_iter=1)
    cal.fit(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]))
    huge = torch.tensor([[1e38, -1e38], [2.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite.*overflow"):
        cal.transform(huge)
    with pytest.raises(ValueError, match="non-finite"):
        cal.decision_function(huge)
    out = cal.transform(huge, mask=torch.tensor([False, True]))
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.zeros(2))


@pytest.mark.parametrize("eps,dtype", [(1e-4, torch.float32), (1e-10, torch.float64)])
def test_brier_near_perfect(eps, dtype):
    p = torch.tensor([[1 - eps, eps]], dtype=dtype)
    expected = ((p.double() - torch.tensor([[1.0, 0.0]], dtype=torch.float64)) ** 2).sum()
    assert brier_score(p, torch.tensor([0])) == pytest.approx(expected.item(), rel=1e-12)


def test_near_tie_can_round_to_equal_probabilities():
    cal = TemperatureScaling(device="cpu", max_iter=1).fit(
        torch.tensor([[0.0, 2.0], [2.0, 0.0]]), torch.tensor([1, 0])
    )
    z = torch.tensor([[0.0, 1e-8]])
    assert z.argmax(1).item() == 1
    # The guarantee is in exact arithmetic; float32 may erase this gap.
    assert cal.transform(z).tolist() == [[0.5, 0.5]]
