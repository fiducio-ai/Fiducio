"""Calibrate two ensemble pooling strategies on disjoint synthetic cases."""

from __future__ import annotations

import torch

from fiducio import CMSap, negative_log_likelihood


def main():
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(42)
    # (members, cases, classes, height, width)
    members = torch.randn(3, 12, 4, 8, 8, generator=g)
    labels = torch.randint(0, 4, (12, 8, 8), generator=g)
    pooled = {
        "mean probabilities": (members.softmax(dim=2).mean(dim=0), "probs"),
        "mean logits": (members.mean(dim=0), "logits"),
    }
    for name, (predictions, input_type) in pooled.items():
        # Fit on calibration cases only, then apply that fitted instance to test cases.
        cal = CMSap(input_type=input_type, device="cpu", max_iter=30)
        cal.fit(predictions[:8], labels[:8])
        out = cal.transform(predictions[8:])
        assert torch.equal(out.argmax(1), predictions[8:].argmax(1))
        print(f"{name}: test NLL={negative_log_likelihood(out, labels[8:]):.4f}")


if __name__ == "__main__":
    main()
