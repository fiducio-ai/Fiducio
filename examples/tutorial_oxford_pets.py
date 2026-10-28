"""End-to-end calibration tutorial on the public Oxford-IIIT Pet dataset.

Trains a small U-Net on CPU, calibrates it on a held-out split and reports metrics
on a third, untouched test split. Needs ``pip install "fiducio[plots]"`` (which
brings Pillow and matplotlib). The first run downloads ~810 MB into ``--data-dir`` and takes a few minutes on CPU;
``--smoke`` uses a tiny synthetic dataset instead and needs no network. Run with::

    python examples/tutorial_oxford_pets.py            # full tutorial
    python examples/tutorial_oxford_pets.py --smoke    # seconds, no download

The three classes are the dataset's trimap: pet, background and pet boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")  # headless backend, no display needed
    import matplotlib.pyplot as plt
except ImportError:
    print('This tutorial needs matplotlib and Pillow. Install: pip install "fiducio[plots]"')
    sys.exit(0)

from fiducio import (
    CMSap,
    CMSop,
    MatrixScaling,
    TemperatureScaling,
    brier_score,
    diagnose,
    expected_calibration_error,
    negative_log_likelihood,
)
from fiducio.plots import reliability_diagram

BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/"
ARCHIVES = {
    "images.tar.gz": "67195c5e1c01f1ab5f9b6a5d22b8c27a580d896ece458917e61d459337fa318d",
    "annotations.tar.gz": "52425fb6de5c424942b7626b428656fcbd798db970a937df61750c0f1d358e91",
}
NUM_CLASSES = 3  # trimap 1/2/3 -> 0 pet, 1 background, 2 boundary


# --- 1. Data ---------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in ARCHIVES.items():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, path)
        if sha256(path) != expected:
            raise RuntimeError(f"{path} does not match the expected SHA-256; delete and retry.")


def load_pets(data_dir: Path, n: int, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``n`` random images ``(N, 3, size, size)`` and trimap labels ``(N, size, size)``."""
    from PIL import Image

    download(data_dir)
    with tarfile.open(data_dir / "annotations.tar.gz") as tar:
        lines = tar.extractfile("annotations/list.txt").read().decode().splitlines()  # type: ignore[union-attr]
    names = sorted(line.split()[0] for line in lines if not line.startswith("#"))
    g = torch.Generator().manual_seed(seed)
    wanted = {names[i] for i in torch.randperm(len(names), generator=g)[:n].tolist()}

    images: dict[str, torch.Tensor] = {}
    labels: dict[str, torch.Tensor] = {}
    # Stream each archive once, decoding only the files we picked.
    with tarfile.open(data_dir / "images.tar.gz") as tar:
        for member in tar:
            stem = Path(member.name).stem
            if member.isfile() and member.name.endswith(".jpg") and stem in wanted:
                img = Image.open(tar.extractfile(member)).convert("RGB")  # type: ignore[arg-type]
                img = img.resize((size, size), Image.BILINEAR)
                images[stem] = torch.tensor(list(img.tobytes()), dtype=torch.uint8).view(
                    size, size, 3
                )
    with tarfile.open(data_dir / "annotations.tar.gz") as tar:
        for member in tar:
            stem = Path(member.name).stem
            if member.name.startswith("annotations/trimaps/") and stem in wanted:
                if member.name.endswith(".png") and "/._" not in member.name:
                    tri = Image.open(tar.extractfile(member)).resize((size, size), Image.NEAREST)  # type: ignore[arg-type]
                    labels[stem] = torch.tensor(list(tri.tobytes()), dtype=torch.long).view(
                        size, size
                    )
    keep = sorted(images.keys() & labels.keys())
    x = torch.stack([images[k] for k in keep]).permute(0, 3, 1, 2).float() / 255.0
    return x, torch.stack([labels[k] for k in keep]) - 1


def synthetic(n: int, size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy discs with a boundary ring: a network-free stand-in for ``--smoke``."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij"))
    centre = torch.rand(n, 2, 1, 1, generator=g) * size * 0.5 + size * 0.25
    dist = (coords[None] - centre).float().pow(2).sum(1).sqrt()
    radius = size * 0.2
    labels = torch.ones(n, size, size, dtype=torch.long)  # background
    labels[dist < radius] = 0  # pet
    labels[(dist >= radius) & (dist < radius + 2)] = 2  # boundary
    signal = torch.zeros(n, 3, size, size) + (labels[:, None] == 0).float()
    return signal + torch.randn(n, 3, size, size, generator=g), labels


# --- 2. Model --------------------------------------------------------------------------


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(),
    )


class SmallUNet(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.enc1, self.enc2 = block(3, width), block(width, 2 * width)
        self.mid = block(2 * width, 4 * width)
        self.up2 = nn.ConvTranspose2d(4 * width, 2 * width, 2, stride=2)
        self.dec2 = block(4 * width, 2 * width)
        self.up1 = nn.ConvTranspose2d(2 * width, width, 2, stride=2)
        self.dec1 = block(2 * width, width)
        self.head = nn.Conv2d(width, NUM_CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        m = self.mid(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(m), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)  # (B, C, H, W) logits


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), 16):
            idx = perm[i : i + 16]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {epoch + 1}/{epochs}  train loss {total / len(x):.4f}")


@torch.no_grad()
def logits_of(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i : i + 32]) for i in range(0, len(x), 32)])


# --- 3. Evaluation ---------------------------------------------------------------------


def dice(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Macro Dice over classes present in the labels."""
    pred = probs.argmax(1)
    scores = []
    for c in range(NUM_CLASSES):
        inter = ((pred == c) & (labels == c)).sum().item()
        total = (pred == c).sum().item() + (labels == c).sum().item()
        if (labels == c).any():
            scores.append(2 * inter / max(total, 1))
    return sum(scores) / len(scores)


def report(name: str, probs: torch.Tensor, labels: torch.Tensor) -> None:
    print(
        f"  {name:<18} NLL {negative_log_likelihood(probs, labels):.4f}  "
        f"ECE {expected_calibration_error(probs, labels):.4f}  "
        f"Brier {brier_score(probs, labels):.4f}  Dice {dice(probs, labels):.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="tiny synthetic run, no download")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache/fiducio/oxford-pets")
    parser.add_argument("--n-images", type=int, default=1200)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out", type=Path, default=Path("outputs/tutorial_reliability.png"))
    args = parser.parse_args()
    if args.size % 4 or args.n_images < 8:
        parser.error("--size must be a multiple of 4 and --n-images at least 8")
    if args.smoke:
        args.n_images, args.size, args.epochs = 90, 32, 2

    torch.manual_seed(0)
    if args.smoke:
        x, y = synthetic(args.n_images, args.size, seed=0)
    else:
        x, y = load_pets(args.data_dir, args.n_images, args.size, seed=0)

    # Three disjoint splits. Calibrating on the training images would see an
    # over-fitted model, and calibrating on the test split would leak the answer.
    n = len(x)
    n_train, n_cal = int(0.5 * n), int(0.25 * n)
    splits = {
        "train": slice(0, n_train),
        "calibration": slice(n_train, n_train + n_cal),
        "test": slice(n_train + n_cal, n),
    }
    for name, s in splits.items():
        print(f"{name:<12} {len(x[s])} images")

    print("Training:")
    model = SmallUNet()
    train(model, x[splits["train"]], y[splits["train"]], args.epochs)

    cal_logits = logits_of(model, x[splits["calibration"]])
    cal_labels = y[splits["calibration"]]
    test_logits = logits_of(model, x[splits["test"]])
    test_labels = y[splits["test"]]

    # Check how reliable the model is on the calibration split before calibrating.
    # The background is class 1 in the trimap labels.
    print("\nDiagnosis on the calibration split:")
    print(
        diagnose(
            cal_logits,
            cal_labels,
            class_names=["pet", "background", "boundary"],
            background_class=1,
        )
    )

    # Fit on the calibration split only.
    ts = TemperatureScaling(device="cpu").fit(cal_logits, cal_labels)
    ms = MatrixScaling(lambda_reg=1e-2, device="cpu").fit(cal_logits, cal_labels)
    cmsap = CMSap(device="cpu").fit(cal_logits, cal_labels)
    cmsop = CMSop(device="cpu").fit(cal_logits, cal_labels)

    print("Test metrics (the calibrators never saw these images):")
    raw = F.softmax(test_logits, dim=1)
    report("uncalibrated", raw, test_labels)
    report("TemperatureScaling", ts.transform(test_logits), test_labels)
    report("MatrixScaling", ms.transform(test_logits), test_labels)
    report("CMSap", cmsap.transform(test_logits), test_labels)
    report("CMSop", cmsop.transform(test_logits), test_labels)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    reliability_diagram(raw, test_labels, ax=axes[0], title="Uncalibrated (test)")
    reliability_diagram(ts.transform(test_logits), test_labels, ax=axes[1], title="TS (test)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to {args.out}")


if __name__ == "__main__":
    main()
