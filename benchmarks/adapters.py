"""One adapter per library: ``fit(z, y)`` on logits ``(N, C)``, ``predict(z)`` -> probabilities.

Input contract (documented on the benchmark page):
* Fiducio and torch-uncertainty receive logits.
* netcal and dirichletcal only accept probabilities, so they receive ``softmax(z)`` in float64.
  netcal silently leaves T = 1 on float32 inputs; its ``T`` multiplies the logits (Fiducio divides).
Imports happen in ``__init__`` so that they stay out of the timings.
"""

from __future__ import annotations

import importlib.metadata as md

import torch

C_DEFAULT = 4


class Identity:
    lib, dtype = "none", "float32"

    def __init__(self, C):
        pass

    def fit(self, z, y):
        pass

    def predict(self, z):
        return z.softmax(1)


class Fiducio:
    lib, dtype = "fiducio", "float32"
    cls = "TemperatureScaling"

    def __init__(self, C):
        import fiducio

        self.m = getattr(fiducio, self.cls)(input_type="logits", device="cpu")

    def fit(self, z, y):
        self.m.fit(z, y)

    def predict(self, z):
        return self.m.transform(z)


class Netcal:
    lib, dtype = "netcal", "float64"

    def __init__(self, C):
        from netcal.scaling import TemperatureScaling

        self.m = TemperatureScaling()

    def fit(self, z, y):
        self.m.fit(z.softmax(1).double().numpy(), y.numpy(), tensorboard=False)

    def predict(self, z):
        p = torch.tensor(self.m.transform(z.softmax(1).double().numpy())).float()
        return (
            torch.stack([1 - p, p], 1) if p.ndim == 1 else p
        )  # binary mode returns P(class 1) only


class TorchUncertainty:
    lib, dtype = "torch-uncertainty", "float32"
    cls = "TemperatureScaler"

    def __init__(self, C):
        from torch_uncertainty import post_processing as pp

        kw = {} if self.cls == "TemperatureScaler" else {"num_classes": C}
        self.m = getattr(pp, self.cls)(device="cpu", **kw)

    def fit(self, z, y):
        from torch.utils.data import DataLoader, TensorDataset

        self.m.fit(DataLoader(TensorDataset(z, y), batch_size=65536), progress=False)

    def predict(self, z):
        with torch.no_grad():
            return self.m(z).softmax(1)


class Dirichletcal:
    lib, dtype = "dirichletcal", "float64"  # JAX is float32 unless x64 (diff a few 1e-6)

    def __init__(self, C):
        from dirichletcal import FullDirichletCalibrator

        self.m = FullDirichletCalibrator()

    def fit(self, z, y):
        self.m.fit(z.softmax(1).double().numpy(), y.numpy())

    def predict(self, z):
        return torch.tensor(self.m.predict_proba(z.softmax(1).double().numpy())).float()


class FiducioProbs(Fiducio):
    """Fiducio with ``input_type='probs'``: calibrators take ``log(p)`` internally."""

    lib, dtype = "fiducio", "float32"

    def __init__(self, C):
        import fiducio

        self.m = getattr(fiducio, self.cls)(input_type="probs", device="cpu")

    def fit(self, z, y):
        self.m.fit(z.softmax(1), y)

    def predict(self, z):
        return self.m.transform(z.softmax(1))


class NetcalRaw(Netcal):
    """netcal without the adapter's float64 cast, to expose dtype sensitivity."""

    def fit(self, z, y):
        self.m.fit(z.softmax(1).numpy(), y.numpy(), tensorboard=False)

    def predict(self, z):
        p = torch.tensor(self.m.transform(z.softmax(1).numpy())).float()
        return torch.stack([1 - p, p], 1) if p.ndim == 1 else p


def _sub(base, **attrs):
    return type(attrs.pop("name"), (base,), attrs)


# name -> (adapter class, matching Fiducio method for paired differences)
ADAPTERS = {
    "identity": (Identity, None),
    "fiducio TS": (_sub(Fiducio, name="F1", cls="TemperatureScaling"), None),
    "fiducio VS": (_sub(Fiducio, name="F2", cls="VectorScaling"), None),
    "fiducio MS": (_sub(Fiducio, name="F3", cls="MatrixScaling"), None),
    "fiducio DC": (_sub(Fiducio, name="F4", cls="DirichletCalibration"), None),
    "netcal TS": (Netcal, "fiducio TS"),
    "torch-uncertainty TS": (TorchUncertainty, "fiducio TS"),
    "torch-uncertainty VS": (_sub(TorchUncertainty, name="T2", cls="VectorScaler"), "fiducio VS"),
    "torch-uncertainty MS": (_sub(TorchUncertainty, name="T3", cls="MatrixScaler"), "fiducio MS"),
    "dirichletcal DC": (Dirichletcal, "fiducio DC"),
}

# Axis 4: Fiducio only, identity as the reference decision.
DOWNSTREAM_ADAPTERS = {
    name: ADAPTERS[name]
    for name in ("identity", "fiducio TS", "fiducio VS", "fiducio MS", "fiducio DC")
}
DOWNSTREAM_ADAPTERS |= {
    "fiducio CMSap": (_sub(Fiducio, name="F5", cls="CMSap"), None),
    "fiducio CMSop": (_sub(Fiducio, name="F6", cls="CMSop"), None),
}

# Axis 5: every adapter, plus the probability-input and raw-dtype variants that
# the robustness cases are about. No paired reference: the axis records status,
# not an improvement.
ROBUSTNESS_ADAPTERS = ADAPTERS | {
    "fiducio VS (probs)": (_sub(FiducioProbs, name="FP1", cls="VectorScaling"), None),
    "fiducio DC (probs)": (_sub(FiducioProbs, name="FP2", cls="DirichletCalibration"), None),
    "netcal TS (raw dtype)": (NetcalRaw, None),
}


def version(lib):
    try:
        return md.version(lib)
    except md.PackageNotFoundError:
        return ""
