"""Device resolution, so the same code runs on a laptop and on a Sherlock GPU node."""

import os

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a device.

    Order of precedence: explicit argument, then $PINN_DEVICE, then autodetect
    (cuda, then Apple MPS, then cpu).
    """
    choice = requested or os.environ.get("PINN_DEVICE", "auto")

    if choice != "auto":
        return torch.device(choice)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    return device.type
