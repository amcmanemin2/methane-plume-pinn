#!/usr/bin/env python
"""Training entry point.

The model itself is not implemented yet -- the PDE, boundary conditions and
loss terms come from Chapter 4 of the thesis. For now this verifies that the
environment is sound: torch imports, a device resolves, the configured paths
are reachable, and autograd produces correct derivatives (which is the one
capability a PINN cannot do without).
"""

import argparse
import sys

import torch
import yaml

from methane_pinn.device import describe, resolve_device
from methane_pinn.paths import data_root, run_dir, run_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run the environment check and exit without training.",
    )
    return parser.parse_args(argv)


def check_autograd(device: torch.device) -> float:
    """Second derivative of sin(x) should be -sin(x); report the max error.

    A PINN residual needs higher-order autograd on whichever device it runs on,
    and MPS in particular has historically been patchy about it.
    """
    x = torch.linspace(0.0, 3.14159, 128, device=device, requires_grad=True)
    y = torch.sin(x).sum()
    (dy,) = torch.autograd.grad(y, x, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    return (d2y + torch.sin(x)).abs().max().item()


def main(argv=None) -> int:
    args = parse_args(argv)

    with open(args.config) as fh:
        config = yaml.safe_load(fh) or {}

    torch.manual_seed(config.get("run", {}).get("seed", 0))
    device = resolve_device(config.get("run", {}).get("device"))
    out = run_dir(args.run_name)

    print(f"torch      {torch.__version__}")
    print(f"device     {describe(device)}")
    print(f"data root  {data_root()}  (exists: {data_root().exists()})")
    print(f"run root   {run_root()}")
    print(f"run dir    {out}")

    error = check_autograd(device)
    print(f"autograd   second-derivative max error {error:.2e}")
    if error > 1e-4:
        print("FAIL: higher-order autograd is inaccurate on this device", file=sys.stderr)
        return 1

    if args.check_only:
        return 0

    print("\nEnvironment OK. Model not implemented yet -- see configs/default.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
