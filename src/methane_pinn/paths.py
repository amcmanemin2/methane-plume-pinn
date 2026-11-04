"""Filesystem layout.

Paths come from the environment so that nothing is hardcoded to one machine.
On Sherlock, point DATA_ROOT at $OAK or $SCRATCH and RUN_ROOT at $SCRATCH;
on a laptop both default to directories inside the repo.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(os.path.expandvars(raw)).expanduser() if raw else default


def data_root() -> Path:
    """Where input data is read from. Never written to."""
    return _from_env("PINN_DATA_ROOT", REPO_ROOT / "data")


def les_root() -> Path:
    """Raw FastEddy output trees (read-only). Default: $SCRATCH/fasteddy/aus/runs/muswellbrook."""
    scratch = Path(os.environ.get("SCRATCH", str(Path.home())))
    return _from_env("PINN_LES_ROOT", scratch / "fasteddy" / "aus" / "runs" / "muswellbrook")


def run_root() -> Path:
    """Where checkpoints, logs and figures are written."""
    return _from_env("PINN_RUN_ROOT", REPO_ROOT / "runs")


def run_dir(name: str, create: bool = True) -> Path:
    """Directory for a single named run."""
    path = run_root() / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
