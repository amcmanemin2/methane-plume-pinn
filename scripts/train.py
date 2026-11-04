#!/usr/bin/env python3
"""Train the pit-2 methane PINN (PLAN.md).

    python3 scripts/train.py --check-only                         # environment check only
    python3 scripts/train.py --config configs/default.yaml --run-name base
    python3 scripts/train.py --config configs/smoke.yaml --run-name smoke --note "first smoke test"

Outputs (best.pt, last.pt, losses.csv, losses.json, config.yaml, git.patch) go to
$PINN_RUN_ROOT/<YYYYmmdd-HHMMSS>_<run-name>/; every run appends a row to $PINN_RUN_ROOT/runs_log.csv.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import torch
import yaml

from methane_pinn import runs as R
from methane_pinn.device import describe, resolve_device
from pathlib import Path

from methane_pinn.paths import REPO_ROOT, data_root, run_dir, run_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-name", default="smoke")
    parser.add_argument("--note", default="", help="what changed relative to the previous run (logged)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config entry, e.g. training.steps=500 (value parsed as YAML)")
    parser.add_argument("--check-only", action="store_true",
                        help="Run the environment check and exit without training.")
    return parser.parse_args(argv)


def load_config(path, sets):
    """YAML config; a top-level `base:` key names a config file whose entries this one overrides."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    base = cfg.pop("base", None)
    if base:
        merged = load_config(Path(path).parent / base, [])
        _deep_update(merged, cfg)
        cfg = merged
    return apply_overrides(cfg, sets)


def _deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v


def git_info(out_dir=None):
    """Short hash and dirty flag; writes the uncommitted diff to out_dir/git.patch."""
    def run(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL)
    try:
        h = run("rev-parse", "--short", "HEAD").strip()
        diff = run("diff", "HEAD")
        untracked = run("ls-files", "--others", "--exclude-standard").split()
    except Exception:
        return dict(hash="unknown", dirty=None)
    if out_dir is not None:
        (out_dir / "git.patch").write_text(diff)
    return dict(hash=h, dirty=bool(diff.strip() or untracked), untracked=untracked)


def append_runs_log(row: dict):
    path = run_root() / "runs_log.csv"
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def apply_overrides(cfg, sets):
    for item in sets:
        key, val = item.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(val)
    return cfg


def check_autograd(device: torch.device) -> float:
    """Second derivative of sin(x) should be -sin(x); report the max error."""
    x = torch.linspace(0.0, 3.14159, 128, device=device, requires_grad=True)
    y = torch.sin(x).sum()
    (dy,) = torch.autograd.grad(y, x, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    return (d2y + torch.sin(x)).abs().max().item()


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)

    device = resolve_device(cfg.get("run", {}).get("device"))
    print(f"torch      {torch.__version__}")
    print(f"device     {describe(device)}")
    print(f"data root  {data_root()}  (exists: {data_root().exists()})")
    print(f"run root   {run_root()}")
    error = check_autograd(device)
    print(f"autograd   second-derivative max error {error:.2e}")
    if error > 1e-4:
        print("FAIL: higher-order autograd is inaccurate on this device", file=sys.stderr)
        return 1
    if args.check_only:
        return 0

    from methane_pinn.sampling import TrainSet
    from methane_pinn.train import all_seeds, train_pinn

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = run_dir(f"{stamp}_{args.run_name}")
    names = cfg["data"].get("train_runs")
    train_runs = [R.get(n) for n in names] if names else R.runs("train")
    print(f"run dir    {out}\ntraining on {len(train_runs)} runs: {[r.name for r in train_runs]}", flush=True)
    git = git_info(out)
    meta = dict(git=git["hash"], git_dirty=git["dirty"], jobid=os.environ.get("SLURM_JOB_ID", ""),
                device=describe(device), train_runs=[r.name for r in train_runs], note=args.note,
                overrides=args.set, config_file=args.config, started=stamp, seeds=all_seeds(cfg))
    (out / "config.yaml").write_text(yaml.safe_dump(cfg | {"meta": meta}, sort_keys=False))
    append_runs_log(dict(started=stamp, run_dir=out.name, jobid=meta["jobid"], git=git["hash"],
                         dirty=git["dirty"], config=args.config, overrides=" ".join(args.set), note=args.note))

    t0 = time.time()
    ts = TrainSet(train_runs, stride=cfg["data"]["stride"], device=device,
                  sigma_h=cfg["physics"]["sigma_h"], sigma_z=cfg["physics"]["sigma_z"],
                  mask_thresh=cfg["data"].get("plume_thresh", 1e-8))
    print(f"data loaded in {time.time() - t0:.0f} s", flush=True)
    model, losses = train_pinn(ts, cfg, out)
    info = json.loads((out / "losses.json").read_text())["info"]
    meta["result"] = dict(final_val_rmse=losses["val_rmse"][-1], final_val_r2=losses["val_r2"][-1],
                          load_s=round(time.time() - t0 - info["wall_s"]), **info)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg | {"meta": meta}, sort_keys=False))
    print(json.dumps(meta["result"], default=str))
    (run_root() / "LATEST").write_text(out.name + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
