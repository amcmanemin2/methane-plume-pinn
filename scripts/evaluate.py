#!/usr/bin/env python3
"""Evaluate a trained PINN against the full LES fields (PLAN.md §7), on validation and test.

    validation: the training runs with their 200 held-out val combinations (same flows as training)
    test:       the 12 test runs with their 100 test combinations (never seen in training)

    python3 scripts/evaluate.py                                   # latest run dir, best.pt, val + test
    python3 scripts/evaluate.py --run-dir 20261104-153000_smoke --test-runs daytime_NE --n-combos 2

Writes <run dir>/eval/: metrics_val.csv, metrics_test.csv, eval_<split>_<run>.npz and figures.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methane_pinn import data as D
from methane_pinn import plots as P
from methane_pinn import runs as R
from methane_pinn.device import resolve_device
from methane_pinn.evaluate import evaluate_run, load_model, metrics_from_stats, predict_points, save_results
from methane_pinn.paths import run_root


def level_slice(les: D.LESRun, frames, q, height):
    """LES concentration at `height` m AGL on the full 20 m grid, (F, Ny, Nx)."""
    za = les.grid["z_agl"][:4]
    with h5py.File(les.path, "r") as h:
        b = h["c"][frames[0]:frames[-1] + 1, :, :4]                     # (F, S, 4, Ny, Nx)
    c = np.tensordot(q / R.Q0_KGPH, b, axes=([0], [1]))                   # (F, 4, Ny, Nx)
    k = np.clip((za < height).sum(0) - 1, 0, 2)                          # (Ny, Nx)
    zk = np.take_along_axis(za, k[None], 0)[0]
    zk1 = np.take_along_axis(za, k[None] + 1, 0)[0]
    w = (height - zk) / (zk1 - zk)
    ck = np.take_along_axis(c, k[None, None].repeat(c.shape[0], 0), 1)[:, 0]
    ck1 = np.take_along_axis(c, k[None, None].repeat(c.shape[0], 0) + 1, 1)[:, 0]
    return ck * (1 - w) + ck1 * w


def maps_for_run(model, run, split, q, args, batch, device, ev_dir, box=None):
    les = D.LESRun.open(run)
    g = les.grid
    lay = D.sensor_layout(les)
    met = torch.as_tensor(D.sensor_series(les, lay)["met"], device=device)
    frames = np.arange(1, len(g["time"]))
    xs, ys, topo = g["x"], g["y"], g["topo"]
    if box is not None:       # sub-domain: its snapshots and its columns only
        frames = frames[g["time"][frames] <= box["t0"] + box["T"] + 1e-6]
        ix = np.flatnonzero((xs >= box["x0"]) & (xs <= box["x0"] + box["Lx"]))
        iy = np.flatnonzero((ys >= box["y0"]) & (ys <= box["y0"] + box["Ly"]))
        xs, ys, topo = xs[ix], ys[iy], topo[np.ix_(iy, ix)]
    les_f = level_slice(les, frames, q, args.map_height)
    if box is not None:
        les_f = les_f[:, iy][:, :, ix]
    Xg, Yg = np.meshgrid(xs, ys)
    X = torch.as_tensor(np.stack([Xg.ravel(), Yg.ravel(), (topo + args.map_height).ravel(),
                                  np.zeros(Xg.size)], -1), dtype=torch.float32, device=device)
    pinn_f = np.empty_like(les_f)
    for jf, it in enumerate(frames):
        X[:, 3] = float(g["time"][it])
        pinn_f[jf] = predict_points(model, X, q, met[it].expand(len(X), -1), batch).reshape(Xg.shape)
    on = np.flatnonzero(q > 0)
    t_last = (g["time"][frames[-1]] - 3600.0) / 60.0
    P.slice_maps(xs, ys, [les_f[-1], les_f.mean(0)], [pinn_f[-1], pinn_f.mean(0)], g["src_xy"],
                 ev_dir / f"maps_{split}_{run.name}.png",
                 [f"{run.name} t = {t_last:.0f} min, {args.map_height:.0f} m AGL",
                  f"{run.name} mean of evaluated snapshots, {args.map_height:.0f} m AGL"],
                 sensors_xy=lay.xy, lidar_xy=lay.lidar_xy)
    print(f"maps {split} {run.name}: {len(on)} sources, {q.sum():.0f} kg/h", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="run directory name under $PINN_RUN_ROOT (default: LATEST)")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--splits", nargs="*", default=["val", "test"])
    p.add_argument("--val-runs", nargs="*", default=None, help="default: the runs the model was trained on")
    p.add_argument("--test-runs", nargs="*", default=None, help="default: all 12 test runs")
    p.add_argument("--n-combos", type=int, default=None, help="per run; default from the config")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--map-height", type=float, default=10.0, help="m AGL for the map figures")
    args = p.parse_args(argv)

    device = resolve_device(None)
    name = args.run_dir or (run_root() / "LATEST").read_text().strip()
    out = run_root() / name
    model, state = load_model(out / args.ckpt, device)
    trained_on = state["runs"]
    ev = state["cfg"].get("evaluation", {})
    stride = args.stride or ev.get("stride", 2)
    batch = ev.get("batch", 262144)
    print(f"{out}: checkpoint step {state['step']}, val points {state['val']}, trained on {trained_on}", flush=True)
    ev_dir = out / "eval"
    scen = state["cfg"].get("scenario")
    splits = ["scenario"] if scen and args.splits == ["val", "test"] else args.splits
    for split in splits:
        combos, box = None, None
        if split == "scenario":
            from train import scenario_rates
            runs, combos, n_combos = [R.get(scen["run"])], scenario_rates(scen)[None], 1
            box = state["domain"] if scen.get("subdomain") else None
        elif split == "val":
            runs = [R.get(n) for n in (args.val_runs or trained_on)]
            n_combos = args.n_combos or ev.get("n_combos_val", D.N_COMBOS["val"])
        else:
            runs = [R.get(n) for n in args.test_runs] if args.test_runs else R.runs("test")
            n_combos = args.n_combos or ev.get("n_combos", D.N_COMBOS["test"])
        print(f"[{split}] {[r.name for r in runs]}, {n_combos} combinations, stride {stride}", flush=True)
        results = [evaluate_run(model, r, n_combos, stride, batch, device, split=split, train_runs=trained_on,
                                combos=combos, box=box) for r in runs]
        rows = save_results(results, ev_dir, tag=split)
        for row in rows:
            print({k: (round(v, 4) if isinstance(v, float) and abs(v) > 1e-3 else v) for k, v in row.items()})
        P.metrics_vs_time(results, ev_dir / f"metrics_vs_time_{split}.png", metrics_from_stats)
        P.sensor_parity(results, ev_dir / f"sensor_parity_{split}.png", title=f"Sensors, 10 m AGL ({split})")
        P.field_parity(results, ev_dir / f"field_parity_{split}.png", title=f"Grid cells ({split})")
        for run in runs:       # maps: first combination of the split, last snapshot and release mean
            q = combos[0] if combos is not None else D.combination_table(run, split)[0]
            maps_for_run(model, run, split, q, args, batch, device, ev_dir, box=box)
    if (out / "losses.csv").exists():
        P.loss_curves(out, ev_dir / "loss_curves.png")
    print(f"wrote {ev_dir}")


if __name__ == "__main__":
    main()
