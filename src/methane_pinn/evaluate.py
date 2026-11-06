"""Evaluation against the full LES fields (PLAN.md §7).

For every test run and test combination, the PINN prediction is compared with the superposed LES
field on the evaluation grid (every `stride`-th column, all levels) at each of the release
snapshots t = 3720..7200 s, and on the time mean over those snapshots. Per (combination, snapshot)
we store SSE, SST, sum(y^2) and n, from which RMSE, R^2 and relative L2 follow; we also store the
methane time series at the 20 sensors (10 m AGL).
"""

import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from methane_pinn import data as D
from methane_pinn import runs as R
from methane_pinn.model import PlumeModel
from methane_pinn.physics import Terrain
from methane_pinn.sampling import STAB_INDEX


def load_model(ckpt_path, device="cpu"):
    """Rebuild a PlumeModel from a checkpoint written by train.train_pinn."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg, sd = state["cfg"], state["model"]
    g = D.LESRun.open(state["runs"][0]).grid
    terrain = Terrain(g["x"], g["y"], g["topo"], device=device)
    n_met = sd["met_mean"].numel()
    net = cfg["network"]
    model = PlumeModel(state["domain"], terrain, n_src=D.N_SRC, n_met=n_met, met_mean=np.zeros(n_met),
                       met_std=np.ones(n_met), q_max=D.Q_TOTAL_MAX_KGPH, c_ref=float(cfg["physics"]["c_ref"]),
                       n_classes=len(STAB_INDEX), inputs=net.get("inputs", "full"),
                       K_trainable=cfg["physics"].get("K_trainable", True),
                       hidden_size=net["hidden_size"], num_layers=net["num_layers"])
    model.load_state_dict(sd)
    return model.to(device).eval(), state


# Revised criterion (TRAINING_LOG.md, entry D1/E4): the near-source peak cells dominate linear metrics,
# so every evaluation also reports the far field and a log-space fit, next to the PLAN.md metrics.
NEAR_SOURCE_M = 100.0       # far field = horizontal distance > 100 m from every pit-2 source
LOG_FLOOR = 1e-8            # kg m-3; log-space metrics on cells with LES c above this
OVERFIT_PASS_REL_L2 = 0.05  # overfit test passes if far-field relative L2 < 0.05


def far_field_mask(x, y, src_xy, radius=NEAR_SOURCE_M):
    """True where the horizontal distance to every source exceeds `radius` (arrays of any shape)."""
    x, y = np.asarray(x, float)[..., None], np.asarray(y, float)[..., None]
    d = np.hypot(x - src_xy[:, 0], y - src_xy[:, 1]).min(-1)
    return d > radius


def point_metrics(pred, y, far):
    """Metrics on a point set: all cells (PLAN.md), far field only, and log space (c > LOG_FLOOR)."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    out = {}
    for tag, m in (("all", np.ones_like(y, bool)), ("far", np.asarray(far, bool))):
        sse, sst, syy, n = _stats(pred[m], y[m])
        out.update({f"{tag}_{k}": float(v) for k, v in metrics_from_stats(sse, sst, syy, n).items()})
        out[f"{tag}_n"] = int(m.sum())
    pl = y > LOG_FLOOR
    dl = np.abs(np.log10(np.maximum(pred[pl], 1e-30)) - np.log10(y[pl]))
    out.update(log_median_abs_dlog10=float(np.median(dl)) if pl.any() else float("nan"),
               log_frac_within_2x=float((dl < np.log10(2)).mean()) if pl.any() else float("nan"), log_n=int(pl.sum()))
    return out


def _stats(pred, y):
    """SSE, SST (about mean of y), sum(y^2), n for flat float64 arrays."""
    d = pred - y
    return float(d @ d), float(((y - y.mean()) ** 2).sum()), float(y @ y), y.size


def metrics_from_stats(sse, sst, syy, n):
    """RMSE, R^2, relative L2 (elementwise over arrays of accumulated statistics)."""
    sse, sst, syy, n = (np.asarray(a, float) for a in (sse, sst, syy, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        return dict(rmse=np.sqrt(sse / n), r2=1 - sse / sst, rel_l2=np.sqrt(sse / syy))


@torch.no_grad()
def predict_points(model, X, q, met, batch):
    """Model prediction at physical points X (N, 4) for one combination q (S,) and met (N, M)."""
    out = np.empty(len(X))
    qb = torch.as_tensor(q, dtype=torch.float32, device=X.device)
    for i in range(0, len(X), batch):
        Xi = X[i:i + batch]
        out[i:i + batch] = model(Xi, qb.expand(len(Xi), -1), met[i:i + batch]).double().cpu().numpy()
    return out


def check_split(run: R.Run, split: str, train_runs=()):
    """Leakage guard: test metrics only on test runs never trained on; val only on training runs."""
    if split == "test" and (run.split != "test" or run.name in set(train_runs)):
        raise ValueError(f"{run.name}: not a held-out test run (split={run.split}, trained on: {list(train_runs)})")
    if split in ("val", "scenario") and run.split != "train":
        raise ValueError(f"{run.name}: {split} evaluation is only for training runs")


@torch.no_grad()
def evaluate_run(model, run: R.Run, n_combos=100, stride=2, batch=262144, device="cpu", log=print,
                 split="test", train_runs=(), n_parity=50, seed=0, combos=None, box=None):
    """Metrics of one run against the superposed LES for the first n_combos combinations of `split`.

    box: a sub-domain dict (x0, y0, Lx, Ly, Lz [m AGL], t0, T) as in TrainSet.set_subdomain; only grid
    cells inside it and snapshots with t <= t0 + T are evaluated.

    n_parity random grid cells per (combination, snapshot) are kept for field parity plots.
    """
    check_split(run, split, train_runs)
    prng = np.random.default_rng(seed)
    les = D.LESRun.open(run)
    g = les.grid
    layout = D.sensor_layout(les)
    met_all = torch.as_tensor(D.sensor_series(les, layout)["met"], device=device)
    combos = D.combination_table(run, split)[:n_combos] if combos is None else np.atleast_2d(combos)
    frames = np.arange(1, len(g["time"]))                          # t > 3600 s (release snapshots)
    if box is not None:
        frames = frames[g["time"][frames] <= box["t0"] + box["T"] + 1e-6]
    s = stride
    x, y, z = g["x"][::s], g["y"][::s], g["z"][:, ::s, ::s]
    K, ny, nx = z.shape
    Xg = np.stack(np.broadcast_arrays(x[None, None, :], y[None, :, None], z, 0.0), -1).reshape(-1, 4)
    cells = np.arange(len(Xg))
    if box is not None:
        zagl = g["z_agl"][:, ::s, ::s].reshape(-1)
        inb = ((Xg[:, 0] >= box["x0"]) & (Xg[:, 0] <= box["x0"] + box["Lx"]) & (Xg[:, 1] >= box["y0"])
               & (Xg[:, 1] <= box["y0"] + box["Ly"]) & (zagl <= box["Lz"]))
        cells = np.flatnonzero(inb)
        Xg = Xg[cells]
    X = torch.as_tensor(Xg, dtype=torch.float32, device=device)
    with h5py.File(les.path, "r") as h:
        basis = np.stack([h["c"][it, :, :, ::s, ::s] for it in frames])       # (F, S, K, ny, nx)
    # sensors: LES at 10 m AGL from the full-resolution columns
    za = g["z_agl"]
    sens_basis = np.empty((len(frames), D.N_SRC, len(layout.cols)))
    with h5py.File(les.path, "r") as h:
        for n, (j, i) in enumerate(layout.cols):
            prof = h["c"][frames[0]:frames[-1] + 1, :, :12, j, i]               # (F, S, 12)
            k = np.searchsorted(za[:12, j, i], D.SENSOR_AGL_M) - 1
            w = (D.SENSOR_AGL_M - za[k, j, i]) / (za[k + 1, j, i] - za[k, j, i])
            sens_basis[:, :, n] = prof[..., k] * (1 - w) + prof[..., k + 1] * w
    Xs = np.stack([layout.xy[:, 0], layout.xy[:, 1],
                   g["topo"][layout.cols[:, 0], layout.cols[:, 1]] + D.SENSOR_AGL_M, np.zeros(len(layout.cols))], -1)
    Xs = torch.as_tensor(Xs, dtype=torch.float32, device=device)

    nC, nF = len(combos), len(frames)
    acc = {k: np.zeros((nC, nF)) for k in ("sse", "sst", "syy", "n")}
    tm = {k: np.zeros(nC) for k in ("sse", "sst", "syy", "n")}
    far = far_field_mask(Xg[:, 0], Xg[:, 1], g["src_xy"])          # > 100 m from every source
    acc_far = {k: np.zeros((nC, nF)) for k in ("sse", "sst", "syy", "n")}
    tm_far = {k: np.zeros(nC) for k in ("sse", "sst", "syy", "n")}
    log_med, log_2x = np.full((nC, nF), np.nan), np.full((nC, nF), np.nan)
    tm_log_med, tm_log_2x = np.full(nC, np.nan), np.full(nC, np.nan)

    def _log_fit(p, y):
        pl = y > LOG_FLOOR
        if not pl.any():
            return np.nan, np.nan
        dl = np.abs(np.log10(np.maximum(p[pl], 1e-30)) - np.log10(y[pl]))
        return float(np.median(dl)), float((dl < np.log10(2)).mean())
    sens_pred, sens_true = np.zeros((nC, nF, len(layout.cols))), np.zeros((nC, nF, len(layout.cols)))
    par_pred, par_true = np.zeros((nC, nF, n_parity)), np.zeros((nC, nF, n_parity))
    t0 = time.time()
    for ic, q in enumerate(combos):
        w = q / R.Q0_KGPH
        sum_p, sum_y = np.zeros(X.shape[0]), np.zeros(X.shape[0])
        for jf, it in enumerate(frames):
            t = float(g["time"][it])
            X[:, 3] = t
            met = met_all[it].expand(X.shape[0], -1)
            p = predict_points(model, X, q, met, batch)
            yv = np.tensordot(w, basis[jf].reshape(D.N_SRC, -1)[:, cells], axes=1)
            for k, v in zip(("sse", "sst", "syy", "n"), _stats(p, yv)):
                acc[k][ic, jf] = v
            for k, v in zip(("sse", "sst", "syy", "n"), _stats(p[far], yv[far])):
                acc_far[k][ic, jf] = v
            log_med[ic, jf], log_2x[ic, jf] = _log_fit(p, yv)
            sel = prng.integers(0, len(p), n_parity)
            par_pred[ic, jf], par_true[ic, jf] = p[sel], yv[sel]
            sum_p += p
            sum_y += yv
            Xs[:, 3] = t
            sens_pred[ic, jf] = predict_points(model, Xs, q, met_all[it].expand(len(Xs), -1), batch)
            sens_true[ic, jf] = w @ sens_basis[jf]
        for k, v in zip(("sse", "sst", "syy", "n"), _stats(sum_p / nF, sum_y / nF)):
            tm[k][ic] = v
        for k, v in zip(("sse", "sst", "syy", "n"), _stats((sum_p / nF)[far], (sum_y / nF)[far])):
            tm_far[k][ic] = v
        tm_log_med[ic], tm_log_2x[ic] = _log_fit(sum_p / nF, sum_y / nF)
        if ic == 0 or (ic + 1) % 10 == 0:
            m = metrics_from_stats(acc["sse"][ic], acc["sst"][ic], acc["syy"][ic], acc["n"][ic])
            log(f"  {run.name} combo {ic + 1}/{nC}: mean R2 {np.nanmean(m['r2']):.3f}  "
                f"({(time.time() - t0) / (ic + 1):.1f} s/combo)", flush=True)
    return dict(run=run.name, stab=run.stab, bearing=run.bearing, split=split, combos=combos, times=g["time"][frames],
                par_pred=par_pred, par_true=par_true,
                **{f"snap_{k}": v for k, v in acc.items()}, **{f"tmean_{k}": v for k, v in tm.items()},
                **{f"snapfar_{k}": v for k, v in acc_far.items()}, **{f"tmeanfar_{k}": v for k, v in tm_far.items()},
                snap_log_med=log_med, snap_log_2x=log_2x, tmean_log_med=tm_log_med, tmean_log_2x=tm_log_2x,
                sens_pred=sens_pred, sens_true=sens_true, sens_xy=layout.xy, sens_near=layout.near)


def summarize(results):
    """Rows of metrics per run (per-snapshot means, time-mean, sensors) plus an overall row."""
    rows = []
    for r in results:
        snap = metrics_from_stats(r["snap_sse"], r["snap_sst"], r["snap_syy"], r["snap_n"])
        tmean = metrics_from_stats(r["tmean_sse"], r["tmean_sst"], r["tmean_syy"], r["tmean_n"])
        snapf = metrics_from_stats(r["snapfar_sse"], r["snapfar_sst"], r["snapfar_syy"], r["snapfar_n"])
        tmeanf = metrics_from_stats(r["tmeanfar_sse"], r["tmeanfar_sst"], r["tmeanfar_syy"], r["tmeanfar_n"])
        sp, st = r["sens_pred"].ravel(), r["sens_true"].ravel()
        sse, sst, syy, n = _stats(sp, st)
        sens = metrics_from_stats(sse, sst, syy, n)
        rows.append(dict(split=r.get("split", "test"), run=r["run"], stab=r["stab"], bearing=r["bearing"],
                         snap_rmse=float(np.nanmean(snap["rmse"])), snap_r2=float(np.nanmedian(snap["r2"])),
                         snap_rel_l2=float(np.nanmedian(snap["rel_l2"])),
                         tmean_rmse=float(np.nanmean(tmean["rmse"])), tmean_r2=float(np.nanmedian(tmean["r2"])),
                         tmean_rel_l2=float(np.nanmedian(tmean["rel_l2"])),
                         sensor_rmse=float(sens["rmse"]), sensor_r2=float(sens["r2"]),
                         far_snap_rel_l2=float(np.nanmedian(snapf["rel_l2"])), far_snap_r2=float(np.nanmedian(snapf["r2"])),
                         far_tmean_rel_l2=float(np.nanmedian(tmeanf["rel_l2"])), far_tmean_r2=float(np.nanmedian(tmeanf["r2"])),
                         log_snap_med_dlog10=float(np.nanmedian(r["snap_log_med"])),
                         log_snap_within_2x=float(np.nanmedian(r["snap_log_2x"])),
                         log_tmean_med_dlog10=float(np.nanmedian(r["tmean_log_med"])),
                         log_tmean_within_2x=float(np.nanmedian(r["tmean_log_2x"]))))
    # pooled rows per stability class and overall (means of the per-run values)
    for group in [*sorted({r["stab"] for r in rows}), "all"]:
        sub = [r for r in rows if group == "all" or r["stab"] == group]
        if len(sub) > 1:
            agg = dict(split=sub[0]["split"], run=f"mean_{group}", stab=group, bearing="all")
            for k in sub[0]:
                if k not in agg:
                    agg[k] = float(np.nanmean([r[k] for r in sub]))
            rows.append(agg)
    return rows


def save_results(results, out_dir: Path, tag="test"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        np.savez_compressed(out_dir / f"eval_{tag}_{r['run']}.npz", **{k: v for k, v in r.items()})
    rows = summarize(results)
    keys = list(rows[0])
    lines = [",".join(keys)] + [",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k])
                                         for k in keys) for row in rows]
    (out_dir / f"metrics_{tag}.csv").write_text("\n".join(lines) + "\n")
    return rows
