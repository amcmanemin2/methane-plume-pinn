#!/usr/bin/env python3
"""Sanity checks before any full training run (plan of 2026-11-04).

    python3 scripts/sanity.py                       # all checks, GPU recommended
    python3 scripts/sanity.py --only norm les       # a subset: norm, les, overfit, timing
    python3 scripts/sanity.py --only overfit --overfit-parts a --set training.clip_grad=null \
        --note "overfit 4a without gradient clipping"

Checks
    norm     input/output normalization and the size of each PDE term at initialization (units)
    les      PDE residual of the LES field itself by finite differences; NNLS fit of K_h, K_z
    overfit  (a) data only and (b) all terms, on 1 run x 10 combinations with a fixed point set
    timing   a few epochs at the full Jia point counts (100k per term): s/epoch and peak GPU memory

Writes $PINN_RUN_ROOT/<YYYYmmdd-HHMMSS>_sanity/report.json, plus overfit run dirs with loss curves.
Nothing here changes the configuration; the numbers are for review.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import append_runs_log, git_info, load_config  # noqa: E402

from methane_pinn import data as D  # noqa: E402
from methane_pinn import plots as P  # noqa: E402
from methane_pinn import runs as R  # noqa: E402
from methane_pinn.device import describe, resolve_device  # noqa: E402
from methane_pinn.diagnostics import fit_K, les_pde_terms, residual_with_K  # noqa: E402
from methane_pinn.evaluate import OVERFIT_PASS_REL_L2, far_field_mask, load_model, point_metrics  # noqa: E402
from methane_pinn.paths import run_dir  # noqa: E402
from methane_pinn.physics import source_term  # noqa: E402
from methane_pinn.sampling import TrainSet  # noqa: E402
from methane_pinn.train import build_model, evaluate_points, init_output_bias, train_pinn  # noqa: E402

OVERFIT_RUN = "daytime_N"
LES_RUNS = ("daytime_N", "convective_NE", "stable_N")       # one training run per class
LES_SOURCES = (0, 7, 14)
LES_FRAMES = (5, 15, 29)


def rng_stats(a):
    a = np.asarray(a, float)
    return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()), std=float(a.std()))


# ---------------------------------------------------------------------------------------------
def check_norm(ts, cfg, log):
    """Scaled inputs per point type, met normalization (train vs test), output and PDE terms at init."""
    out = {}
    model = build_model(ts, cfg)
    out["out_bias"] = init_output_bias(model, ts)
    rng = np.random.default_rng(0)
    n = 20000
    sets = dict(data=ts.data_points(n, rng), collocation=ts.collocation_points(n, rng, 0.2),
                ground=ts.ground_points(n, rng, 0.2), initial=ts.initial_points(n, rng, 0.2))
    for k, pts in sets.items():
        with torch.no_grad():
            sc = model.scaled_coords(pts["X"]).cpu().numpy()
            metn = ((pts["met"] - model.met_mean) / model.met_std).cpu().numpy()
            c = model(pts["X"], pts["q"], pts["met"]).cpu().numpy()
        out[f"scaled_{k}"] = {name: rng_stats(sc[:, i]) for i, name in enumerate(("x", "y", "z_agl", "t"))}
        out[f"q_over_qmax_{k}"] = rng_stats((pts["q"].sum(-1) / model.q_max).cpu().numpy())
        out[f"met_norm_{k}"] = dict(mean_of_means=float(metn.mean(0).mean()), mean_of_stds=float(metn.std(0).mean()))
        out[f"c_init_{k}"] = rng_stats(c)
    d = sets["data"]["c"].cpu().numpy()
    out["les_c_data_points"] = dict(**rng_stats(d), frac_zero=float((d <= 0).mean()), rms=float(np.sqrt((d ** 2).mean())),
                                    p50=float(np.percentile(d, 50)), p99=float(np.percentile(d, 99)))

    # met normalization from the 12 training runs, applied to the test runs (information only)
    lay = ts.layout
    tr = np.concatenate([D.sensor_series(D.LESRun.open(r), lay)["met"] for r in R.runs("train")])
    te = np.concatenate([D.sensor_series(D.LESRun.open(r), lay)["met"] for r in R.runs("test")])
    mu, sd = tr.mean(0), np.maximum(tr.std(0), 1e-3)
    out["met_train_norm"] = dict(mean=float(((tr - mu) / sd).mean()), std=float(((tr - mu) / sd).std(0).mean()))
    zt = (te - mu) / sd
    out["met_test_norm_with_train_stats"] = dict(mean=float(zt.mean()), mean_abs_channel_mean=float(np.abs(zt.mean(0)).mean()),
                                                 std=float(zt.std(0).mean()), max_abs=float(np.abs(zt).max()))

    # PDE terms at init, physical units [kg m-3 s-1] and scaled by tau / c_ref
    pts = sets["collocation"]
    X = pts["X"].detach().clone().requires_grad_(True)
    c = model(X, pts["q"], pts["met"])
    g = torch.autograd.grad(c.sum(), X, create_graph=True)[0]
    cxx = torch.autograd.grad(g[:, 0].sum(), X, retain_graph=True)[0][:, 0]
    cyy = torch.autograd.grad(g[:, 1].sum(), X, retain_graph=True)[0][:, 1]
    czz = torch.autograd.grad(g[:, 2].sum(), X, retain_graph=True)[0][:, 2]
    wind = ts.wind(pts["run"], *pts["X"].unbind(-1))
    Kh, Kz = model.K(pts["stab"])
    zagl = pts["X"][:, 2] - ts.terrain(pts["X"][:, 0], pts["X"][:, 1])
    s = source_term(pts["X"][:, 0], pts["X"][:, 1], zagl, pts["q"], ts.src_t, cfg["physics"]["sigma_h"],
                    cfg["physics"]["sigma_z"], ts.z_src)
    rms = lambda a: float(torch.sqrt((a.detach() ** 2).mean()))  # noqa: E731
    terms = dict(c_t=g[:, 3], adv=(wind * g[:, :3]).sum(-1), diff_h=Kh * (cxx + cyy), diff_z=Kz * czz, source=s)
    scale = cfg["physics"]["tau"] / model.c_ref
    out["pde_terms_init_rms"] = {k: dict(physical=rms(v), scaled=rms(v) * scale) for k, v in terms.items()}
    out["pde_terms_init_rms"]["source_near_sources_max"] = float(s.max())
    out["wind_rms_m_s"] = [float(v) for v in torch.sqrt((wind ** 2).mean(0))]
    log(json.dumps(out["pde_terms_init_rms"], indent=1))
    return out


# ---------------------------------------------------------------------------------------------
def check_les(cfg, log):
    """FD residual of the LES field and K fit per class (training runs only)."""
    out = {}
    Kh0, Kz0 = cfg["physics"]["K_init"]
    for name in LES_RUNS:
        run = R.get(name)
        assert run.split == "train"
        les = D.LESRun.open(run)
        t0 = time.time()
        terms = [les_pde_terms(les, s, it) for s in LES_SOURCES for it in LES_FRAMES]
        fit = fit_K(terms)
        fit["rel_residual_K_init"] = residual_with_K(terms, Kh0, Kz0)
        fit["rel_residual_K0"] = residual_with_K(terms, 0.0, 0.0)
        fit["rms_c_plume"] = float(np.sqrt(np.mean(np.concatenate([(t["c"][t["mask"]]) ** 2 for t in terms]))))
        out[name] = fit
        log(f"  LES FD {name}: K_h {fit['K_h']:.2f}, K_z {fit['K_z']:.3f} m2/s, rel. residual {fit['rel_residual']:.3f} "
            f"(K_init {fit['rel_residual_K_init']:.3f}, K=0 {fit['rel_residual_K0']:.3f}), "
            f"rms c_t {fit['rms_c_t']:.2e}, adv {fit['rms_adv']:.2e}, {time.time() - t0:.0f} s", flush=True)
    return out


# ---------------------------------------------------------------------------------------------
def check_overfit(ts, cfg, root, log, steps_a=5000, steps_b=1000, parts=("a", "b")):
    """(a) data only, (b) all terms; 10 training combinations of one run, fixed data points."""
    out = {}
    table = D.combination_table(ts.runs[0], "train")[:10]
    ts.restrict_combinations(table)
    fixed = ts.fixed_data_points(200_000, seed=99, p_plume=0.5, p_face=0.1)
    for tag, data_only, steps in (("a_data_only", True, steps_a), ("b_all_terms", False, steps_b)):
        if tag[0] not in parts:
            continue
        c = copy.deepcopy(cfg)
        c["training"].update(steps=steps, val_every=max(steps // 10, 1), log_every=max(steps // 20, 1),
                             early_stop_loss=None, n_collocation=10000, n_ground=10000, n_initial=10000,
                             n_pde_monitor=5000)
        d = root / f"overfit_{tag}"
        t0 = time.time()
        model, L = train_pinn(ts, c, d, log=log, fixed_data=fixed, data_only=data_only)
        info = json.loads((d / "losses.json").read_text())["info"]
        crit = {}
        for ck in ("last.pt", "best.pt"):
            mk, _ = load_model(d / ck, ts.device)
            crit[ck] = overfit_criterion(mk, ts, fixed)
        out[tag] = dict(steps=steps, wall_s=time.time() - t0, metrics=crit, best_val=min(L["val_rmse"]),
                        last_terms={k: L[k][-1] for k in ("data", "pde", "bc", "ic")}, info=info,
                        passed=bool(crit["best.pt"]["far_rel_l2"] < OVERFIT_PASS_REL_L2) if data_only else None)
        P.loss_curves(d, d / "loss_curves.png")
        b = crit["best.pt"]
        log(f"  overfit {tag} (best.pt): far-field relL2 {b['far_rel_l2']:.4f} (pass < {OVERFIT_PASS_REL_L2}), "
            f"all-cells relL2 {b['all_rel_l2']:.4f}, median |dlog10| {b['log_median_abs_dlog10']:.3f}, "
            f"within 2x {b['log_frac_within_2x']:.2%}  ({time.time() - t0:.0f} s)", flush=True)
    # the checkpoint carries the training-run met normalization
    first = root / f"overfit_{'a_data_only' if 'a' in parts else 'b_all_terms'}" / "best.pt"
    model, state = load_model(first, ts.device)
    out["ckpt_met_mean_matches_train"] = bool(np.allclose(model.met_mean.cpu().numpy(), ts.met_mean, rtol=1e-5))
    out["ckpt_runs"] = state["runs"]
    ts.q_pool = None
    ts.combos = [D.combination_table(r, "train") for r in ts.runs]
    return out


@torch.no_grad()
def overfit_criterion(model, ts, pts, batch=50000):
    """Revised overfit criterion (TRAINING_LOG.md D1/E4): far-field + log-space + all-cells metrics."""
    pred = torch.cat([model(pts["X"][i:i + batch], pts["q"][i:i + batch], pts["met"][i:i + batch])
                      for i in range(0, len(pts["c"]), batch)]).cpu().numpy()
    X = pts["X"].cpu().numpy()
    far = far_field_mask(X[:, 0], X[:, 1], ts.src_xy)
    return point_metrics(pred, pts["c"].cpu().numpy(), far)


def check_timing(ts, cfg, root, log):
    """A few epochs at the full point counts of configs/default.yaml (Jia: 100k per term)."""
    c = copy.deepcopy(cfg)
    c["training"].update(steps=6, profile_steps=4, val_every=10**9, log_every=1, early_stop_loss=None,
                         n_val_points=1000, n_pde_monitor=1000)
    try:
        _, _ = train_pinn(ts, c, root / "timing", log=log)
        info = json.loads((root / "timing" / "losses.json").read_text())["info"]
    except torch.cuda.OutOfMemoryError as e:
        return dict(oom=str(e)[:300])
    tot = sum(info["profile_s_per_step"].values())
    steps = cfg["training"]["steps"]
    return dict(points=dict(data=c["training"]["n_data"], collocation=c["training"]["n_collocation"],
                            ground=c["training"]["n_ground"], initial=c["training"]["n_initial"]),
                profile_s_per_step=info["profile_s_per_step"], s_per_epoch=tot, peak_mem_GB=info.get("peak_mem_GB"),
                estimate_hours_for_steps=dict(steps=steps, hours_1_run=tot * steps / 3600))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--only", nargs="*", default=["norm", "les", "overfit", "timing"])
    p.add_argument("--overfit-parts", nargs="*", default=["a", "b"], help="a = data only, b = all terms")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (YAML value)")
    p.add_argument("--note", default="", help="what changed relative to the previous run (logged)")
    p.add_argument("--name", default="sanity", help="suffix of the output directory")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg["run"].get("device"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = run_dir(f"{stamp}_{args.name}")
    git = git_info(root)
    report = dict(device=describe(device), config=args.config, overrides=args.set, note=args.note,
                  only=args.only, overfit_parts=args.overfit_parts, git=git["hash"], git_dirty=git["dirty"],
                  jobid=os.environ.get("SLURM_JOB_ID", ""))
    append_runs_log(dict(started=stamp, run_dir=root.name, jobid=report["jobid"], git=git["hash"],
                         dirty=git["dirty"], config=args.config, overrides=" ".join(args.set), note=args.note))

    def log(*a, **k):
        print(*a, **{kk: v for kk, v in k.items() if kk != "flush"}, flush=True)

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=1, default=float))

    print(f"sanity -> {root}", flush=True)
    if "les" in args.only:
        report["les_fd"] = check_les(cfg, log)
        save()
    if {"norm", "overfit", "timing"} & set(args.only):
        t0 = time.time()
        ts = TrainSet([R.get(OVERFIT_RUN)], stride=cfg["data"]["stride"], device=device,
                      sigma_h=cfg["physics"]["sigma_h"], sigma_z=cfg["physics"]["sigma_z"],
                      mask_thresh=cfg["data"].get("plume_thresh", 1e-8))
        report["load_s_one_run"] = time.time() - t0
        if "norm" in args.only:
            report["norm"] = check_norm(ts, cfg, log)
            save()
        if "overfit" in args.only:
            report["overfit"] = check_overfit(ts, cfg, root, log, parts=args.overfit_parts)
            save()
        if "timing" in args.only:
            report["timing_full_jia_sizes"] = check_timing(ts, cfg, root, log)
            log(json.dumps(report["timing_full_jia_sizes"], indent=1, default=float))
            save()
    save()
    print(f"wrote {root / 'report.json'}")


if __name__ == "__main__":
    main()
