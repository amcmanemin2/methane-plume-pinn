"""Training loop (PLAN.md §6; Jia 2025 eq. 4.6 + data term, R3 resampling).

Loss = w_data * L_data + w_pde * L_pde + w_bc * L_ground + w_ic * L_ic, with

    L_data   mean((c - c_LES)^2) / c_ref^2  (+ data_log * mean((log(c+eps) - log(c_LES+eps))^2))
    L_pde    mean((tau / c_ref * residual)^2)            residual of the advection-diffusion PDE
    L_ground mean((n . K grad c / (c_ref * U_ref))^2)    zero flux through the terrain
    L_ic     mean((c / c_ref)^2) at t = 3600 s

One epoch is one Adam step on the full point set (Jia's full-batch epoch). Collocation points are
kept in a pool and refreshed with R3 after every epoch: points whose |residual| exceeds the pool
mean are kept and the rest are redrawn.

Loss weights (loss_weights.auto = true): at the first epoch the gradient norm of each term on the
network parameters is measured, and w_i = |grad L_data| / |grad L_i| (so w_data = 1). The weights
are then frozen for the rest of the run and written to the log and the checkpoint.

Output bias (network.out_bias = auto): log(mean LES c on plume cells / c_ref), from the training
runs and training combinations only.
"""

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from methane_pinn import data as D
from methane_pinn.model import PlumeModel
from methane_pinn.physics import ground_flux_residual, pde_residual, source_term
from methane_pinn.sampling import STAB_INDEX, TrainSet, concat, select

TERMS = ("data", "pde", "bc", "ic")
VAL_SEED = 12345            # fixed validation points (val combinations of the training runs)
PDE_MONITOR_SEED = 54321    # fixed collocation points for an unbiased PDE-residual curve
BIAS_SEED = 777             # points used to set the output bias


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def all_seeds(cfg: dict) -> dict:
    """Every seed that determines a run (written to config.yaml)."""
    return dict(run=cfg["run"]["seed"], combination_splits=dict(D._SPLIT_SEED), val_points=VAL_SEED,
                pde_monitor=PDE_MONITOR_SEED, output_bias=BIAS_SEED, sensors=D.SENSOR_SEED)


def build_model(ts: TrainSet, cfg: dict) -> PlumeModel:
    net, phys = cfg["network"], cfg["physics"]
    out_bias = net.get("out_bias", "auto")
    return PlumeModel(
        ts.domain, ts.terrain, n_src=len(ts.src_xy), n_met=ts.met_t.shape[-1],
        met_mean=ts.met_mean, met_std=ts.met_std, q_max=D.Q_TOTAL_MAX_KGPH,
        c_ref=phys["c_ref"], n_classes=len(STAB_INDEX), K_init=tuple(phys["K_init"]),
        hidden_size=net["hidden_size"], num_layers=net["num_layers"],
        out_bias=0.0 if out_bias == "auto" else float(out_bias),
    ).to(ts.device)


@torch.no_grad()
def init_output_bias(model: PlumeModel, ts: TrainSet, n=200_000) -> float:
    """Set the last-layer bias to log(mean LES c on plume cells / c_ref) (training data only)."""
    pts = ts.data_points(n, np.random.default_rng(BIAS_SEED), p_plume=1.0, p_face=0.0)
    mean_c = float(pts["c"].clamp(min=0).mean())
    b = math.log(max(mean_c, 1e-30) / model.c_ref)
    model.net.network[-1].bias.fill_(b)
    return b


# ---------------------------------------------------------------------------------------------
# Loss terms (same structure as the class example: one function per term)
# ---------------------------------------------------------------------------------------------

def physics_loss(model, ts, pts, phys):
    """Scaled PDE residual at collocation points; returns (loss, |residual| per point)."""
    wind = ts.wind(pts["run"], *pts["X"].unbind(-1))
    K_h, K_z = model.K(pts["stab"])
    zagl = pts["X"][:, 2] - ts.terrain(pts["X"][:, 0], pts["X"][:, 1])
    s = source_term(pts["X"][:, 0], pts["X"][:, 1], zagl.detach(), pts["q"], ts.src_t,
                    phys["sigma_h"], phys["sigma_z"], ts.z_src)
    X = pts["X"].detach().clone().requires_grad_(True)
    r, _ = pde_residual(lambda Xi: model(Xi, pts["q"], pts["met"]), X, wind, K_h, K_z, s)
    r = r * (phys["tau"] / model.c_ref)
    return (r ** 2).mean(), r.detach().abs()


def boundary_loss(model, ts, pts, phys):
    """Zero diffusive flux through the terrain."""
    K_h, K_z = model.K(pts["stab"])
    X = pts["X"].detach().clone().requires_grad_(True)
    f = ground_flux_residual(lambda Xi: model(Xi, pts["q"], pts["met"]), X, ts.terrain, K_h, K_z)
    return ((f / (model.c_ref * phys["U_ref"])) ** 2).mean()


def initial_loss(model, pts):
    return ((model(pts["X"], pts["q"], pts["met"]) / model.c_ref) ** 2).mean()


def data_loss(model, pts, w_log=0.0, eps_rel=1e-3):
    c = model(pts["X"], pts["q"], pts["met"])
    loss = ((c - pts["c"]) / model.c_ref).pow(2).mean()
    if w_log > 0:
        eps = eps_rel * model.c_ref
        loss = loss + w_log * (torch.log(c + eps) - torch.log(pts["c"].clamp(min=0) + eps)).pow(2).mean()
    return loss


def balance_weights(model, terms: dict) -> dict:
    """w_i = |grad L_data| / |grad L_i| on the network parameters (computed once, then frozen)."""
    params = [p for p in model.net.parameters() if p.requires_grad]
    norms = {}
    for k, L in terms.items():
        g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        norms[k] = math.sqrt(sum(float((gi ** 2).sum()) for gi in g if gi is not None))
    ref = norms["data"]
    w = {k: (ref / n if n > 0 else 1.0) for k, n in norms.items()}
    return w, norms


# ---------------------------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate_points(model, pts, batch=65536):
    """RMSE [kg m-3], R^2 and relative L2 on a fixed point set."""
    pred = torch.cat([model(pts["X"][i:i + batch], pts["q"][i:i + batch], pts["met"][i:i + batch])
                      for i in range(0, len(pts["c"]), batch)])
    y = pts["c"]
    sse = ((pred - y) ** 2).sum().item()
    sst = ((y - y.mean()) ** 2).sum().item()
    return dict(rmse=math.sqrt(sse / len(y)), r2=1 - sse / sst, rel_l2=math.sqrt(sse / (y ** 2).sum().item()))


def pde_monitor(model, ts, pts, phys, batch=20000):
    """Mean scaled squared PDE residual on a fixed point set (not the R3 pool)."""
    tot = 0.0
    n = len(pts["X"])
    for i in range(0, n, batch):
        sub = {k: v[i:i + batch] for k, v in pts.items()}
        L, _ = physics_loss(model, ts, sub, phys)
        tot += float(L.detach()) * len(sub["X"])
    return tot / n


class _Timer:
    """Wall-clock per section (CUDA-synchronized) for the profiling epochs."""

    def __init__(self, device, on):
        self.on, self.cuda = on, torch.device(device).type == "cuda"
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                if timer.on and timer.cuda:
                    torch.cuda.synchronize()
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                if timer.on:
                    if timer.cuda:
                        torch.cuda.synchronize()
                    timer.t[name] = timer.t.get(name, 0.0) + time.perf_counter() - self.t0
        return _Ctx()


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

def train_pinn(ts: TrainSet, cfg: dict, out_dir: Path, log=print, fixed_data=None, data_only=False):
    """Train and checkpoint to out_dir (best.pt by validation RMSE, last.pt). Returns (model, losses).

    fixed_data: a point dict used as the full data batch at every epoch (overfit test).
    data_only: train on the data term alone (overfit test).
    """
    tr, phys, w_cfg = cfg["training"], cfg["physics"], cfg["loss_weights"]
    set_seeds(cfg["run"]["seed"])
    rng = np.random.default_rng(cfg["run"]["seed"])
    model = build_model(ts, cfg)
    info = {}
    if cfg["network"].get("out_bias", "auto") == "auto":
        info["out_bias"] = init_output_bias(model, ts)
        log(f"output bias set from training data: {info['out_bias']:.3f} "
            f"(c0 = {model.c_ref * math.exp(info['out_bias']):.3e} kg/m3)")
    opt = torch.optim.Adam(model.parameters(), lr=tr["learning_rate"])
    gamma = (tr["final_lr"] / tr["learning_rate"]) ** (1.0 / max(tr["steps"], 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    w_log = w_cfg.get("data_log", 0.0)
    weights = {k: float(w_cfg.get(k, 1.0)) for k in TERMS}
    auto_w = bool(w_cfg.get("auto", False)) and not data_only
    if data_only:
        weights = dict(data=1.0, pde=0.0, bc=0.0, ic=0.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    val = fixed_data if fixed_data is not None else \
        ts.data_points(tr["n_val_points"], np.random.default_rng(VAL_SEED), combos="val")
    mon = None if data_only else ts.collocation_points(tr.get("n_pde_monitor", 20000),
                                                       np.random.default_rng(PDE_MONITOR_SEED),
                                                       p_source=tr["p_source"])
    pool = None if data_only else ts.collocation_points(tr["n_collocation"], rng, p_source=tr["p_source"])

    losses = {k: [] for k in ("total", *TERMS)} | {"step": [], "val_step": [], "val_rmse": [], "val_r2": [],
                                                    "val_rel_l2": [], "pde_monitor": []}
    csv_path = out_dir / "losses.csv"
    K_names = [f"K{a}_{s}" for s in STAB_INDEX for a in ("h", "z")]
    header = ["step", "total", *TERMS, *[f"w{k}_{k}" for k in TERMS], "lr", "grad_norm", *K_names, "s_per_step"]
    fh = open(csv_path, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(header)

    best, t_start = float("inf"), time.time()
    prof_steps = range(2, 2 + tr.get("profile_steps", 5))
    timer = _Timer(ts.device, False)
    zero = torch.zeros((), device=ts.device)
    stopped_early = False
    for step in range(1, tr["steps"] + 1):
        timer.on = step in prof_steps
        if step == prof_steps.start and torch.device(ts.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        opt.zero_grad()
        with timer("sample"):
            dpts = fixed_data if fixed_data is not None else \
                ts.data_points(tr["n_data"], rng, p_plume=tr["p_plume"], p_face=tr["p_face"])
            if not data_only:
                gpts = ts.ground_points(tr["n_ground"], rng, tr["p_source"])
                ipts = ts.initial_points(tr["n_initial"], rng, tr["p_source"])
        with timer("forward"):
            terms = {"data": data_loss(model, dpts, w_log=w_log)}
            if not data_only:
                terms["pde"], r_abs = physics_loss(model, ts, pool, phys)
                terms["bc"] = boundary_loss(model, ts, gpts, phys)
                terms["ic"] = initial_loss(model, ipts)
            else:
                terms.update(pde=zero, bc=zero, ic=zero)
        if step == 1 and auto_w:
            weights, norms = balance_weights(model, terms)
            info["weights"], info["grad_norms_step1"] = weights, norms
            log("loss weights (frozen): " + ", ".join(f"{k}={v:.3e}" for k, v in weights.items()))
            log("grad norms at step 1: " + ", ".join(f"{k}={v:.3e}" for k, v in norms.items()))
        total = sum(weights[k] * terms[k] for k in TERMS)
        with timer("backward"):
            total.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("clip_grad") or float("inf"))
            opt.step()
        sched.step()
        with timer("r3"):
            if not data_only:      # R3 resampling (Jia Alg. 3, lines 23-26)
                keep = r_abs > r_abs.mean()
                new = ts.collocation_points(int((~keep).sum()), rng, p_source=tr["p_source"])
                pool = concat(select(pool, keep), new)

        vals = {k: float(terms[k]) for k in TERMS}
        tot = float(total)
        K = torch.exp(model.log_K).detach().cpu().numpy().ravel().tolist()
        el = time.time() - t_start
        writer.writerow([step, tot, *[vals[k] for k in TERMS], *[weights[k] * vals[k] for k in TERMS],
                         sched.get_last_lr()[0], float(gnorm), *K, el / step])
        losses["step"].append(step)
        losses["total"].append(tot)
        for k in TERMS:
            losses[k].append(vals[k])
        if not all(math.isfinite(v) for v in (tot, *vals.values())):
            log(f"step {step}: non-finite loss {vals}; stopping")
            break
        if step == prof_steps.stop - 1:
            info["profile_s_per_step"] = {k: v / len(prof_steps) for k, v in timer.t.items()}
            if torch.device(ts.device).type == "cuda":
                info["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
            log(f"profile (s/step): {json.dumps({k: round(v, 4) for k, v in info['profile_s_per_step'].items()})}"
                f"  peak GPU mem {info.get('peak_mem_GB', 0):.2f} GB", flush=True)
        if step % tr["log_every"] == 0 or step == 1:
            fh.flush()
            log(f"step {step:7d}  loss {tot:.3e}  " + "  ".join(f"{k} {vals[k]:.3e}" for k in TERMS)
                + f"  |g| {float(gnorm):.2e}  K {np.round(K, 3).tolist()}  lr {sched.get_last_lr()[0]:.2e}  "
                f"{el / step:.3f} s/step", flush=True)
        early = tr.get("early_stop_loss") and tot < tr["early_stop_loss"]
        if step % tr["val_every"] == 0 or step == tr["steps"] or early:
            m = evaluate_points(model, val)
            pm = float("nan") if mon is None else pde_monitor(model, ts, mon, phys)
            losses["val_step"].append(step)
            for k in ("rmse", "r2", "rel_l2"):
                losses[f"val_{k}"].append(m[k])
            losses["pde_monitor"].append(pm)
            log(f"  val  rmse {m['rmse']:.3e} kg/m3  R2 {m['r2']:.4f}  relL2 {m['rel_l2']:.3f}  "
                f"pde(fixed pts) {pm:.3e}", flush=True)
            state = dict(model=model.state_dict(), cfg=cfg, step=step, val=m, info=info,
                         runs=[r.name for r in ts.runs], domain=ts.domain)
            torch.save(state, out_dir / "last.pt")
            if m["rmse"] < best:
                best = m["rmse"]
                torch.save(state, out_dir / "best.pt")
            (out_dir / "losses.json").write_text(json.dumps(losses | {"info": info}))
        if early:
            log(f"early stop at step {step}: total loss {tot:.3e} < {tr['early_stop_loss']}")
            stopped_early = True
            break
    fh.close()
    info["stopped_early"] = stopped_early
    info["steps_run"] = step
    info["wall_s"] = time.time() - t_start
    (out_dir / "losses.json").write_text(json.dumps(losses | {"info": info}))
    return model, losses
