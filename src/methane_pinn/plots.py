"""Report figures (static PNG, 200 dpi, light surface).

Colors follow the dataviz reference palette:
- stability classes = categorical slots 1-3 (blue, orange, aqua; validated all-pairs);
  lines carry direct labels because aqua is below 3:1 contrast on the surface
- concentration = one-hue blue sequential ramp on a log scale
- PINN - LES error = blue <-> red diverging ramp with a gray midpoint
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = {"daytime": "#2a78d6", "convective": "#eb6834", "stable": "#1baf7a"}
CLASS_LABELS = {"daytime": "Daytime (6 m/s)", "convective": "Convective (3 m/s)", "stable": "Stable F (1.2 m/s)"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
                                                         "#f2a6a5", "#e34948", "#9b2222"])


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
        "axes.titlesize": 10, "axes.titlecolor": INK, "lines.linewidth": 2.0, "legend.frameon": False,
    })


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def loss_curves(run_dir, path, smooth=50):
    """One panel per loss term (raw and weighted; separate axes, log scale), plus LR, val R2 and the
    PDE residual on fixed points. Reads losses.csv (every epoch) and losses.json (validation)."""
    _style()
    run_dir = Path(run_dir)
    L = np.genfromtxt(run_dir / "losses.csv", delimiter=",", names=True)
    V = json.loads((run_dir / "losses.json").read_text())
    terms = [("data", "Data"), ("pde", "PDE residual (R3 pool)"), ("bc", "Ground flux"), ("ic", "Initial condition")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.4))
    step = np.atleast_1d(L["step"])

    def sm(y):
        y = np.atleast_1d(y)
        if len(y) <= smooth:
            return y
        return np.convolve(y, np.ones(smooth) / smooth, mode="same")

    for ax, (k, title) in zip(axes[0], terms):
        raw, wtd = np.atleast_1d(L[k]), np.atleast_1d(L[f"w{k}_{k}"])
        ax.plot(step, raw, color=GRID, lw=0.8)
        ax.plot(step, sm(raw), color=CLASS_COLORS["daytime"], label="raw")
        ax.plot(step, sm(wtd), color=CLASS_COLORS["convective"], label="weighted")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("epoch")
    axes[0, 0].legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(step, np.atleast_1d(L["total"]), color=GRID, lw=0.8)
    ax.plot(step, sm(L["total"]), color=CLASS_COLORS["daytime"])
    ax.set_yscale("log"); ax.set_title("Total (weighted)", loc="left"); ax.set_xlabel("epoch")
    ax = axes[1, 1]
    ax.plot(step, np.atleast_1d(L["lr"]), color=CLASS_COLORS["daytime"])
    ax.set_yscale("log"); ax.set_title("Learning rate", loc="left"); ax.set_xlabel("epoch")
    ax = axes[1, 2]
    if V.get("val_step"):
        ax.plot(V["val_step"], V["val_r2"], color=CLASS_COLORS["daytime"], marker="o", markersize=3)
    ax.set_title("Validation R²", loc="left"); ax.set_xlabel("epoch")
    ax = axes[1, 3]
    pm = np.array(V.get("pde_monitor", []), float)
    if np.isfinite(pm).any():
        ax.plot(V["val_step"], pm, color=CLASS_COLORS["daytime"], marker="o", markersize=3)
        ax.set_yscale("log")
    ax.set_title("PDE residual (fixed points)", loc="left"); ax.set_xlabel("epoch")
    fig.tight_layout()
    return _save(fig, path)


def field_parity(results, path, title, floor=1e-10):
    """PINN vs LES at random grid cells (all combinations and snapshots), log-log hexbin."""
    _style()
    t = np.concatenate([r["par_true"].ravel() for r in results])
    p = np.concatenate([r["par_pred"].ravel() for r in results])
    keep = t > floor
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    if keep.any():
        hb = ax.hexbin(np.log10(t[keep]), np.log10(np.maximum(p[keep], floor)), gridsize=60, cmap=SEQ,
                       bins="log", mincnt=1)
        fig.colorbar(hb, ax=ax, label="cells")
        lo, hi = np.log10(floor), np.log10(max(t[keep].max(), p[keep].max()))
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0, ls="--")
    ax.set_xlabel("log₁₀ LES CH₄ (kg m⁻³)")
    ax.set_ylabel("log₁₀ PINN CH₄ (kg m⁻³)")
    ax.set_title(f"{title} (cells with LES > {floor:.0e}: {keep.mean():.1%})", loc="left", fontsize=9)
    ax.grid(False)
    return _save(fig, path)


def metrics_vs_time(results, path, metric_fn):
    """Median R² and mean RMSE across test combinations vs time, one line per stability class."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for stab, color in CLASS_COLORS.items():
        rs = [r for r in results if r["stab"] == stab]
        if not rs:
            continue
        t = (rs[0]["times"] - 3600.0) / 60.0
        m = [metric_fn(r["snap_sse"], r["snap_sst"], r["snap_syy"], r["snap_n"]) for r in rs]
        r2 = np.nanmedian(np.concatenate([x["r2"] for x in m]), axis=0)
        rmse = np.nanmean(np.concatenate([x["rmse"] for x in m]), axis=0)
        for ax, y in zip(axes, (r2, rmse)):
            ax.plot(t, y, color=color, label=CLASS_LABELS[stab])
            ax.annotate(CLASS_LABELS[stab].split(" (")[0], (t[-1], y[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)
    axes[0].set_title("R² vs LES (median over test combinations)", loc="left")
    axes[0].set_ylabel("R²")
    axes[1].set_title("RMSE vs LES (mean over test combinations)", loc="left")
    axes[1].set_ylabel("RMSE (kg m⁻³)")
    for ax in axes:
        ax.set_xlabel("minutes since release start")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    return _save(fig, path)


def slice_maps(x, y, les, pinn, src_xy, path, title, sensors_xy=None, lidar_xy=None, vmin_frac=1e-4):
    """Rows of LES | PINN | PINN - LES maps. les/pinn: lists of (Ny, Nx) fields, one row each."""
    _style()
    rows = len(les)
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.7 * rows), squeeze=False)
    cx, cy = src_xy.mean(0)
    ext = [(x[0] - cx) / 1e3, (x[-1] - cx) / 1e3, (y[0] - cy) / 1e3, (y[-1] - cy) / 1e3]
    for r, (a, b, label) in enumerate(zip(les, pinn, title)):
        vmax = max(np.nanmax(a), np.nanmax(b))
        norm = LogNorm(vmin=vmax * vmin_frac, vmax=vmax)
        for ax, f, name in zip(axes[r, :2], (a, b), ("LES", "PINN")):
            im = ax.imshow(np.clip(f, vmax * vmin_frac, None), origin="lower", extent=ext, cmap=SEQ, norm=norm)
            ax.set_title(f"{name} · {label}", loc="left")
        fig.colorbar(im, ax=axes[r, :2], shrink=0.85, label="CH₄ (kg m⁻³)")
        err = b - a
        lim = max(np.nanpercentile(np.abs(err), 99.0), 1e-30)      # 99th pct: the source cell would swamp the scale
        ime = axes[r, 2].imshow(np.clip(err, -lim, lim), origin="lower", extent=ext, cmap=DIV,
                                norm=TwoSlopeNorm(0, -lim, lim))
        axes[r, 2].set_title(f"PINN − LES · {label}", loc="left")
        fig.colorbar(ime, ax=axes[r, 2], shrink=0.85, label="kg m⁻³ (clipped at 99th pct)", extend="both")
        for ax in axes[r]:
            ax.grid(False)
            ax.plot((src_xy[:, 0] - cx) / 1e3, (src_xy[:, 1] - cy) / 1e3, "o", ms=3, mfc=INK, mec=SURFACE, mew=0.5)
            if sensors_xy is not None:
                ax.plot((sensors_xy[:, 0] - cx) / 1e3, (sensors_xy[:, 1] - cy) / 1e3, "^", ms=5,
                        mfc="none", mec=INK2, mew=0.8)
            if lidar_xy is not None:
                ax.plot((lidar_xy[0] - cx) / 1e3, (lidar_xy[1] - cy) / 1e3, "s", ms=6, mfc="none", mec=INK, mew=1.0)
            ax.set_xlabel("x from pit-2 centre (km)")
            ax.set_ylabel("y (km)")
    return _save(fig, path)


def sensor_parity(results, path, floor=1e-9, title="Sensors, 10 m AGL (test runs)"):
    """PINN vs LES methane at the 20 sensors (all test combinations and snapshots), log-log."""
    _style()
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lo, hi = np.inf, 0
    for stab, color in CLASS_COLORS.items():
        rs = [r for r in results if r["stab"] == stab]
        if not rs:
            continue
        t = np.concatenate([r["sens_true"].ravel() for r in rs])
        p = np.concatenate([r["sens_pred"].ravel() for r in rs])
        keep = t > floor
        ax.scatter(t[keep], np.maximum(p[keep], floor), s=8, alpha=0.35, color=color, linewidths=0,
                   label=CLASS_LABELS[stab], rasterized=True)
        if keep.any():
            lo, hi = min(lo, t[keep].min()), max(hi, t[keep].max(), p[keep].max())
    if hi > 0:
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1.0, ls="--")
        ax.annotate("1:1", (hi, hi), xytext=(-18, -4), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("LES CH₄ at sensor (kg m⁻³)")
    ax.set_ylabel("PINN CH₄ (kg m⁻³)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8, markerscale=2)
    return _save(fig, path)
