"""Raw FastEddy output -> one cropped HDF5 file per run (PLAN.md §1, §3, §8).

The crop is the same for every run: 200 x 200 columns (4 x 4 km) centred on the pit-2 source
centroid, and the lowest K terrain-following levels, where K is the first level whose height
above ground is >= 1000 m everywhere in the crop. Levels are kept native (no vertical
regridding); the height above ground of every cell is stored alongside.

Per run, for the 31 output frames t = 3600..7200 s, the file holds:

    c       (T, S, K, Ny, Nx)  pit-2 tracer mass concentration [kg m-3], S = 15 (AuxScalar_15..29)
    u v w   (T, K, Ny, Nx)     wind [m s-1] (FastEddy output velocities are plain, not rho-weighted)
    theta   (T, K, Ny, Nx)     potential temperature [K]   (needed for the virtual sensors)
    x, y    (Nx,), (Ny,)       cell-centre coordinates [m], same frame as the emission manifest
    z, z_agl, dV (K, Ny, Nx)   cell-centre height ASL / above ground [m], cell volume [m3]
    topo    (Ny, Nx)           terrain height [m ASL]
    time, step (T,)            model time [s], FE file step
    mass    (T, S)             crop-integrated tracer mass [kg] (quality check only)
    src_xy  (S, 2), src_scalar (S,)

The raw files are only read. Nothing is written next to them.
"""

import csv
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")   # Lustre (scratch, Oak) rejects flock

import h5py
import numpy as np

from methane_pinn import runs as R
from methane_pinn.paths import REPO_ROOT, les_root

MANIFEST = Path(os.environ.get(
    "PINN_EMISSION_MANIFEST",
    "/home/groups/abrandt/anm49/fasteddy/aus/data/04-emission_sources/muswellbrook/old/"
    "emission_20x20x5_10km_2pits_500kgph_1h_manifest.csv",
))

CROP_NX = 200
CROP_NY = 200
CROP_TOP_AGL = 1000.0     # m; keep levels up to the first one fully above this
WIND_VARS = ("u", "v", "w", "theta")


def read_sources(scalars=R.PIT2_SCALARS, path=MANIFEST):
    """(S, 2) source x, y [m] for the requested tracer indices, in that order."""
    xy = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            xy[int(row["scalar_index"])] = (float(row["xLoc_m"]), float(row["yLoc_m"]))
    return np.array([xy[i] for i in scalars], dtype=np.float64)


def fe_path(run: R.Run, step: int) -> Path:
    return les_root() / run.tree / run.run_dir / "output" / f"FE.{step}"


def crop_window(h5, src_xy):
    """Index window (i0, j0, K) of the crop. Deterministic given the grid and the sources."""
    x = h5["xPos"][0, 0, 0, :].astype(np.float64)
    y = h5["yPos"][0, 0, :, 0].astype(np.float64)
    cx, cy = src_xy.mean(axis=0)
    i0 = int(np.argmin(np.abs(x - cx))) - CROP_NX // 2
    j0 = int(np.argmin(np.abs(y - cy))) - CROP_NY // 2
    if i0 < 0 or j0 < 0 or i0 + CROP_NX > x.size or j0 + CROP_NY > y.size:
        raise ValueError("crop does not fit inside the LES domain")
    topo = h5["topoPos"][0, j0:j0 + CROP_NY, i0:i0 + CROP_NX]
    zc = h5["zPos"][0, :, j0:j0 + CROP_NY, i0:i0 + CROP_NX]
    agl_min = (zc - topo[None]).min(axis=(1, 2))
    above = np.nonzero(agl_min >= CROP_TOP_AGL)[0]
    if above.size == 0:
        raise ValueError(f"no level is fully above {CROP_TOP_AGL} m AGL")
    return i0, j0, int(above[0]) + 1


def cell_volumes(z_full_col, topo, nz, dx, dy):
    """Cell volumes for the lowest nz levels of a terrain-following grid.

    Interfaces are midway between cell centres; the bottom interface is the terrain
    (same convention as fire_safety_odor_analysis/scripts/fo_reduce.py).
    z_full_col must contain at least nz + 1 levels.
    """
    zf = np.empty((nz + 1,) + topo.shape, dtype=np.float64)
    zf[0] = topo
    zf[1:] = 0.5 * (z_full_col[:nz] + z_full_col[1:nz + 1])
    return (np.diff(zf, axis=0) * dx * dy).astype(np.float32)


def _git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def extract_run(run: R.Run, out_path: Path, times=None, force=False, log=print):
    """Extract one run to out_path (written to a temp name, then renamed)."""
    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists; pass force=True to overwrite")
    times = list(times) if times is not None else R.snapshot_times()
    steps = [R.step_of(t) for t in times]
    missing = [s for s in steps if not fe_path(run, s).exists()]
    if missing:
        raise FileNotFoundError(f"{run.name}: missing FE files for steps {missing[:5]}...")

    src_xy = read_sources()
    nS, nT = len(R.PIT2_SCALARS), len(steps)

    with h5py.File(fe_path(run, steps[0]), "r") as h:
        i0, j0, K = crop_window(h, src_xy)
        sl = (0, slice(0, K), slice(j0, j0 + CROP_NY), slice(i0, i0 + CROP_NX))
        x = h["xPos"][0, 0, 0, i0:i0 + CROP_NX].astype(np.float64)
        y = h["yPos"][0, 0, j0:j0 + CROP_NY, 0].astype(np.float64)
        topo = h["topoPos"][0, j0:j0 + CROP_NY, i0:i0 + CROP_NX].astype(np.float32)
        z_full = h["zPos"][0, :K + 1, j0:j0 + CROP_NY, i0:i0 + CROP_NX].astype(np.float64)
    dx, dy = float(np.diff(x).mean()), float(np.diff(y).mean())
    z = z_full[:K].astype(np.float32)
    dV = cell_volumes(z_full, topo.astype(np.float64), K, dx, dy)
    dV64 = dV.astype(np.float64)
    shape3 = (K, CROP_NY, CROP_NX)
    log(f"{run.name}: crop i0={i0} j0={j0} K={K} -> {shape3}, x {x[0]:.0f}-{x[-1]:.0f} m, "
        f"y {y[0]:.0f}-{y[-1]:.0f} m, {nT} frames t={times[0]:.0f}-{times[-1]:.0f} s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp{os.getpid()}")
    t_start = time.time()
    try:
        with h5py.File(tmp, "w") as o:
            ch3 = (1, K, 50, 50)
            dc = o.create_dataset("c", (nT, nS) + shape3, "f4", chunks=(1, 1) + ch3[1:],
                                  compression="lzf", shuffle=True)
            dc.attrs.update(units="kg m-3", long_name="pit-2 tracer mass concentration",
                            scalars=np.array(R.PIT2_SCALARS), q0_kgph=R.Q0_KGPH)
            dw = {v: o.create_dataset(v, (nT,) + shape3, "f4", chunks=ch3,
                                      compression="lzf", shuffle=True) for v in WIND_VARS}
            for v, u in zip(WIND_VARS, ("m s-1", "m s-1", "m s-1", "K")):
                dw[v].attrs["units"] = u
            mass = np.zeros((nT, nS))

            for it, step in enumerate(steps):
                tf = time.time()
                with h5py.File(fe_path(run, step), "r") as h:
                    t_file = float(h["time"][0])
                    if abs(t_file - times[it]) > 1e-3:
                        raise ValueError(f"FE.{step} has time {t_file}, expected {times[it]}")
                    for k, s in enumerate(R.PIT2_SCALARS):
                        a = h[f"AuxScalar_{s}"][sl]
                        dc[it, k] = a
                        mass[it, k] = float(np.dot(a.ravel().astype(np.float64), dV64.ravel()))
                    for v in WIND_VARS:
                        dw[v][it] = h[v][sl]
                expect = R.Q0_KGPH * max(times[it] - R.RELEASE_START_S, 0.0) / 3600.0
                log(f"  [{it + 1:2d}/{nT}] t={times[it]:6.0f} s  {time.time() - tf:5.1f} s  "
                    f"crop mass/emitted {mass[it].mean() / expect if expect else float('nan'):.3f} "
                    f"(range {mass[it].min():.2f}-{mass[it].max():.2f} kg)", flush=True)

            o["x"], o["y"] = x, y
            o["z"], o["z_agl"], o["dV"], o["topo"] = z, z - topo[None], dV, topo
            o["time"], o["step"] = np.asarray(times), np.asarray(steps, dtype=np.int64)
            o["mass"] = mass
            o["src_xy"], o["src_scalar"] = src_xy, np.array(R.PIT2_SCALARS)
            for k in ("x", "y", "z", "z_agl", "topo", "src_xy"):
                o[k].attrs["units"] = "m"
            o["dV"].attrs["units"], o["time"].attrs["units"], o["mass"].attrs["units"] = "m3", "s", "kg"
            o.attrs.update(
                run=run.name, stab=run.stab, bearing=run.bearing, split=run.split,
                tree=run.tree, run_dir=run.run_dir,
                source=str(les_root() / run.tree / run.run_dir / "output"),
                crop_i0=i0, crop_j0=j0, crop_K=K, crop_top_agl_m=CROP_TOP_AGL,
                dx=dx, dy=dy, q0_kgph=R.Q0_KGPH,
                release_start_s=R.RELEASE_START_S, release_end_s=R.RELEASE_END_S,
                manifest=str(MANIFEST), git=_git_hash(), host=socket.gethostname(),
                jobid=os.environ.get("SLURM_JOB_ID", ""),
                created_utc=datetime.now(timezone.utc).isoformat(),
            )
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    log(f"{run.name}: wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB) "
        f"in {(time.time() - t_start) / 60:.1f} min")
    return out_path
