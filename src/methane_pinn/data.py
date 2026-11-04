"""Extracted LES runs: superposition, source-combination sampling, virtual sensors (PLAN.md §3–§5).

Every extracted run file holds the 15 single-source pit-2 fields c_i (kg m-3, each at
q0 = 33.33 kg/h). Because methane is a passive tracer, the field for emission rates
q = (q_1..q_15) [kg/h] is exactly

    c(x, t; q) = sum_i (q_i / q0) c_i(x, t)

within one run, so augmented samples are never stored: only their rate vectors are.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np

from methane_pinn import runs as R
from methane_pinn.paths import data_root

N_SRC = len(R.PIT2_SCALARS)

# Augmentation (PLAN.md §4): K ~ U{1..15} active sources, total Q ~ U(0, 5000] kg/h, Dirichlet(1) split
Q_TOTAL_MAX_KGPH = 5000.0

# Combinations per split (PLAN.md §5)
N_COMBOS = {"train": 2000, "val": 200, "test": 100}
_SPLIT_SEED = {"train": 1, "val": 2, "test": 3}

# Virtual sensors (PLAN.md §3)
SENSOR_SEED = 2026
SENSOR_AGL_M = 10.0
N_SENSORS_NEAR, N_SENSORS_FAR = 10, 10
NEAR_RADIUS_M = 500.0
LIDAR_HEIGHTS_M = tuple(range(10, 111, 10))      # 10, 20, ..., 110 m AGL
LIDAR_SEARCH_RADIUS_M = 300.0


def les_dir() -> Path:
    return data_root() / "les"


def run_path(run: R.Run | str) -> Path:
    name = run if isinstance(run, str) else run.name
    return les_dir() / f"{name}.h5"


# ---------------------------------------------------------------------------------------------
# Source combinations
# ---------------------------------------------------------------------------------------------

def sample_combinations(n: int, rng: np.random.Generator, n_src: int = N_SRC,
                        q_max: float = Q_TOTAL_MAX_KGPH) -> np.ndarray:
    """(n, n_src) emission rates [kg/h].

    K ~ U{1..n_src} sources are switched on (a uniform random subset), the total rate is
    Q ~ U(0, q_max], and Q is split among the active sources with Dirichlet(1) weights.
    """
    k = rng.integers(1, n_src + 1, n)                                  # active sources per row
    keys = rng.random((n, n_src))
    kth = np.sort(keys, axis=1)[np.arange(n), k - 1]
    on = keys <= kth[:, None]                                          # uniform random subset of size k
    e = rng.exponential(size=(n, n_src)) * on                          # Dirichlet(1) = normalized Exp(1)
    total = q_max * (1.0 - rng.random(n))                              # (0, q_max]
    return total[:, None] * e / e.sum(axis=1, keepdims=True)


def combination_table(run: R.Run, split: str) -> np.ndarray:
    """Fixed, reproducible rate table for one run and split (seeded by run and split).

    train/val are drawn for training runs, test for test runs. Train and val use
    different seeds, so they are different combinations of the same 15 fields.
    """
    if (split == "test") != (run.split == "test"):
        raise ValueError(f"{run.name} is a {run.split} run; no {split!r} combinations for it")
    run_id = R.ALL_RUNS.index(run)
    rng = np.random.default_rng([_SPLIT_SEED[split], run_id])
    return sample_combinations(N_COMBOS[split], rng)


def superpose(basis: np.ndarray, q_kgph: np.ndarray, axis: int) -> np.ndarray:
    """sum_i (q_i/q0) basis_i along `axis` (the source axis of `basis`)."""
    w = np.asarray(q_kgph, dtype=np.float64) / R.Q0_KGPH
    return np.tensordot(np.moveaxis(basis, axis, -1), w, axes=([-1], [0])).astype(basis.dtype)


# ---------------------------------------------------------------------------------------------
# One extracted run
# ---------------------------------------------------------------------------------------------

@dataclass
class LESRun:
    """Read access to one extracted run file. Arrays are (T, [S,] K, Ny, Nx)."""

    path: Path

    @classmethod
    def open(cls, run: R.Run | str) -> "LESRun":
        return cls(run_path(run))

    def _h5(self):
        return h5py.File(self.path, "r")

    @cached_property
    def attrs(self) -> dict:
        with self._h5() as h:
            return dict(h.attrs)

    @cached_property
    def grid(self) -> dict:
        """Static fields: x, y, z, z_agl, dV, topo, time, src_xy."""
        with self._h5() as h:
            return {k: h[k][:] for k in ("x", "y", "z", "z_agl", "dV", "topo", "time", "src_xy")}

    @property
    def shape(self) -> tuple:
        with self._h5() as h:
            return h["c"].shape          # (T, S, K, Ny, Nx)

    def basis(self, it, sl=np.s_[:, :, :]) -> np.ndarray:
        """Single-source fields at frame(s) it, (S, ...) or (T, S, ...). sl indexes (K, Ny, Nx)."""
        with self._h5() as h:
            return h["c"][(it, slice(None)) + tuple(sl)]

    def field(self, it, q_kgph, sl=np.s_[:, :, :]) -> np.ndarray:
        """Concentration [kg m-3] for emission rates q at frame(s) it."""
        b = self.basis(it, sl)
        single = isinstance(it, (int, np.integer))
        return superpose(b, q_kgph, axis=0 if single else 1)

    def wind(self, var: str, it=slice(None), sl=np.s_[:, :, :]) -> np.ndarray:
        with self._h5() as h:
            return h[var][(it,) + tuple(sl)]

    def source_cells(self) -> np.ndarray:
        """(S, 2) column indices (j, i) of the source cells."""
        g = self.grid
        i = np.abs(g["x"][None, :] - g["src_xy"][:, :1]).argmin(axis=1)
        j = np.abs(g["y"][None, :] - g["src_xy"][:, 1:]).argmin(axis=1)
        return np.stack([j, i], axis=1)


# ---------------------------------------------------------------------------------------------
# Virtual sensors (same locations in every run)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorLayout:
    cols: np.ndarray        # (20, 2) column indices (j, i) of the point sensors
    near: np.ndarray        # (20,) bool, True for the 10 sensors within 500 m of a source
    lidar_col: tuple        # (j, i) of the wind-profile sensor
    xy: np.ndarray          # (20, 2) sensor x, y [m]
    lidar_xy: tuple

    @property
    def n_inputs(self) -> int:
        return 4 * len(self.cols) + 3 * len(LIDAR_HEIGHTS_M)     # 80 + 33 = 113


def sensor_layout(run: LESRun, seed: int = SENSOR_SEED) -> SensorLayout:
    """10 random columns within 500 m of a pit-2 source, 10 random columns farther away, and
    the lidar on the lowest terrain cell within 300 m of the source centroid. Source cells are
    excluded. Depends only on the (shared) grid, so it is identical for every run."""
    g = run.grid
    X, Y = np.meshgrid(g["x"], g["y"])                     # (Ny, Nx)
    d = np.min(np.hypot(X[..., None] - g["src_xy"][:, 0], Y[..., None] - g["src_xy"][:, 1]), axis=-1)
    free = np.ones_like(d, dtype=bool)
    for j, i in run.source_cells():
        free[j, i] = False
    free[[0, -1], :] = False                               # keep off the crop faces
    free[:, [0, -1]] = False

    rng = np.random.default_rng(seed)
    near_idx = np.argwhere(free & (d <= NEAR_RADIUS_M))
    far_idx = np.argwhere(free & (d > NEAR_RADIUS_M))
    cols = np.concatenate([near_idx[rng.choice(len(near_idx), N_SENSORS_NEAR, replace=False)],
                           far_idx[rng.choice(len(far_idx), N_SENSORS_FAR, replace=False)]])
    near = np.r_[np.ones(N_SENSORS_NEAR, bool), np.zeros(N_SENSORS_FAR, bool)]

    cx, cy = g["src_xy"].mean(axis=0)
    cand = free & (np.hypot(X - cx, Y - cy) <= LIDAR_SEARCH_RADIUS_M)
    topo = np.where(cand, g["topo"], np.inf)
    lj, li = np.unravel_index(np.argmin(topo), topo.shape)
    xy = np.stack([g["x"][cols[:, 1]], g["y"][cols[:, 0]]], axis=1)
    return SensorLayout(cols, near, (int(lj), int(li)), xy, (float(g["x"][li]), float(g["y"][lj])))


def _interp_column(prof: np.ndarray, zagl: np.ndarray, heights) -> np.ndarray:
    """Linear interpolation of (T, K) profiles at heights AGL (zagl is (K,), increasing)."""
    heights = np.atleast_1d(np.asarray(heights, float))
    k = np.clip(np.searchsorted(zagl, heights) - 1, 0, len(zagl) - 2)
    w = (heights - zagl[k]) / (zagl[k + 1] - zagl[k])
    return prof[:, k] * (1 - w) + prof[:, k + 1] * w      # (T, len(heights))


def sensor_series(run: LESRun, layout: SensorLayout) -> dict:
    """Met inputs per frame.

    Returns {'met': (T, 113) array, 'names': [...]} with, for each of the 20 point sensors,
    u, v, w, theta at 10 m AGL, followed by u, v, w of the lidar at 10..110 m AGL.
    """
    za = run.grid["z_agl"]
    cols, names = [], []
    with h5py.File(run.path, "r") as h:
        kmax = int(np.searchsorted(za.min(axis=(1, 2)), max(LIDAR_HEIGHTS_M))) + 2
        for n, (j, i) in enumerate(layout.cols):
            for v in ("u", "v", "w", "theta"):
                prof = h[v][:, :kmax, j, i]
                cols.append(_interp_column(prof, za[:kmax, j, i], SENSOR_AGL_M)[:, 0])
                names.append(f"s{n:02d}_{v}")
        lj, li = layout.lidar_col
        for v in ("u", "v", "w"):
            prof = h[v][:, :kmax, lj, li]
            vals = _interp_column(prof, za[:kmax, lj, li], LIDAR_HEIGHTS_M)
            for m, z in enumerate(LIDAR_HEIGHTS_M):
                cols.append(vals[:, m])
                names.append(f"lidar_{v}_{z:03d}m")
    return {"met": np.stack(cols, axis=1).astype(np.float32), "names": names}
