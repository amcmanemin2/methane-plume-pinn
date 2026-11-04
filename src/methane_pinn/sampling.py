"""Training points for the PINN (PLAN.md §4–§6).

TrainSet holds the training runs in host RAM at a horizontal stride (default every 2nd column,
40 m) and produces four kinds of points:

    data         LES concentration at random grid cells of a training combination (superposed on
                 the fly); a share is drawn from the plume and from the crop faces (the Dirichlet BC)
    collocation  random points in the crop with freshly sampled source rates (PDE residual);
                 a share is placed around active sources, where the source term lives
    ground       random points on the terrain (zero-flux condition)
    initial      random points at t = 3600 s (c = 0)

Each point carries its run index (for the wind and met inputs), its stability class, its rates
q (S,) [kg/h] and its met vector m(t) (M,), linearly interpolated between the 2-min frames.
"""

import time

import h5py
import numpy as np
import torch

from methane_pinn import data as D
from methane_pinn import runs as R
from methane_pinn.physics import Terrain, WindField, model_top_and_eta

STAB_INDEX = {s: i for i, s in enumerate(R.TREES)}


class TrainSet:
    def __init__(self, run_list, stride=2, device="cpu", mask_thresh=1e-9, log=print,
                 sigma_h=10.0, sigma_z=5.0):
        self.runs = list(run_list)
        bad = [r.name for r in self.runs if r.split != "train"]
        if bad:
            raise ValueError(f"TrainSet only accepts training runs; got test runs {bad}")
        self.device = device
        self.q_pool = None              # if set, PDE/ground/IC rates are drawn from these rows
        self.sigma_h, self.sigma_z = sigma_h, sigma_z
        les = [D.LESRun.open(r) for r in self.runs]
        g = les[0].grid
        self.layout = D.sensor_layout(les[0])
        s = stride
        self.x, self.y = g["x"][::s], g["y"][::s]
        self.z = g["z"][:, ::s, ::s].astype(np.float32)               # (K, ny, nx)
        self.times = g["time"].astype(np.float64)
        self.src_xy = g["src_xy"].astype(np.float32)
        self.K, self.ny, self.nx = self.z.shape
        self.T = len(self.times)
        self.domain = dict(x0=float(g["x"][0]), y0=float(g["y"][0]), t0=float(self.times[0]),
                           Lx=float(g["x"][-1] - g["x"][0]), Ly=float(g["y"][-1] - g["y"][0]),
                           Lz=float(les[0].attrs["crop_top_agl_m"]), T=float(self.times[-1] - self.times[0]))
        self.z_src = float(g["z_agl"][0].mean())                      # first-level height AGL
        self.terrain = Terrain(g["x"], g["y"], g["topo"], device=device)
        H, eta = model_top_and_eta(g["z_agl"], g["topo"])

        self.basis, self.met, self.combos, self.plume_idx = [], [], [], []
        uvw = np.empty((len(les), self.T, self.K, self.ny, self.nx, 3), np.float16)
        for n, (run, lr) in enumerate(zip(self.runs, les)):
            t0 = time.time()
            if not np.allclose(lr.grid["z"], g["z"]):
                raise ValueError(f"{run.name}: grid differs from {self.runs[0].name}")
            with h5py.File(lr.path, "r") as h:
                b = np.stack([h["c"][it, :, :, ::s, ::s] for it in range(self.T)])   # (T, S, K, ny, nx)
                for c, v in enumerate(("u", "v", "w")):
                    uvw[n, ..., c] = np.stack([h[v][it, :, ::s, ::s] for it in range(self.T)])
            self.basis.append(b)
            self.plume_idx.append(np.flatnonzero(b.max(axis=(0, 1)).ravel() > mask_thresh))
            self.met.append(D.sensor_series(lr, self.layout)["met"])
            self.combos.append(D.combination_table(run, "train"))
            log(f"  loaded {run.name}: basis {b.nbytes / 1e9:.2f} GB, plume cells "
                f"{len(self.plume_idx[-1])}/{b[0, 0].size}, {time.time() - t0:.0f} s")
        self.wind = WindField(uvw, self.x, self.y, eta, H, self.times, self.terrain, device=device)
        del uvw
        self.stab = np.array([STAB_INDEX[r.stab] for r in self.runs])
        met = np.concatenate(self.met)
        self.met_mean, self.met_std = met.mean(0), np.maximum(met.std(0), 1e-3)
        self.met_t = torch.as_tensor(np.stack(self.met), device=device)            # (R, T, M)
        self.src_t = torch.as_tensor(self.src_xy, device=device)
        self.stab_t = torch.as_tensor(self.stab, device=device)
        k, j, i = np.meshgrid(np.arange(self.K), np.arange(self.ny), np.arange(self.nx), indexing="ij")
        face = (i == 0) | (i == self.nx - 1) | (j == 0) | (j == self.ny - 1) | (k == self.K - 1)
        self.face_idx = np.flatnonzero(face.ravel())

    def restrict_combinations(self, table: np.ndarray):
        """Use only these rate rows for data and physics points (overfit test)."""
        self.q_pool = np.asarray(table, float)
        self.combos = [self.q_pool for _ in self.runs]

    def fixed_data_points(self, n, seed, combos="train", p_plume=0.5, p_face=0.1):
        """A reproducible data point set (drawn once with its own seed)."""
        return self.data_points(n, np.random.default_rng(seed), combos=combos, p_plume=p_plume, p_face=p_face)

    # ---------------------------------------------------------------------------------------------
    def _met_at(self, run, t):
        f = ((t - self.domain["t0"]) / (self.times[1] - self.times[0])).clamp(0, self.T - 1 - 1e-4)
        n = torch.floor(f).long()
        w = (f - n).unsqueeze(-1)
        return self.met_t[run, n] * (1 - w) + self.met_t[run, n + 1] * w

    def _to(self, *arrs):
        return [torch.as_tensor(a, device=self.device, dtype=torch.float32) for a in arrs]

    def data_points(self, n, rng, combos="train", p_plume=0.5, p_face=0.1, frames=None):
        """LES targets at grid cells. combos: 'train', 'val' or a list of (R,) rate tables."""
        tables = self.combos if combos == "train" else \
            [D.combination_table(r, "val") for r in self.runs] if combos == "val" else combos
        run = rng.integers(0, len(self.runs), n)
        frames = np.arange(self.T) if frames is None else np.asarray(frames)
        it = rng.choice(frames, n)
        u = rng.random(n)
        cell = rng.integers(0, self.K * self.ny * self.nx, n)
        cell = np.where(u < p_face, rng.choice(self.face_idx, n), cell)
        q = np.empty((n, len(self.src_xy)))
        c = np.empty(n)
        for r in range(len(self.runs)):
            m = run == r
            if not m.any():
                continue
            pl = (u[m] >= p_face) & (u[m] < p_face + p_plume)
            cr = cell[m]
            cr[pl] = rng.choice(self.plume_idx[r], pl.sum())
            cell[m] = cr
            qr = tables[r][rng.integers(0, len(tables[r]), m.sum())]
            k, j, i = np.unravel_index(cr, (self.K, self.ny, self.nx))
            b = self.basis[r][it[m], :, k, j, i]                          # (n_r, S)
            c[m] = (b * qr).sum(-1) / R.Q0_KGPH
            q[m] = qr
        k, j, i = np.unravel_index(cell, (self.K, self.ny, self.nx))
        X = np.stack([self.x[i], self.y[j], self.z[k, j, i], self.times[it]], axis=1)
        X, q, c = self._to(X, q, c)
        run_t = torch.as_tensor(run, device=self.device)
        return dict(X=X, q=q, c=c, run=run_t, met=self._met_at(run_t, X[:, 3]), stab=self.stab_t[run_t])

    def _random_points(self, n, rng, on_ground=False, t=None, p_source=0.0):
        """Uniform points in the crop (optionally a share placed around active sources)."""
        d = self.domain
        q = D.sample_combinations(n, rng) if self.q_pool is None else \
            self.q_pool[rng.integers(0, len(self.q_pool), n)]
        x = d["x0"] + d["Lx"] * rng.random(n)
        y = d["y0"] + d["Ly"] * rng.random(n)
        zagl = np.zeros(n) if on_ground else d["Lz"] * rng.random(n)
        near = rng.random(n) < p_source
        if near.any():
            k = near.sum()
            p = q[near] / q[near].sum(1, keepdims=True)
            src = (p.cumsum(1) > rng.random((k, 1))).argmax(1)               # source ~ its share of q
            x[near] = self.src_xy[src, 0] + rng.normal(0, 2 * self.sigma_h, k)
            y[near] = self.src_xy[src, 1] + rng.normal(0, 2 * self.sigma_h, k)
            if not on_ground:
                zagl[near] = np.abs(rng.normal(0, 3 * self.sigma_z, k))
        x = np.clip(x, d["x0"], d["x0"] + d["Lx"])
        y = np.clip(y, d["y0"], d["y0"] + d["Ly"])
        tt = np.full(n, d["t0"]) if t is not None else d["t0"] + d["T"] * rng.random(n)
        run = rng.integers(0, len(self.runs), n)
        x_t, y_t, zagl_t, tt_t, q_t = self._to(x, y, zagl, tt, q)
        z_t = zagl_t + self.terrain(x_t, y_t)
        X = torch.stack([x_t, y_t, z_t, tt_t], dim=1)
        run_t = torch.as_tensor(run, device=self.device)
        return dict(X=X, q=q_t, run=run_t, met=self._met_at(run_t, tt_t), stab=self.stab_t[run_t])

    def collocation_points(self, n, rng, p_source=0.2):
        return self._random_points(n, rng, p_source=p_source)

    def ground_points(self, n, rng, p_source=0.2):
        return self._random_points(n, rng, on_ground=True, p_source=p_source)

    def initial_points(self, n, rng, p_source=0.2):
        return self._random_points(n, rng, t=self.domain["t0"], p_source=p_source)


def concat(a: dict, b: dict) -> dict:
    return {k: torch.cat([a[k], b[k]]) for k in a}


def select(a: dict, mask) -> dict:
    return {k: v[mask] for k, v in a.items()}
