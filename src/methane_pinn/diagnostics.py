"""Finite-difference check of the PINN's PDE on the LES field itself (sanity check 3).

The LES grid is terrain-following: column (j, i), level k, with physical height z[k, j, i]. A
derivative at fixed physical height follows from the chain rule in index space:

    dc/dz       = (dc/dk) / (dz/dk)
    dc/dx|_z    = dc/dx|_k - (dz/dx|_k) * dc/dz          (same for y)

Second derivatives apply the same operators to the first derivatives. All differences are second
order central in the interior (np.gradient). dc/dt is central over two output intervals (240 s).

From these, for one source at a time, we form

    A = c_t + u c_x + v c_y + w c_z          (what the PINN's PDE says must equal K-diffusion)
    L_h = c_xx + c_yy,   L_z = c_zz

and fit K_h, K_z >= 0 minimizing |A - K_h L_h - K_z L_z| on plume cells away from the source.
The relative residual |r| / |A| is how much of the LES the constant-K advection-diffusion equation
with 2-min wind snapshots cannot explain: a floor for the PDE loss, and a measure of how much the
PDE term conflicts with the data term.
"""

import h5py
import numpy as np
from scipy.optimize import nnls

from methane_pinn import data as D


class TFGrid:
    """Physical-coordinate derivatives on a terrain-following grid with heights z (K, Ny, Nx)."""

    def __init__(self, x, y, z):
        self.dx, self.dy = float(x[1] - x[0]), float(y[1] - y[0])
        self.z_k = np.gradient(z, axis=0)                     # dz/dk
        self.z_x = np.gradient(z, self.dx, axis=2)            # dz/dx at fixed k
        self.z_y = np.gradient(z, self.dy, axis=1)

    def d_z(self, f):
        return np.gradient(f, axis=0) / self.z_k

    def d_x(self, f, fz=None):
        fz = self.d_z(f) if fz is None else fz
        return np.gradient(f, self.dx, axis=2) - self.z_x * fz

    def d_y(self, f, fz=None):
        fz = self.d_z(f) if fz is None else fz
        return np.gradient(f, self.dy, axis=1) - self.z_y * fz


def les_pde_terms(les: D.LESRun, src: int, it: int):
    """A, L_h, L_z, c and a validity mask on the full crop for source index src at frame it."""
    g = les.grid
    grid = TFGrid(g["x"], g["y"], g["z"].astype(np.float64))
    dt = float(g["time"][it + 1] - g["time"][it - 1])
    with h5py.File(les.path, "r") as h:
        c3 = h["c"][it - 1:it + 2, src].astype(np.float64)    # (3, K, Ny, Nx)
        u, v, w = (h[n][it].astype(np.float64) for n in ("u", "v", "w"))
    c = c3[1]
    c_t = (c3[2] - c3[0]) / dt
    cz = grid.d_z(c)
    cx, cy = grid.d_x(c, cz), grid.d_y(c, cz)
    A = c_t + u * cx + v * cy + w * cz
    L_h = grid.d_x(cx) + grid.d_y(cy)
    L_z = grid.d_z(cz)
    K, ny, nx = c.shape
    kk, jj, ii = np.meshgrid(np.arange(K), np.arange(ny), np.arange(nx), indexing="ij")
    sj, si = les.source_cells()[src]
    near_src = (np.abs(jj - sj) <= 3) & (np.abs(ii - si) <= 3) & (kk <= 3)
    interior = (kk >= 1) & (kk <= K - 3) & (jj >= 2) & (jj <= ny - 3) & (ii >= 2) & (ii <= nx - 3)
    plume = c > 1e-3 * c.max()
    mask = interior & plume & ~near_src
    return dict(A=A, L_h=L_h, L_z=L_z, c=c, c_t=c_t, adv=A - c_t, mask=mask)


def fit_K(terms_list, n_max=400_000, seed=0):
    """NNLS fit of (K_h, K_z) over the masked cells of several term dicts; returns a summary."""
    rng = np.random.default_rng(seed)
    A, Lh, Lz, ct, adv = [], [], [], [], []
    for t in terms_list:
        m = t["mask"]
        for lst, key in ((A, "A"), (Lh, "L_h"), (Lz, "L_z"), (ct, "c_t"), (adv, "adv")):
            lst.append(t[key][m])
    A, Lh, Lz, ct, adv = (np.concatenate(a) for a in (A, Lh, Lz, ct, adv))
    if len(A) > n_max:
        sel = rng.choice(len(A), n_max, replace=False)
        A, Lh, Lz, ct, adv = A[sel], Lh[sel], Lz[sel], ct[sel], adv[sel]
    s = np.sqrt((A ** 2).mean())                              # scale for conditioning
    (Kh, Kz), _ = nnls(np.stack([Lh, Lz], 1) / s, A / s)
    r = A - Kh * Lh - Kz * Lz
    rms = lambda a: float(np.sqrt((a ** 2).mean()))  # noqa: E731
    return dict(K_h=float(Kh), K_z=float(Kz), n_cells=int(len(A)),
                rel_residual=rms(r) / rms(A), rel_residual_K0=1.0,
                rms_c_t=rms(ct), rms_adv=rms(adv), rms_A=rms(A),
                rms_Kh_Lh=rms(Kh * Lh), rms_Kz_Lz=rms(Kz * Lz), rms_residual=rms(r))


def residual_with_K(terms_list, K_h, K_z):
    """Relative residual |A - K_h L_h - K_z L_z| / |A| on the masked cells for given K."""
    num = den = 0.0
    for t in terms_list:
        m = t["mask"]
        r = t["A"][m] - K_h * t["L_h"][m] - K_z * t["L_z"][m]
        num += float((r ** 2).sum())
        den += float((t["A"][m] ** 2).sum())
    return float(np.sqrt(num / den))
