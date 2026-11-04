"""Physics for the pit-2 methane PINN (PLAN.md §3; Jia 2025 eqs. 4.1, 4.5, 4.6).

Governing equation in physical coordinates (x, y, z ASL, t), solved for c [kg m-3]:

    dc/dt + u dc/dx + v dc/dy + w dc/dz - K_h (d2c/dx2 + d2c/dy2) - K_z d2c/dz2 = s(x, y, z; q)

with the 3D LES wind (u, v, w) and constant eddy diffusivities K_h, K_z per stability class
(trainable). The network sees height above ground, z_agl = z - h(x, y), where h is a C2 cubic
B-spline of the terrain. Derivatives are taken with autograd with respect to the physical
inputs, so the terrain chain-rule terms are exact.

Ground condition: zero normal diffusive flux, n . (K grad c) = 0 with n ~ (-h_x, -h_y, 1).
"""

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------------------------
# Terrain: uniform cubic B-spline with the grid values as control points (C2, slightly smoothing)
# ---------------------------------------------------------------------------------------------

def _bspline_weights(s):
    """Uniform cubic B-spline basis for local coordinate s in [0, 1): (..., 4)."""
    s2, s3 = s * s, s * s * s
    return torch.stack([(1 - s) ** 3, 3 * s3 - 6 * s2 + 4, -3 * s3 + 3 * s2 + 3 * s + 1, s3], dim=-1) / 6.0


class Terrain:
    """h(x, y) [m ASL] from a regular grid of terrain heights, differentiable to second order."""

    def __init__(self, x, y, topo, device="cpu"):
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.x0, self.y0 = float(x[0]), float(y[0])
        self.dx, self.dy = float(x[1] - x[0]), float(y[1] - y[0])
        self.nx, self.ny = len(x), len(y)
        # pad 2 cells by edge replication so every query inside the grid has 4x4 control points
        P = np.pad(np.asarray(topo, float), 2, mode="edge")
        self.width = P.shape[1]
        self.P = torch.as_tensor(P.ravel(), dtype=torch.float32, device=device)

    def to(self, device):
        self.P = self.P.to(device)
        return self

    def __call__(self, x, y):
        """h at points (x, y) (same shape). Clamped to the grid extent."""
        u = ((x - self.x0) / self.dx).clamp(0.0, self.nx - 1 - 1e-4)
        v = ((y - self.y0) / self.dy).clamp(0.0, self.ny - 1 - 1e-4)
        iu, iv = torch.floor(u.detach()), torch.floor(v.detach())
        wu, wv = _bspline_weights(u - iu), _bspline_weights(v - iv)          # (..., 4)
        # control points iu-1 .. iu+2 sit at padded indices iu+1 .. iu+4
        iu, iv = iu.long() + 1, iv.long() + 1
        out = torch.zeros_like(u)
        for a in range(4):
            for b in range(4):
                out = out + wv[..., a] * wu[..., b] * self.P[(iv + a) * self.width + iu + b]
        return out


# ---------------------------------------------------------------------------------------------
# LES wind on the terrain-following grid: z_agl = eta_k (H - h)
# ---------------------------------------------------------------------------------------------

def model_top_and_eta(z_agl, topo):
    """Fit z_agl[k] = eta_k (H - topo) per level; return (H, eta (K,))."""
    K = z_agl.shape[0]
    t = topo.ravel().astype(np.float64)
    Hs, etas = [], []
    for k in range(K):
        slope, icept = np.polyfit(t, z_agl[k].ravel().astype(np.float64), 1)
        etas.append(-slope)
        Hs.append(-icept / slope)
    H = float(np.median(Hs))
    eta = np.array([np.median(z_agl[k] / (H - topo)) for k in range(K)])
    return H, eta


class WindField:
    """Quadrilinear interpolation of (u, v, w) for several runs on one shared crop grid.

    uvw: (R, T, K, Ny, Nx, 3) array (any float dtype; kept on `device`), sampled at x, y (1D),
    model levels with coefficients eta (K,), and times (T,). Queries are not differentiated.
    """

    def __init__(self, uvw, x, y, eta, H, times, terrain: Terrain, device="cpu", dtype=torch.float16):
        self.uvw = torch.as_tensor(np.asarray(uvw), device=device).to(dtype)
        self.R, self.T, self.K, self.Ny, self.Nx, _ = self.uvw.shape
        self.x0, self.dx = float(x[0]), float(x[1] - x[0])
        self.y0, self.dy = float(y[0]), float(y[1] - y[0])
        self.t0, self.dt = float(times[0]), float(times[1] - times[0]) if len(times) > 1 else 1.0
        self.eta = torch.as_tensor(eta, dtype=torch.float32, device=device)
        self.H = float(H)
        self.terrain = terrain
        self.flat = self.uvw.reshape(-1, 3)

    @staticmethod
    def _split(f, n):
        f = f.clamp(0.0, n - 1 - 1e-4)
        i0 = torch.floor(f)
        return i0.long(), f - i0

    @torch.no_grad()
    def __call__(self, run, x, y, z, t):
        """(N, 3) wind [m s-1] at physical points; run is an (N,) long tensor of run indices."""
        h = self.terrain(x, y)
        eta = (z - h) / (self.H - h)
        kf = torch.searchsorted(self.eta, eta.contiguous()) - 1
        kf = kf.clamp(0, self.K - 2)
        e0, e1 = self.eta[kf], self.eta[kf + 1]
        wk = ((eta - e0) / (e1 - e0)).clamp(0.0, 1.0)
        i, wi = self._split((x - self.x0) / self.dx, self.Nx)
        j, wj = self._split((y - self.y0) / self.dy, self.Ny)
        if self.T > 1:
            n, wn = self._split((t - self.t0) / self.dt, self.T)
        else:
            n, wn = torch.zeros_like(i), torch.zeros_like(wi)
        out = torch.zeros(x.shape[0], 3, device=x.device)
        for dn, an in ((0, 1 - wn), (1, wn)):
            for dk, ak in ((0, 1 - wk), (1, wk)):
                for dj, aj in ((0, 1 - wj), (1, wj)):
                    for di, ai in ((0, 1 - wi), (1, wi)):
                        idx = ((((run * self.T + (n + dn).clamp(max=self.T - 1)) * self.K + kf + dk)
                                * self.Ny + (j + dj).clamp(max=self.Ny - 1)) * self.Nx
                               + (i + di).clamp(max=self.Nx - 1))
                        out += (an * ak * aj * ai).unsqueeze(-1) * self.flat[idx].float()
        return out


# ---------------------------------------------------------------------------------------------
# Source term (Jia eq. 4.5, anisotropic, with a ground image so each source integrates to q)
# ---------------------------------------------------------------------------------------------

def source_term(x, y, z_agl, q_kgph, src_xy, sigma_h=10.0, sigma_z=5.0, z_src=4.3):
    """s [kg m-3 s-1] at N points for per-point rates q_kgph (N, S); src_xy (S, 2) tensor."""
    dx = x.unsqueeze(-1) - src_xy[:, 0]
    dy = y.unsqueeze(-1) - src_xy[:, 1]
    gh = torch.exp(-(dx * dx + dy * dy) / (2 * sigma_h ** 2)) / (2 * math.pi * sigma_h ** 2)
    zz = z_agl.unsqueeze(-1)
    gz = (torch.exp(-(zz - z_src) ** 2 / (2 * sigma_z ** 2)) + torch.exp(-(zz + z_src) ** 2 / (2 * sigma_z ** 2))) \
        / (math.sqrt(2 * math.pi) * sigma_z)
    return ((q_kgph / 3600.0) * gh * gz).sum(-1)


# ---------------------------------------------------------------------------------------------
# Residuals (autograd w.r.t. the physical coordinates)
# ---------------------------------------------------------------------------------------------

def _grad(out, inp):
    return torch.autograd.grad(out, inp, grad_outputs=torch.ones_like(out), create_graph=True)[0]


def pde_residual(c_fn, X, wind, K_h, K_z, source):
    """Residual of the advection-diffusion equation at points X = (N, 4) [x, y, z, t] (physical).

    c_fn(X) -> (N,) concentration [kg m-3]; wind (N, 3); K_h, K_z (N,) or scalars;
    source (N,) [kg m-3 s-1]. Returns (residual (N,), c (N,)).
    """
    X = X.requires_grad_(True) if not X.requires_grad else X
    c = c_fn(X)
    g = _grad(c, X)                                   # (N, 4): c_x, c_y, c_z, c_t
    c_x, c_y, c_z, c_t = g.unbind(-1)
    c_xx = _grad(c_x, X)[:, 0]
    c_yy = _grad(c_y, X)[:, 1]
    c_zz = _grad(c_z, X)[:, 2]
    adv = wind[:, 0] * c_x + wind[:, 1] * c_y + wind[:, 2] * c_z
    r = c_t + adv - K_h * (c_xx + c_yy) - K_z * c_zz - source
    return r, c


def ground_flux_residual(c_fn, X, terrain: Terrain, K_h, K_z):
    """n . (K grad c) at ground points X (z = h(x, y)), with n ~ (-h_x, -h_y, 1) (unnormalized)."""
    X = X.requires_grad_(True) if not X.requires_grad else X
    c = c_fn(X)
    g = _grad(c, X)
    xy = X[:, :2].detach().clone().requires_grad_(True)
    h = terrain(xy[:, 0], xy[:, 1])
    hx, hy = torch.autograd.grad(h, xy, torch.ones_like(h))[0].unbind(-1)
    return -K_h * (hx * g[:, 0] + hy * g[:, 1]) + K_z * g[:, 2]
