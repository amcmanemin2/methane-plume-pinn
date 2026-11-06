"""Checks for physics.py and model.py.   python3 -m unittest discover -s tests -v"""

import math
import unittest

import numpy as np
import torch

from methane_pinn.model import PlumeModel
from methane_pinn.physics import (Terrain, WindField, ground_flux_residual, model_top_and_eta,
                                  pde_residual, source_term)

torch.set_default_dtype(torch.float32)


def puff(X, u, Kh, Kz, M=1.0, t0=-50.0):
    """Exact solution of c_t + u c_x = Kh (c_xx + c_yy) + Kz c_zz (instantaneous release at t0, origin)."""
    x, y, z, t = X.unbind(-1)
    tau = t - t0
    return M / ((4 * math.pi * tau) ** 1.5 * Kh * math.sqrt(Kz)) * torch.exp(
        -((x - u[0] * tau) ** 2 + (y - u[1] * tau) ** 2) / (4 * Kh * tau) - (z - u[2] * tau) ** 2 / (4 * Kz * tau))


class TestTerrain(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(10.0, 410.0, 20.0)
        self.y = np.arange(30.0, 330.0, 20.0)
        X, Y = np.meshgrid(self.x, self.y)
        self.rng = np.random.default_rng(0)
        self.rough = 50 * np.sin(X / 60.0) * np.cos(Y / 45.0) + 5 * self.rng.standard_normal(X.shape)
        self.plane = 0.3 * X - 0.1 * Y + 100.0

    def test_reproduces_planes_in_interior(self):
        t = Terrain(self.x, self.y, self.plane)
        xq = torch.tensor(self.rng.uniform(60, 340, 200), dtype=torch.float32)
        yq = torch.tensor(self.rng.uniform(80, 260, 200), dtype=torch.float32)
        np.testing.assert_allclose(t(xq, yq).numpy(), (0.3 * xq - 0.1 * yq + 100).numpy(), atol=1e-3)

    def test_autograd_matches_finite_differences(self):
        t = Terrain(self.x, self.y, self.rough)
        xq = torch.tensor(self.rng.uniform(60, 340, 50), dtype=torch.float64).float().requires_grad_(True)
        yq = torch.tensor(self.rng.uniform(80, 260, 50)).float()
        (hx,) = torch.autograd.grad(t(xq, yq).sum(), xq)
        d = 0.05
        fd = (t(xq.detach() + d, yq) - t(xq.detach() - d, yq)) / (2 * d)
        np.testing.assert_allclose(hx.numpy(), fd.numpy(), rtol=2e-2, atol=2e-3)

    def test_second_derivative_continuous_across_knots(self):
        t = Terrain(self.x, self.y, self.rough)

        def hxx(xv):
            xq = torch.tensor([xv], requires_grad=True)
            (g,) = torch.autograd.grad(t(xq, torch.tensor([150.0])).sum(), xq, create_graph=True)
            return torch.autograd.grad(g.sum(), xq)[0].item()
        knot = float(self.x[8])
        self.assertAlmostEqual(hxx(knot - 1e-3), hxx(knot + 1e-3), delta=1e-3)


class TestOperators(unittest.TestCase):
    def test_puff_is_an_exact_solution(self):
        u, Kh, Kz = (3.0, -1.0, 0.2), 8.0, 2.0
        g = torch.Generator().manual_seed(0)
        X = torch.stack([torch.rand(500, generator=g) * 200 - 50, torch.rand(500, generator=g) * 100 - 80,
                         torch.rand(500, generator=g) * 40 - 20, torch.rand(500, generator=g) * 60], -1).double()
        wind = torch.tensor(u, dtype=torch.float64).expand(500, 3)
        r, c = pde_residual(lambda Xi: puff(Xi, u, Kh, Kz), X.clone().requires_grad_(True),
                            wind, Kh, Kz, torch.zeros(500, dtype=torch.float64))
        X2 = X.clone().requires_grad_(True)
        c_t = torch.autograd.grad(puff(X2, u, Kh, Kz).sum(), X2)[0][:, 3]
        self.assertLess(r.abs().max().item(), 1e-9 * max(c_t.abs().max().item(), 1e-30) + 1e-14)
        self.assertGreater(c_t.abs().max().item(), 0)

    def test_ground_flux_on_flat_terrain_is_kz_dcdz(self):
        t = Terrain(np.arange(0, 200, 20.0), np.arange(0, 200, 20.0), np.zeros((10, 10)))
        X = torch.tensor([[50.0, 60.0, 0.0, 10.0], [120.0, 30.0, 0.0, 20.0]])
        f = ground_flux_residual(lambda Xi: (Xi[:, 0] * 0.01 + 2 * Xi[:, 2]) ** 2, X.clone(), t, 3.0, 0.5)
        c_z = 2 * 2 * (X[:, 0] * 0.01 + 2 * X[:, 2])
        np.testing.assert_allclose(f.detach().numpy(), (0.5 * c_z).numpy(), rtol=1e-6)

    def test_source_integrates_to_rate(self):
        src = torch.tensor([[0.0, 0.0], [300.0, 0.0]])
        xs = torch.arange(-60.0, 61.0, 1.0)
        zs = torch.arange(0.25, 40.0, 0.5)
        Xg, Yg, Zg = torch.meshgrid(xs, xs, zs, indexing="ij")
        q = torch.tensor([[3600.0, 7200.0]]).expand(Xg.numel(), 2)
        s = source_term(Xg.ravel(), Yg.ravel(), Zg.ravel(), q, src)
        total = s.sum().item() * 1.0 * 1.0 * 0.5
        self.assertAlmostEqual(total, 1.0, delta=2e-3)        # only the 1 kg/s source is inside the box


class TestConstantField(unittest.TestCase):
    """A constant field has no derivatives: residual = -s exactly, ground flux = 0."""

    def test_residual_is_minus_source(self):
        g = torch.Generator().manual_seed(3)
        n = 400
        X = torch.stack([torch.rand(n, generator=g) * 100, torch.rand(n, generator=g) * 100,
                         torch.rand(n, generator=g) * 30, torch.rand(n, generator=g) * 60], -1).double()
        c0 = 2.5e-6
        src = torch.tensor([[50.0, 50.0]], dtype=torch.float64)
        q = torch.full((n, 1), 1000.0, dtype=torch.float64)
        s = source_term(X[:, 0], X[:, 1], X[:, 2], q, src)
        wind = torch.tensor([3.0, -1.0, 0.2], dtype=torch.float64).expand(n, 3)
        const = lambda Xi: c0 + 0.0 * (Xi ** 3).sum(-1)  # noqa: E731
        r, c = pde_residual(const, X.clone().requires_grad_(True), wind, 10.0, 1.0, s)
        np.testing.assert_allclose(r.detach().numpy(), -s.numpy(), rtol=0, atol=1e-20)
        self.assertTrue(torch.allclose(c, torch.full_like(c, c0)))
        terrain = Terrain(np.arange(0, 200, 20.0), np.arange(0, 200, 20.0),
                          np.random.default_rng(0).uniform(0, 50, (10, 10)))
        f = ground_flux_residual(const, X[:, [0, 1, 2, 3]].float().clone(), terrain, 10.0, 1.0)
        self.assertEqual(float(f.abs().max()), 0.0)

    def test_physics_loss_scaling(self):
        """physics_loss = mean((tau/c_ref * residual)^2); with a constant field this is (tau/c_ref * s)^2."""
        from types import SimpleNamespace

        from methane_pinn.train import physics_loss

        class Const(torch.nn.Module):
            c_ref = 1e-6

            def K(self, stab):
                return torch.full(stab.shape, 10.0), torch.full(stab.shape, 1.0)

            def forward(self, X, q, met):
                return 3e-6 + 0.0 * (X ** 3).sum(-1)

        flat = Terrain(np.arange(0, 200, 20.0), np.arange(0, 200, 20.0), np.zeros((10, 10)))
        ts = SimpleNamespace(wind=lambda run, x, y, z, t: torch.ones(len(x), 3), terrain=flat,
                             src_t=torch.tensor([[100.0, 100.0]]), z_src=4.0)
        n = 300
        g = torch.Generator().manual_seed(4)
        X = torch.stack([95 + 10 * torch.rand(n, generator=g), 95 + 10 * torch.rand(n, generator=g),
                         10 * torch.rand(n, generator=g), 60 * torch.rand(n, generator=g)], -1)
        pts = dict(X=X, q=torch.full((n, 1), 500.0), met=torch.zeros(n, 1), run=torch.zeros(n, dtype=torch.long),
                   stab=torch.zeros(n, dtype=torch.long))
        phys = dict(tau=2.0, sigma_h=10.0, sigma_z=5.0)
        L, r_abs = physics_loss(Const(), ts, pts, phys)
        s = source_term(X[:, 0], X[:, 1], X[:, 2], pts["q"], ts.src_t, 10.0, 5.0, 4.0)
        expect = ((2.0 / 1e-6 * s) ** 2).mean()
        self.assertAlmostEqual(float(L) / float(expect), 1.0, places=5)
        self.assertGreater(float(expect), 0.0)


class TestFullModelResidual(unittest.TestCase):
    """Autograd PDE residual through PlumeModel (terrain chain rule, scaling, exp output) vs central
    finite differences in physical (x [m], y [m], z [m ASL], t [s]), in float64."""

    def test_autograd_matches_fd(self):
        x, y = np.arange(0, 420, 20.0), np.arange(0, 420, 20.0)
        Xg, Yg = np.meshgrid(x, y)
        topo = 100 + 0.3 * Xg + 40 * np.sin(Xg / 70.0) * np.cos(Yg / 90.0)     # sloped and curved
        terrain = Terrain(x, y, topo)
        dom = dict(x0=0.0, y0=0.0, t0=3600.0, Lx=400.0, Ly=400.0, Lz=200.0, T=3600.0)
        torch.manual_seed(0)
        m = PlumeModel(dom, terrain, n_src=2, n_met=3, met_mean=np.zeros(3), met_std=np.ones(3),
                       q_max=100.0, c_ref=1e-6, n_classes=1, hidden_size=32, num_layers=3, out_bias=0.0).double()
        terrain.P = terrain.P.double()
        n = 64
        rng = np.random.default_rng(5)
        xq, yq = rng.uniform(60, 340, n), rng.uniform(60, 340, n)
        zq = terrain(torch.tensor(xq), torch.tensor(yq)).numpy() + rng.uniform(5, 150, n)
        tq = rng.uniform(4000, 7000, n)
        X = torch.tensor(np.stack([xq, yq, zq, tq], 1))
        q = torch.tensor(rng.uniform(0, 50, (n, 2)))
        met = torch.tensor(rng.normal(size=(n, 3)))
        wind = torch.tensor(rng.normal(0, 3, (n, 3)))
        Kh, Kz, s = 7.0, 0.8, torch.zeros(n, dtype=torch.float64)
        f = lambda Xi: m(Xi, q, met)  # noqa: E731
        r, _ = pde_residual(f, X.clone().requires_grad_(True), wind, Kh, Kz, s)

        h = np.array([0.5, 0.5, 0.5, 0.5])                                    # m, m, m, s
        c0 = f(X)
        d1, d2 = [], []
        for a in range(4):
            e = torch.zeros(4, dtype=torch.float64)
            e[a] = h[a]
            cp, cm = f(X + e), f(X - e)
            d1.append((cp - cm) / (2 * h[a]))
            d2.append((cp - 2 * c0 + cm) / h[a] ** 2)
        r_fd = d1[3] + wind[:, 0] * d1[0] + wind[:, 1] * d1[1] + wind[:, 2] * d1[2] \
            - Kh * (d2[0] + d2[1]) - Kz * d2[2]
        scale = max(float(d1[3].abs().max()), float((wind[:, 0] * d1[0]).abs().max()), float((Kz * d2[2]).abs().max()))
        err = float((r.detach() - r_fd).abs().max())
        self.assertLess(err, 1e-4 * scale, f"max |autograd - FD| = {err:.3e}, term scale {scale:.3e}")
        self.assertGreater(scale, 0.0)


class TestScenarioModel(unittest.TestCase):
    """Stage 1 (Jia per scenario): coords-only inputs and fixed K."""

    def _model(self, **kw):
        x = np.arange(0, 400, 20.0)
        terrain = Terrain(x, x, np.zeros((20, 20)))
        dom = dict(x0=0.0, y0=0.0, t0=0.0, Lx=380.0, Ly=380.0, Lz=100.0, T=100.0)
        return PlumeModel(dom, terrain, n_src=2, n_met=3, met_mean=np.zeros(3), met_std=np.ones(3), q_max=10.0,
                          c_ref=1.0, n_classes=3, hidden_size=16, num_layers=2, **kw)

    def test_coords_only_ignores_rates_and_met(self):
        m = self._model(inputs="coords")
        self.assertEqual(m.net.network[0].in_features, 4)
        X = torch.tensor([[100.0, 50.0, 20.0, 30.0]])
        a = m(X, torch.zeros(1, 2), torch.zeros(1, 3))
        b = m(X, torch.full((1, 2), 7.0), torch.randn(1, 3))
        self.assertEqual(float(a), float(b))
        self.assertEqual(self._model().net.network[0].in_features, 4 + 2 + 3)

    def test_fixed_K_per_class(self):
        K = [[19.0, 6.6], [1.1, 1.8], [2.4, 0.25]]
        m = self._model(K_init=K, K_trainable=False)
        self.assertFalse(m.log_K.requires_grad)
        Kh, Kz = m.K(torch.tensor([0, 1, 2]))
        np.testing.assert_allclose(Kh.detach().numpy(), [19.0, 1.1, 2.4], rtol=1e-6)
        np.testing.assert_allclose(Kz.detach().numpy(), [6.6, 1.8, 0.25], rtol=1e-6)
        self.assertTrue(self._model(K_init=(10.0, 1.0)).log_K.requires_grad)


class TestWind(unittest.TestCase):
    def test_exact_for_trilinear_field(self):
        x, y = np.arange(0.0, 200.0, 20.0), np.arange(0.0, 160.0, 20.0)
        topo = np.full((8, 10), 50.0)
        H = 1000.0
        eta = np.linspace(0.005, 0.5, 6)
        times = np.array([0.0, 120.0, 240.0])
        terrain = Terrain(x, y, topo)
        T, K = len(times), len(eta)
        tt, kk, jj, ii = np.meshgrid(times, eta, y, x, indexing="ij")
        f = 0.01 * ii - 0.02 * jj + 3.0 * kk + 0.001 * tt
        uvw = np.stack([f, 2 * f, -f], axis=-1)[None]
        w = WindField(uvw, x, y, eta, H, times, terrain, dtype=torch.float32)
        rng = np.random.default_rng(1)
        n = 200
        xq, yq = rng.uniform(0, 180, n), rng.uniform(0, 140, n)
        eq, tq = rng.uniform(eta[0], eta[-1], n), rng.uniform(0, 240, n)
        zq = 50.0 + eq * (H - 50.0)
        out = w(torch.zeros(n, dtype=torch.long), *[torch.tensor(a, dtype=torch.float32) for a in (xq, yq, zq, tq)])
        expect = 0.01 * xq - 0.02 * yq + 3.0 * eq + 0.001 * tq
        np.testing.assert_allclose(out[:, 0].numpy(), expect, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out[:, 2].numpy(), -expect, rtol=1e-4, atol=1e-4)

    def test_model_top_fit(self):
        topo = np.random.default_rng(0).uniform(100, 600, (20, 20))
        eta = np.linspace(0.001, 0.4, 10)
        z_agl = eta[:, None, None] * (2782.0 - topo[None])
        H, e = model_top_and_eta(z_agl, topo)
        self.assertAlmostEqual(H, 2782.0, delta=1e-6 * 2782)
        np.testing.assert_allclose(e, eta, rtol=1e-6)


class TestModel(unittest.TestCase):
    def test_terrain_enters_derivatives(self):
        """d c / d x at fixed z must include -h_x * d c / d z_agl."""
        x, y = np.arange(0, 400, 20.0), np.arange(0, 400, 20.0)
        X_, _ = np.meshgrid(x, y)
        terrain = Terrain(x, y, 0.5 * X_)                          # 0.5 slope in x
        dom = dict(x0=0.0, y0=0.0, t0=0.0, Lx=380.0, Ly=380.0, Lz=100.0, T=100.0)
        m = PlumeModel(dom, terrain, n_src=2, n_met=3, met_mean=np.zeros(3), met_std=np.ones(3),
                       q_max=10.0, c_ref=1.0, n_classes=1, hidden_size=16, num_layers=2)
        X = torch.tensor([[150.0, 200.0, 120.0, 50.0]], requires_grad=True)
        q, met = torch.zeros(1, 2), torch.zeros(1, 3)
        g = torch.autograd.grad(m(X, q, met).sum(), X)[0][0]
        d = 1e-2
        e = torch.tensor([d, 0.0, 0.0, 0.0])
        fd = (m(X.detach() + e, q, met) - m(X.detach() - e, q, met)) / (2 * d)
        self.assertAlmostEqual(g[0].item(), fd.item(), delta=1e-3 * abs(fd.item()) + 1e-6)
        # the same model on flat terrain gives a different x-derivative
        m.terrain = Terrain(x, y, np.zeros_like(X_))
        g_flat = torch.autograd.grad(m(X, q, met).sum(), X)[0][0]
        self.assertNotAlmostEqual(g[0].item(), g_flat[0].item(), places=6)


if __name__ == "__main__":
    unittest.main()
