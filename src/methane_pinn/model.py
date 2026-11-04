"""PINN for pit-2 methane concentration (PLAN.md §3, §6; Jia 2025 §4.3.3).

    c(x, y, z, t; q, m) = c_ref * exp( NN( x', y', z_agl', t', q / Q_max, m_norm ) )

- x', y', z_agl', t' are scaled to [-0.5, 0.5] (Jia eq. 4.3); z_agl = z - h(x, y) is computed
  inside forward() from the differentiable terrain, so autograd derivatives w.r.t. the physical
  (x, y, z, t) include the terrain terms.
- q: the 15 pit-2 emission rates [kg/h]; m: the 113 met-sensor inputs (normalized).
- tanh hidden layers and an exponential output (c > 0), as in Jia.
"""

import torch
import torch.nn as nn

from methane_pinn.physics import Terrain


class PINN(nn.Module):
    """Fully connected network, same structure as the class example (hw01), with exp output."""

    def __init__(self, input_size, hidden_size=256, num_layers=6, out_bias=-5.0):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.network[-1].bias, out_bias)

    def forward(self, features):
        return self.network(features).squeeze(-1)


class PlumeModel(nn.Module):
    """Wraps the PINN with coordinate scaling, terrain, conditioning inputs and the trainable K."""

    def __init__(self, domain: dict, terrain: Terrain, n_src: int, n_met: int, met_mean, met_std,
                 q_max: float, c_ref: float, n_classes: int, K_init=(10.0, 1.0), **net_kw):
        super().__init__()
        self.terrain = terrain
        d = domain
        self.register_buffer("lo", torch.tensor([d["x0"], d["y0"], 0.0, d["t0"]], dtype=torch.float32))
        self.register_buffer("span", torch.tensor([d["Lx"], d["Ly"], d["Lz"], d["T"]], dtype=torch.float32))
        self.register_buffer("met_mean", torch.as_tensor(met_mean, dtype=torch.float32))
        self.register_buffer("met_std", torch.as_tensor(met_std, dtype=torch.float32))
        self.q_max, self.c_ref = float(q_max), float(c_ref)
        self.net = PINN(4 + n_src + n_met, **net_kw)
        # log K_h, log K_z per stability class [m2 s-1]
        self.log_K = nn.Parameter(torch.log(torch.tensor([K_init] * n_classes, dtype=torch.float32)))

    def K(self, stab_idx):
        """(K_h, K_z) per point for an (N,) tensor of stability-class indices."""
        k = torch.exp(self.log_K)[stab_idx]
        return k[:, 0], k[:, 1]

    def scaled_coords(self, X):
        """X (N, 4) physical [x, y, z, t] -> (N, 4) in [-0.5, 0.5] with z replaced by z_agl."""
        z_agl = X[:, 2] - self.terrain(X[:, 0], X[:, 1])
        Xa = torch.stack([X[:, 0], X[:, 1], z_agl, X[:, 3]], dim=-1)
        return (Xa - self.lo) / self.span - 0.5

    def log_c_star(self, X, q, met):
        feats = torch.cat([self.scaled_coords(X), q / self.q_max, (met - self.met_mean) / self.met_std], dim=-1)
        return self.net(feats)

    def forward(self, X, q, met):
        """Concentration [kg m-3] at physical points X (N, 4) for rates q (N, S) and met (N, M)."""
        return self.c_ref * torch.exp(self.log_c_star(X, q, met))
