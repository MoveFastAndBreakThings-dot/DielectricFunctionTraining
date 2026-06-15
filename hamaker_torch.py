"""
Differentiable Hamaker pair constant for meta-training.

    A_12 = (3/2) k_B T [ (1/2) Δ_1(0)Δ_2(0) + Σ_{n≥1} Δ_1(n)Δ_2(n) ]
"""

import torch
import torch.nn as nn

K_B_J = 1.380649e-23
K_B_EV = 8.617333e-5
T_K = 300.0
ZJ_PER_J = 1e21


def linear_interp_eps(xi_grid, eps, xi_query):
    """
    Interpolate ε(iξ) onto Matsubara frequencies.

    xi_grid : (F,) ascending
    eps     : (B, F)
    xi_query: (M,)
    returns : (B, M)
    """
    xi_grid = xi_grid.to(device=eps.device, dtype=eps.dtype)
    xi_query = xi_query.to(device=eps.device, dtype=eps.dtype)
    m = xi_query.shape[0]
    out = torch.empty(eps.shape[0], m, device=eps.device, dtype=eps.dtype)

    left = xi_query < xi_grid[0]
    right = xi_query > xi_grid[-1]
    mid = ~(left | right)

    if left.any():
        out[:, left] = eps[:, :1].expand(-1, left.sum())
    if right.any():
        out[:, right] = 1.0
    if mid.any():
        q = xi_query[mid]
        idx = torch.searchsorted(xi_grid, q, right=True).clamp(1, len(xi_grid) - 1)
        x0 = xi_grid[idx - 1]
        x1 = xi_grid[idx]
        y0 = eps[:, idx - 1]
        y1 = eps[:, idx]
        w = (q - x0) / (x1 - x0 + 1e-12)
        out[:, mid] = y0 + w.unsqueeze(0) * (y1 - y0)

    return out


def hamaker_pair_j_from_eps(xi_grid, eps1, eps2, xi_matsubara, t_k=T_K, k_b_j=K_B_J):
    """
    eps1, eps2 : (B, F) on the KKR ξ grid (xx component)
    returns    : (B,) Hamaker constant in joules
    """
    d1 = linear_interp_eps(xi_grid, eps1, xi_matsubara)
    d2 = linear_interp_eps(xi_grid, eps2, xi_matsubara)
    delta1 = (d1 - 1.0) / (d1 + 1.0)
    delta2 = (d2 - 1.0) / (d2 + 1.0)
    pref = 1.5 * k_b_j * t_k
    a = 0.5 * pref * delta1[:, 0] * delta2[:, 0]
    if delta1.shape[1] > 1:
        a = a + pref * (delta1[:, 1:] * delta2[:, 1:]).sum(dim=1)
    return a


class HamakerPairLoss(nn.Module):
    """MAE or MSE on pairwise Hamaker constant (zJ)."""

    def __init__(
        self,
        xi_grid,
        n_matsubara=500,
        temperature_k=T_K,
        reduction="mae",
        use_log_a=False,
    ):
        super().__init__()
        self.register_buffer("xi_grid", torch.tensor(xi_grid, dtype=torch.float32))
        xi_step = 2.0 * torch.pi * K_B_EV * temperature_k
        xi_m = torch.arange(n_matsubara, dtype=torch.float32) * xi_step
        self.register_buffer("xi_matsubara", xi_m)
        self.temperature_k = float(temperature_k)
        self.reduction = reduction
        self.use_log_a = use_log_a

    def forward(self, eps1, eps2, a_true_zj):
        """
        eps1, eps2  : (B, F) linear ε(iξ)
        a_true_zj   : (B,) ground-truth pair Hamaker in zJ
        """
        a_pred_j = hamaker_pair_j_from_eps(
            self.xi_grid, eps1, eps2, self.xi_matsubara, t_k=self.temperature_k,
        )
        a_pred_zj = a_pred_j * ZJ_PER_J
        if self.use_log_a:
            pred = torch.log(a_pred_zj.clamp_min(1e-30))
            target = torch.log(a_true_zj.clamp_min(1e-30))
        else:
            pred = a_pred_zj
            target = a_true_zj
        if self.reduction == "mse":
            return torch.mean((pred - target) ** 2)
        return torch.mean(torch.abs(pred - target))
