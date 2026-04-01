"""
Correlated uncertainty in Hamaker constant A for the a=0 power-law dielectric error model:

    σ(ξ; b, c, α) = b / (ξ + c)^α   (ξ in eV; same as fit_error_models.model_power_law_a0)

a=0 so σ→0 as ξ→∞, matching the physical requirement that ε→1 at high frequency.

Linearizing ε_n → ε_n^nom + σ(ξ_n; θ) gives

    ∂A/∂θ_j = Σ_n (∂A/∂ε_n) (∂σ/∂θ_j),

with ∂A/∂ε_n = 6 k_B T w_n Δ_n / (ε_n + 1)^2, w_0 = 1/2, w_{n≥1} = 1,
Δ_n = (ε_n - 1)/(ε_n + 1). Then Var(A) ≈ g^T Σ_θ g.

α is treated as fixed (e.g. mean α from fits); θ = (b, c).
"""

import json
import os

import numpy as np

from lmdb_dataset import LMDBCrystalDataset
from hamaker_validation import (
    K_B_J,
    N_MATSUBARA,
    T_K,
    XI_STEP_EV,
    ZJ_PER_J,
    get_matsubara_sequences,
)


def _matsubara_weights(n_max):
    w = np.ones(n_max)
    w[0] = 0.5
    return w


def _dA_depsn(eps_n, delta_n, w_n, t_k=T_K, k_b=K_B_J):
    den = eps_n + 1.0
    if abs(den) < 1e-15:
        return 0.0
    return 6.0 * k_b * t_k * w_n * delta_n / (den ** 2)


def _dsigma_power_law_a0_dbc(xi_ev, b, c, alpha):
    """∂σ/∂b, ∂σ/∂c for σ = b/(ξ+c)^α  (a=0)."""
    xi_c = xi_ev + c
    if xi_c <= 0:
        raise ValueError("ξ + c must be positive for the power-law model.")
    pa = xi_c ** alpha
    d_db = 1.0 / pa
    d_dc = -alpha * b / (xi_c ** (alpha + 1.0))
    return d_db, d_dc


def grad_A_wrt_bc_power_law_a0(d_n, delta_n, b, c, alpha, xi_step_ev=XI_STEP_EV, n_max=None, t_k=T_K):
    """g = (g_b, g_c) with ξ_n = n * xi_step_ev."""
    if n_max is None:
        n_max = len(d_n)
    w = _matsubara_weights(n_max)
    g_b = g_c = 0.0
    for n in range(n_max):
        xi_n = n * xi_step_ev
        deps = _dA_depsn(d_n[n], delta_n[n], w[n], t_k=t_k)
        d_db, d_dc = _dsigma_power_law_a0_dbc(xi_n, b, c, alpha)
        g_b += deps * d_db
        g_c += deps * d_dc
    return np.array([g_b, g_c], dtype=np.float64)


def var_A_from_grad(g, theta_cov):
    return float(np.asarray(g) @ np.asarray(theta_cov, dtype=np.float64) @ np.asarray(g))


def hamaker_sigma_A_power_law_a0_correlated_zj(
    xi_ev,
    eps_xi,
    b,
    c,
    alpha,
    theta_cov,
    n_max=None,
    xi_step_ev=XI_STEP_EV,
    t_k=T_K,
):
    """
    sqrt(Var(A)) in zJ for a=0 power-law σ = b/(ξ+c)^α; α fixed, Σ_θ is 2×2 for (b, c).

    xi_ev, eps_xi : nominal predicted ε(iξ) on the KKR grid (interpolated to Matsubara).
    b, c         : evaluate g at these (e.g. test-split means from compare_power_law_a0.json).
    theta_cov    : 2×2 covariance of (b, c); diagonal from split stds is typical.
    """
    if n_max is None:
        n_max = N_MATSUBARA
    d_n, delta_n = get_matsubara_sequences(
        xi_ev, eps_xi, t_k=t_k, xi_step_ev=xi_step_ev, n_max=n_max
    )
    g = grad_A_wrt_bc_power_law_a0(
        d_n, delta_n, float(b), float(c), float(alpha), xi_step_ev=xi_step_ev, n_max=n_max, t_k=t_k
    )
    var_a = var_A_from_grad(g, theta_cov)
    return np.sqrt(max(var_a, 0.0)) * ZJ_PER_J, g, var_a


def load_theta_mean_cov_bc_from_compare_power_law_a0(compare_path, split="test"):
    """Diagonal Σ from error_model_fits/compare_power_law_a0.json for (b, c)."""
    with open(compare_path) as f:
        data = json.load(f)
    means = []
    var = []
    for name in ("b", "c"):
        entry = data[name][split]
        if entry is None:
            raise ValueError(f"Missing {split} for {name}")
        means.append(entry["mean"])
        var.append(entry["std"] ** 2)
    return np.array(means, dtype=np.float64), np.diag(var)


def load_alpha_mean_from_compare_power_law_a0(compare_path, split="test"):
    with open(compare_path) as f:
        data = json.load(f)
    entry = data["alpha"][split]
    if entry is None:
        raise ValueError(f"Missing {split} for alpha")
    return float(entry["mean"])


if __name__ == "__main__":
    compare_path = os.path.join("error_model_fits", "compare_power_law_a0.json")
    lmdb_path = "processed_data/crystal_graphs.lmdb"
    if not os.path.exists(compare_path):
        print(f"Run fit_error_models.py first; need {compare_path}")
    elif not os.path.exists(lmdb_path):
        print(f"Need {lmdb_path}")
    else:
        theta_mean, theta_cov = load_theta_mean_cov_bc_from_compare_power_law_a0(compare_path, split="test")
        alpha = load_alpha_mean_from_compare_power_law_a0(compare_path, split="test")

        ds = LMDBCrystalDataset(lmdb_path, use_kkr=True)
        xi = np.asarray(ds.kkr_xi_array, dtype=np.float64)
        _, tgt, _ = ds[0]
        eps_nom = np.asarray(tgt[0], dtype=np.float64)
        ds.close()

        n_prop = min(500, N_MATSUBARA)
        s_zj, g, va = hamaker_sigma_A_power_law_a0_correlated_zj(
            xi, eps_nom, theta_mean[0], theta_mean[1], alpha, theta_cov, n_max=n_prop
        )
        print(f"Demo: first crystal, n_max={n_prop}, alpha={alpha:.4f}")
        print(f"  sigma_A (correlated, a=0 power-law θ) = {s_zj:.4f} zJ, Var(A)={va:.6e} J^2, g=(g_b, g_c)={g}")
