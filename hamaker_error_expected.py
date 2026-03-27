"""
Expected error in Hamaker constant A from uncertainty in ε(iξ).
Uses S(d_n) = a + b * exp(-n/n_0) for uncertainty in d_n = ε(iξ_n), propagates to S_A.
Start with one material; multiple (a, b, n_0) tests.
"""

import os
import numpy as np
import torch

from hamaker_validation import (
    T_K,
    K_B_J,
    ZJ_PER_J,
    get_reference_structure,
    run_model_on_structure,
    get_matsubara_sequences,
)
from preprocess_dataset import load_atom_init, GaussianDistance
from model_loader import load_model


def s_d_n(n, a, b, n_0):
    """Uncertainty in d_n = ε(iξ_n): S(d_n) = a + b * exp(-n/n_0)."""
    return a + b * np.exp(-n / n_0)


def hamaker_error_expected(d_n, delta_n, a, b, n_0, t_k=T_K, n_max=None):
    """
    Propagate S(d_n) = a + b*exp(-n/n_0) to uncertainty S_A in Hamaker constant (J).
    ∂A/∂d_n: n=0 -> (3/2)k_B T * 2*Δ_0/(d_0+1)²; n≥1 -> (3/2)k_B T * 4*Δ_n/(d_n+1)².
    S_A² = Σ_n (∂A/∂d_n)² S(d_n)². Returns S_A in zJ.
    """
    if n_max is None:
        n_max = len(d_n)
    pref = 1.5 * K_B_J * t_k
    var_a = 0.0
    # n=0: weight 1/2 on Δ_0² -> ∂A/∂d_0 = pref * Δ_0 * 2/(d_0+1)²
    d0 = d_n[0]
    d0_1 = d0 + 1.0
    if d0_1 != 0:
        dA_dd0 = pref * 2.0 * delta_n[0] / (d0_1 ** 2)
        s0 = s_d_n(0, a, b, n_0)
        var_a += (dA_dd0 * s0) ** 2
    for n in range(1, n_max):
        dn = d_n[n]
        dn_1 = dn + 1.0
        if dn_1 != 0:
            dA_ddn = pref * 4.0 * delta_n[n] / (dn_1 ** 2)
            sn = s_d_n(n, a, b, n_0)
            var_a += (dA_ddn * sn) ** 2
    s_a_j = np.sqrt(var_a)
    return s_a_j * ZJ_PER_J


def run_one_material(name, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr):
    """Get eps_xi, xi and Matsubara sequences for one material."""
    structure = get_reference_structure(name)
    eps_xi, xi = run_model_on_structure(
        structure, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
    )
    d_n, delta_n = get_matsubara_sequences(xi, eps_xi)
    return d_n, delta_n


def main():
    model_path = "models/mace_kkr_v2.pt"
    if not os.path.exists(model_path):
        pts = [f for f in os.listdir("models")] if os.path.exists("models") else []
        pts = [f for f in pts if f.endswith(".pt")]
        model_path = os.path.join("models", pts[0]) if pts else None
    if not model_path or not os.path.exists(model_path):
        print("No .pt model found in models/")
        return

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    atom_init = load_atom_init("atom_init.json")
    radius = checkpoint.get("radius", 8.0)
    max_num_nbr = checkpoint.get("max_num_nbr", 12)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)

    # One material for now
    name = "Si"
    print(f"Material: {name}")
    d_n, delta_n = run_one_material(
        name, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
    )
    n_max = len(d_n)

    # Multiple (a, b, n_0) tests
    tests = [
        {"a": 0.01, "b": 0.05, "n_0": 100.0},
        {"a": 0.02, "b": 0.10, "n_0": 100.0},
        {"a": 0.01, "b": 0.05, "n_0": 500.0},
        {"a": 0.005, "b": 0.02, "n_0": 200.0},
        {"a": 0.05, "b": 0.0, "n_0": 1.0},
        {"a": 0.0, "b": 0.1, "n_0": 50.0},
    ]

    print(f"n_max = {n_max}. S(d_n) = a + b*exp(-n/n_0). S_A in zJ.")
    print("-" * 60)
    print(f"{'a':>8} {'b':>8} {'n_0':>8}  S_A (zJ)")
    print("-" * 60)
    for p in tests:
        s_a = hamaker_error_expected(d_n, delta_n, p["a"], p["b"], p["n_0"], n_max=n_max)
        print(f"{p['a']:>8.4f} {p['b']:>8.4f} {p['n_0']:>8.1f}  {s_a:>10.2f}")
    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
