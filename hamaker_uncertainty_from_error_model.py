"""
Use the simple-inverse dielectric error model to get the distribution of
expected error (uncertainty) in the Hamaker constant A, in zJ.

Error in ε(iξ):  E(ξ) = b/(ξ + ξ₀)   (ξ in eV), with a = 0.
At high ξ both predicted and actual ε → 1, so error must → 0; a nonzero constant
would make the propagated S_A sum diverge. Upper edge uses mean + k*std for (b, ξ₀) only.
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model
from hamaker_validation import (
    T_K,
    K_B_J,
    XI_STEP_EV,
    ZJ_PER_J,
    N_MATSUBARA,
    get_matsubara_sequences,
)

OUT_DIR = "hamaker_uncertainty"
ERROR_FIT_DIR = "error_model_fits"
# Use a=0 fit (error→0 as ξ→∞); file has b, xi0 only.
COMPARE_JSON = "compare_simple_inverse_a0.json"
os.makedirs(OUT_DIR, exist_ok=True)

# Number of standard deviations above mean for "upper edge" (conservative)
K_STD = 1.0

# Number of terms in the error-propagation sum. None = use N_MATSUBARA (5000).
# 500 matches typical ε(iξ) grid size; reduces S_A to more plausible range.
N_ERR_PROP = 500


def simple_inverse_error(xi_ev, b, xi0, a=0.0):
    """Error in ε at frequency ξ (eV): E(ξ) = a + b/(ξ + ξ0). Use a=0 so E→0 as ξ→∞."""
    return a + b / (xi_ev + xi0)


def hamaker_uncertainty_simple_inverse(d_n, delta_n, b, xi0, a=0.0, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=None):
    """
    Propagate S(d_n) = b/(ξ_n + ξ0) [a=0 so error→0 at high ξ] to S_A in zJ.
    ∂A/∂d_0 = (3/2)k_B T · 2Δ₀/(d₀+1)²,  ∂A/∂d_n = (3/2)k_B T · 4Δ_n/(d_n+1)² (n≥1).
    S_A² = Σ_n (∂A/∂d_n)² S(d_n)².  See docs_hamaker_error_propagation.md.
    """
    if n_max is None:
        n_max = len(d_n)
    pref = 1.5 * K_B_J * t_k
    var_a = 0.0
    for n in range(n_max):
        xi_n = n * xi_step_ev
        s_n = simple_inverse_error(xi_n, b, xi0, a=a)
        dn = d_n[n]
        dn_1 = dn + 1.0
        if dn_1 == 0:
            continue
        if n == 0:
            dA_ddn = pref * 2.0 * delta_n[0] / (dn_1 ** 2)
        else:
            dA_ddn = pref * 4.0 * delta_n[n] / (dn_1 ** 2)
        var_a += (dA_ddn * s_n) ** 2
    return np.sqrt(var_a) * ZJ_PER_J


def get_upper_edge_params(compare_path, split="test", k_std=K_STD):
    """Load simple-inverse comparison; return b, xi0 (mean + k_std*std). a=0 for propagation."""
    with open(compare_path) as f:
        data = json.load(f)
    b_m, b_s = data["b"][split]["mean"], data["b"][split]["std"]
    x_m, x_s = data["xi0"][split]["mean"], data["xi0"][split]["std"]
    b = max(0.0, b_m + k_std * b_s)
    xi0 = max(1e-6, x_m + k_std * x_s)
    return b, xi0


def main():
    model_path = "models/mace_kkr_v2.pt"
    if not os.path.exists(model_path) and os.path.exists("models"):
        pts = [f for f in os.listdir("models") if f.endswith(".pt")]
        model_path = os.path.join("models", pts[0]) if pts else None
    if not model_path or not os.path.exists(model_path):
        print("No .pt model found in models/")
        return

    compare_path = os.path.join(ERROR_FIT_DIR, COMPARE_JSON)
    if not os.path.exists(compare_path):
        print(f"Run fit_error_models.py first to create {compare_path}")
        return

    b_ue, xi0_ue = get_upper_edge_params(compare_path, split="test", k_std=K_STD)
    print("Error in ε(iξ):  E(ξ) = b/(ξ + ξ₀)   (a=0 so error→0 as ξ→∞)")
    print(f"Upper edge (test mean + {K_STD}*std): b={b_ue:.4f}, ξ₀={xi0_ue:.4f} eV")

    lmdb_path = "processed_data/crystal_graphs.lmdb"
    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    use_log = checkpoint.get("use_log", False)

    if "test_idx" not in checkpoint:
        np.random.seed(42)
        n = len(LMDBCrystalDataset(lmdb_path, use_kkr=True))
        idx = np.random.permutation(n)
        train_sz = int(n * 0.8)
        val_sz = int(n * 0.1)
        test_idx = idx[train_sz + val_sz :].tolist()
    else:
        test_idx = checkpoint["test_idx"]

    dataset = LMDBCrystalDataset(lmdb_path, use_kkr=True)
    xi_dataset = np.array(dataset.kkr_xi_array)
    n_freq = len(xi_dataset)
    n_sum = N_ERR_PROP if N_ERR_PROP is not None else N_MATSUBARA
    n_sum = min(n_sum, N_MATSUBARA)
    print(f"Matsubara sum length: N = {n_sum} (N_MATSUBARA={N_MATSUBARA}, data n_freq={n_freq})")

    results = []
    for idx in test_idx:
        idx = int(idx)
        info = dataset.get_sample_info(idx)
        inputs, tgt, _ = collate_pool([dataset[idx]])
        atom_fea_t = inputs[0].to(device)
        nbr_fea_t = inputs[1].to(device)
        nbr_fea_idx_t = inputs[2].to(device)
        crystal_atom_idx = [x.to(device) for x in inputs[3]]
        with torch.no_grad():
            pred = model(atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx)
        pred_real = torch.exp(pred) if use_log else pred
        predicted = pred_real[0, 0].cpu().numpy()

        d_n, delta_n = get_matsubara_sequences(xi_dataset, predicted, n_max=N_MATSUBARA)
        s_a = hamaker_uncertainty_simple_inverse(d_n, delta_n, b_ue, xi0_ue, a=0.0, n_max=n_sum)
        results.append({
            "formula": info["formula"],
            "mpid": info["mpid"],
            "S_A_zJ": float(s_a),
        })
    dataset.close()

    s_a_arr = np.array([r["S_A_zJ"] for r in results])

    dist = {
        "error_in_eps_formula": "E(ξ) = b/(ξ + ξ₀),  a=0 so E→0 as ξ→∞",
        "upper_edge_params": {"a": 0, "b": b_ue, "xi0_eV": xi0_ue, "k_std": K_STD},
        "N_error_propagation": n_sum,
        "N_MATSUBARA": N_MATSUBARA,
        "n_materials": len(results),
        "S_A_zJ_mean": float(np.mean(s_a_arr)),
        "S_A_zJ_std": float(np.std(s_a_arr)),
        "S_A_zJ_median": float(np.median(s_a_arr)),
        "S_A_zJ_percentiles": {p: float(np.percentile(s_a_arr, p)) for p in [5, 25, 50, 75, 95]},
    }
    with open(os.path.join(OUT_DIR, "hamaker_uncertainty_dist.json"), "w") as f:
        json.dump(dist, f, indent=2)

    print("\nDistribution of error in Hamaker constant A (zJ):")
    print(f"  mean   = {dist['S_A_zJ_mean']:.2f}")
    print(f"  std    = {dist['S_A_zJ_std']:.2f}")
    print(f"  median = {dist['S_A_zJ_median']:.2f}")
    print(f"  5–95%  = {dist['S_A_zJ_percentiles'][5]:.2f} – {dist['S_A_zJ_percentiles'][95]:.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(s_a_arr, bins=40, alpha=0.7, color="C0", edgecolor="k", label="S_A (zJ)")
    ax.axvline(np.median(s_a_arr), color="k", ls="--", lw=2, label="Median")
    ax.set_xlabel("Error in A (zJ)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Hamaker constant error (upper-edge simple inverse)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hamaker_uncertainty_dist.png"), dpi=150)
    plt.close()

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
