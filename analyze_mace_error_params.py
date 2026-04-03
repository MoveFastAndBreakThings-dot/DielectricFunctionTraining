"""
Analyze a trained FrequencyMACEWithABC checkpoint.

Part (i): Loss breakdown + distribution of predicted (b, c) on the test set.
Part (ii): Per-crystal Hamaker uncertainty σ_A from the predicted (b, c),
           using uncorrelated propagation:
               σ_A² = Σ_n (∂A/∂ε_n)² · σ(ξ_n)²,   σ(ξ_n) = b̂/(ξ_n + ĉ)^α
Part (iii): "Oracle" (b*, c*) fit per crystal from actual residuals |ε_pred − ε_true|,
            compared against model-predicted (b̂, ĉ) to measure b and c prediction error.

Usage:
    python analyze_mace_error_params.py
    python analyze_mace_error_params.py --model_path models/mace_kkr_with_abc_v1.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Subset

from hamaker_validation import K_B_J, N_MATSUBARA, T_K, XI_STEP_EV, ZJ_PER_J, get_matsubara_sequences
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses_mace_error_params import SpectrumAndPowerLawABCLoss
from mace_model_error_params import FrequencyMACEWithABC

OUT_DIR = "analysis_error_params"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Part (ii) helper: uncorrelated σ_A from a single (b, c) point estimate
# ---------------------------------------------------------------------------

def sigma_A_from_bc_uncorrelated(xi_ev, eps_xi, b, c, alpha, n_max=500, xi_step_ev=XI_STEP_EV, t_k=T_K):
    """
    σ_A = sqrt( Σ_n (∂A/∂ε_n)² · σ(ξ_n)² )  where σ(ξ_n) = b/(ξ_n+c)^α.

    Uses the predicted ε(iξ) to evaluate ∂A/∂ε_n at each Matsubara point.
    Returns σ_A in zJ.
    """
    d_n, delta_n = get_matsubara_sequences(xi_ev, eps_xi, t_k=t_k, xi_step_ev=xi_step_ev, n_max=n_max)
    w = np.ones(n_max)
    w[0] = 0.5
    pref = 1.5 * K_B_J * t_k
    var_a = 0.0
    for n in range(n_max):
        xi_n = n * xi_step_ev
        dn = d_n[n]
        den = dn + 1.0
        if abs(den) < 1e-15:
            continue
        if n == 0:
            dA_ddn = pref * 2.0 * delta_n[0] / (den ** 2)
        else:
            dA_ddn = pref * 4.0 * delta_n[n] / (den ** 2)
        s_n = b / ((xi_n + c) ** alpha)
        var_a += (dA_ddn * s_n) ** 2
    return float(np.sqrt(var_a) * ZJ_PER_J)


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint, device):
    model = FrequencyMACEWithABC(
        orig_atom_fea_len=checkpoint["orig_atom_fea_len"],
        nbr_fea_len=checkpoint["nbr_fea_len"],
        atom_fea_len=checkpoint.get("atom_fea_len", 128),
        n_layers=checkpoint.get("n_layers", 4),
        h_fea_len=checkpoint.get("h_fea_len", 256),
        n_h=checkpoint.get("n_h", 1),
        n_frequencies=checkpoint["n_frequencies"],
        n_output_channels=checkpoint["n_output_channels"],
        dropout=checkpoint.get("dropout", 0.1),
        abc_hidden=checkpoint.get("abc_hidden", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Distribution stats + print helper
# ---------------------------------------------------------------------------

def dist_stats(arr, name):
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }
    print(f"\n  {name}:")
    print(f"    mean   = {stats['mean']:.4f}")
    print(f"    median = {stats['median']:.4f}")
    print(f"    std    = {stats['std']:.4f}")
    print(f"    min    = {stats['min']:.4f}   max = {stats['max']:.4f}")
    print(f"    5–25–75–95 pct: {stats['p5']:.4f} | {stats['p25']:.4f} | {stats['p75']:.4f} | {stats['p95']:.4f}")
    return stats


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_distribution(arr, name, xlabel, save_path, color="C0", log_x=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data = np.log10(arr + 1e-12) if log_x else arr
    xlabel_used = f"log₁₀({xlabel})" if log_x else xlabel

    axes[0].hist(data, bins=50, alpha=0.75, color=color, edgecolor="k")
    axes[0].axvline(np.mean(data), color="k", ls="--", lw=1.5, label="Mean")
    axes[0].axvline(np.median(data), color="r", ls=":", lw=1.5, label="Median")
    axes[0].set_xlabel(xlabel_used)
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{name} — histogram")
    axes[0].legend()

    axes[1].plot(np.sort(data), np.linspace(0, 1, len(data)), color=color, lw=1.5)
    axes[1].axvline(np.mean(data), color="k", ls="--", lw=1.5, label="Mean")
    axes[1].axvline(np.median(data), color="r", ls=":", lw=1.5, label="Median")
    axes[1].set_xlabel(xlabel_used)
    axes[1].set_ylabel("CDF")
    axes[1].set_title(f"{name} — CDF")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_bc_joint(b_arr, c_arr, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(b_arr, c_arr, s=8, alpha=0.4, color="C0")
    ax.set_xlabel("b")
    ax.set_ylabel("c  (eV)")
    ax.set_title("Joint distribution of predicted (b, c)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=f"models/mace_kkr_with_abc_v1.pt")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--n_matsubara", type=int, default=500,
                        help="Number of Matsubara terms for σ_A propagation")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint, device)

    use_log = checkpoint.get("use_log", False)
    alpha = checkpoint.get("alpha", 3.0)
    sigma_transform = checkpoint.get("sigma_transform", "none")
    target_components = checkpoint.get("target_components", ("xx",))

    print(f"Model: {checkpoint.get('model_name', args.model_path)}")
    print(f"  use_log={use_log}, alpha={alpha}, sigma_transform={sigma_transform}")

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True)
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    test_idx = checkpoint.get("test_idx")
    if test_idx is None:
        np.random.seed(42)
        n = len(dataset)
        idx = np.random.permutation(n)
        train_sz, val_sz = int(n * 0.8), int(n * 0.1)
        test_idx = idx[train_sz + val_sz:].tolist()

    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_pool)

    criterion = SpectrumAndPowerLawABCLoss(
        xi_array=dataset.frequency_array,
        alpha=alpha,
        lambda_eps=checkpoint.get("lambda_eps", 1.0),
        lambda_sigma=checkpoint.get("lambda_sigma", 1.0),
        eps_reduction="mae",
        sigma_reduction="mse",
        sigma_transform=sigma_transform,
    ).to(device)
    criterion.eps_pred_is_log = use_log

    # -----------------------------------------------------------------------
    # Part (i): collect losses, b, c
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Part (i): loss breakdown and (b, c) distribution on test set")
    print(f"{'='*60}")

    total_loss = total_loss_eps = total_loss_sigma = 0.0
    n_total = 0
    b_list, c_list = [], []
    records = []       # predicted ε in real space
    records_true = []  # actual ε in real space

    with torch.no_grad():
        for inputs, target, _ in test_loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [x.to(device) for x in crystal_atom_idx]
            target = target.to(device)

            eps_out, bc_out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss, parts = criterion(eps_out, bc_out, target)

            bs = target.shape[0]
            total_loss += loss.item() * bs
            total_loss_eps += parts["loss_eps"] * bs
            total_loss_sigma += parts["loss_sigma"] * bs
            n_total += bs

            bc_cpu = bc_out.cpu().numpy()
            b_list.append(bc_cpu[:, 0])
            c_list.append(bc_cpu[:, 1])

            if use_log:
                eps_real = torch.exp(eps_out).cpu().numpy()
            else:
                eps_real = eps_out.cpu().numpy()
            records.append(eps_real)
            records_true.append(target.cpu().numpy())

    mean_loss = total_loss / n_total
    mean_loss_eps = total_loss_eps / n_total
    mean_loss_sigma = total_loss_sigma / n_total

    print(f"\n  Test loss (total)      = {mean_loss:.6f}")
    print(f"  Test loss (ε part)     = {mean_loss_eps:.6f}")
    print(f"  Test loss (σ part)     = {mean_loss_sigma:.6f}")

    b_arr = np.concatenate(b_list)
    c_arr = np.concatenate(c_list)

    b_stats = dist_stats(b_arr, "b")
    c_stats = dist_stats(c_arr, "c  (eV)")

    plot_distribution(b_arr, "b", "b", os.path.join(OUT_DIR, "dist_b.png"), color="C0")
    plot_distribution(c_arr, "c", "c (eV)", os.path.join(OUT_DIR, "dist_c.png"), color="C1")
    plot_bc_joint(b_arr, c_arr, os.path.join(OUT_DIR, "joint_bc.png"))
    print(f"\n  Plots saved to {OUT_DIR}/")

    # -----------------------------------------------------------------------
    # Part (ii): per-crystal σ_A from predicted (b, c)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Part (ii): Hamaker uncertainty σ_A per crystal (test set)")
    print(f"  Using uncorrelated propagation with predicted (b̂, ĉ) per crystal")
    print(f"  n_matsubara = {args.n_matsubara}")
    print(f"{'='*60}")

    sigma_A_list = []
    eps_all = np.concatenate(records, axis=0)
    eps_true_all = np.concatenate(records_true, axis=0)

    for i, idx in enumerate(test_idx):
        eps_pred_i = eps_all[i, 0, :]
        b_i = float(b_arr[i])
        c_i = float(c_arr[i])
        s_a = sigma_A_from_bc_uncorrelated(xi, eps_pred_i, b_i, c_i, alpha, n_max=args.n_matsubara)
        sigma_A_list.append(s_a)

    sigma_A_arr = np.array(sigma_A_list)

    dist_stats(sigma_A_arr, "σ_A (zJ)")

    plot_distribution(sigma_A_arr, "σ_A", "σ_A (zJ)",
                      os.path.join(OUT_DIR, "dist_sigma_A.png"), color="C2", log_x=False)
    plot_distribution(sigma_A_arr, "σ_A (log scale)", "σ_A (zJ)",
                      os.path.join(OUT_DIR, "dist_sigma_A_log.png"), color="C2", log_x=True)

    # summary figure: b, c, σ_A side by side
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, arr, label, color in zip(
        axes,
        [b_arr, c_arr, sigma_A_arr],
        ["b", "c (eV)", "σ_A (zJ)"],
        ["C0", "C1", "C2"],
    ):
        ax.hist(arr, bins=50, alpha=0.75, color=color, edgecolor="k")
        ax.axvline(np.mean(arr), color="k", ls="--", lw=1.5, label=f"mean={np.mean(arr):.3g}")
        ax.axvline(np.median(arr), color="r", ls=":", lw=1.5, label=f"med={np.median(arr):.3g}")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.suptitle("Test-set distributions: predicted b, c and derived σ_A")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "summary.png"), dpi=150)
    plt.close()

    print(f"\n  All plots saved to {OUT_DIR}/")

    # -----------------------------------------------------------------------
    # Part (iii): oracle (b*, c*) per crystal — predicted vs fitted
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Part (iii): oracle (b*, c*) fit per crystal vs model (b̂, ĉ)")
    print(f"  Fits σ(ξ) = b/(ξ+c)^α to actual |ε_pred − ε_true| for each crystal")
    print(f"  α = {alpha:.3f}")
    print(f"{'='*60}")

    def _fit_bc(residuals, xi_ev, alpha_val):
        """Least-squares fit of b/(ξ+c)^α to residual curve. Returns (b*, c*) or None."""
        mask = residuals > 1e-9
        if mask.sum() < 5:
            return None
        xi_m, r_m = xi_ev[mask], residuals[mask]

        def loss_fn(params):
            log_b, log_c = params
            b_ = np.exp(log_b)
            c_ = np.exp(log_c)
            sigma_ = b_ / (xi_m + c_) ** alpha_val
            return np.mean((np.log1p(sigma_) - np.log1p(r_m)) ** 2)

        best_val, best_res = np.inf, None
        for log_b0, log_c0 in [(np.log(1.0), np.log(0.5)), (np.log(5.0), np.log(1.0)),
                                (np.log(20.0), np.log(2.0)), (np.log(50.0), np.log(0.1))]:
            res = minimize(loss_fn, [log_b0, log_c0], method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7})
            if res.fun < best_val:
                best_val, best_res = res.fun, res
        if best_res is None or not best_res.success and best_res.fun > 1.0:
            return None
        b_fit = float(np.exp(best_res.x[0]))
        c_fit = float(np.exp(best_res.x[1]))
        return b_fit, c_fit

    b_fit_list, c_fit_list, b_pred_matched, c_pred_matched = [], [], [], []
    n_failed = 0
    for i in range(len(test_idx)):
        eps_p = eps_all[i, 0, :]
        eps_t = eps_true_all[i, 0, :]
        residuals = np.abs(eps_p - eps_t)
        result = _fit_bc(residuals, xi, alpha)
        if result is None:
            n_failed += 1
            continue
        b_fit_list.append(result[0])
        c_fit_list.append(result[1])
        b_pred_matched.append(float(b_arr[i]))
        c_pred_matched.append(float(c_arr[i]))

    b_fit_arr = np.array(b_fit_list)
    c_fit_arr = np.array(c_fit_list)
    b_pred_matched = np.array(b_pred_matched)
    c_pred_matched = np.array(c_pred_matched)

    print(f"\n  Crystals fitted successfully: {len(b_fit_arr)} / {len(test_idx)}  ({n_failed} failed)")

    b_err = b_pred_matched - b_fit_arr
    c_err = c_pred_matched - c_fit_arr

    print(f"\n  b  prediction error  (b̂ − b*):")
    print(f"    MAE  = {np.mean(np.abs(b_err)):.4f}")
    print(f"    RMSE = {np.sqrt(np.mean(b_err**2)):.4f}")
    print(f"    bias = {np.mean(b_err):.4f}  (positive → model over-predicts b)")
    print(f"    std  = {np.std(b_err):.4f}")

    print(f"\n  c  prediction error  (ĉ − c*)  [eV]:")
    print(f"    MAE  = {np.mean(np.abs(c_err)):.4f}")
    print(f"    RMSE = {np.sqrt(np.mean(c_err**2)):.4f}")
    print(f"    bias = {np.mean(c_err):.4f}  (positive → model over-predicts c)")
    print(f"    std  = {np.std(c_err):.4f}")

    # --- scatter: predicted vs oracle ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred, fit, label, unit in zip(
        axes,
        [b_pred_matched, c_pred_matched],
        [b_fit_arr, c_fit_arr],
        ["b", "c"],
        ["", " (eV)"],
    ):
        ax.scatter(fit, pred, s=8, alpha=0.35, color="C0")
        lims = [min(fit.min(), pred.min()) * 0.9, max(fit.max(), pred.max()) * 1.1]
        ax.plot(lims, lims, "k--", lw=1.2, label="y = x  (perfect)")
        ax.set_xlabel(f"Oracle {label}*{unit}  (fit to residuals)")
        ax.set_ylabel(f"Model {label}̂{unit}  (predicted)")
        mae = np.mean(np.abs(pred - fit))
        ax.set_title(f"{label}:  MAE = {mae:.3f}{unit}")
        ax.legend(fontsize=8)
    fig.suptitle("Model-predicted (b̂, ĉ) vs oracle-fitted (b*, c*) per test crystal", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bc_predicted_vs_oracle.png"), dpi=150)
    plt.close()

    # --- error distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, err, label, unit in zip(axes, [b_err, c_err], ["b̂ − b*", "ĉ − c*"], ["", " (eV)"]):
        ax.hist(err, bins=50, alpha=0.75, color="C3", edgecolor="k")
        ax.axvline(0, color="k", ls="--", lw=1.2, label="zero error")
        ax.axvline(np.mean(err), color="r", ls=":", lw=1.5, label=f"mean={np.mean(err):.3f}")
        ax.set_xlabel(f"{label}{unit}")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}{unit}")
        ax.legend(fontsize=8)
    fig.suptitle("Prediction error on b and c (model vs oracle)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bc_error_distribution.png"), dpi=150)
    plt.close()

    # --- oracle b*, c* distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, arr, label, unit, color in zip(
        axes,
        [b_fit_arr, c_fit_arr],
        ["b*", "c*"],
        ["", " (eV)"],
        ["C0", "C1"],
    ):
        ax.hist(arr, bins=50, alpha=0.75, color=color, edgecolor="k")
        ax.axvline(np.mean(arr), color="k", ls="--", lw=1.5, label=f"mean={np.mean(arr):.3f}")
        ax.axvline(np.median(arr), color="r", ls=":", lw=1.5, label=f"med={np.median(arr):.3f}")
        ax.set_xlabel(f"Oracle {label}{unit}")
        ax.set_ylabel("Count")
        ax.set_title(f"Oracle {label} distribution{unit}")
        ax.legend(fontsize=8)
    fig.suptitle("Oracle (b*, c*) fitted to per-crystal residuals", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bc_oracle_distribution.png"), dpi=150)
    plt.close()

    print(f"\n  Part (iii) plots saved to {OUT_DIR}/")
    dataset.close()


if __name__ == "__main__":
    main()
