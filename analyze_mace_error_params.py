"""
Analyze a trained FrequencyMACEWithABC checkpoint.

Pipeline (per test crystal):
  1. Model inference  -> eps_pred, b_hat, c_hat
  2. Oracle fit       -> b*, c*  by fitting sigma(xi)=b/(xi+c)^alpha to |eps_pred-eps_true|
  3. b/c error        -> (b_hat-b*), (c_hat-c*): MAE, RMSE, bias, scatter, distributions
  4. Sigma calibration-> compare sigma_pred(xi) and sigma_oracle(xi) to actual residuals,
                         and their 2.5x bands
  5. sigma_A          -> propagate 2.5*sigma through dA/d_eps_n (uncorrelated) for
                         both predicted and oracle (b,c); report and compare distributions

Usage:
    python3 analyze_mace_error_params.py
    python3 analyze_mace_error_params.py --model_path models/mace_kkr_with_abc_v2.pt
"""

import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses_mace_error_params import SpectrumAndPowerLawABCLoss
from mace_model_error_params import FrequencyMACEWithABC

# Physical constants (Lifshitz / Matsubara)
T_K = 300.0
K_B_J = 1.380649e-23        # J/K
K_B_EV = 8.617333e-5        # eV/K
XI_STEP_EV = 2 * np.pi * K_B_EV * T_K   # ~0.1619 eV  (Matsubara spacing)
ZJ_PER_J = 1e21             # 1 zJ = 10^-21 J
N_MATSUBARA = 5000


def get_matsubara_sequences(xi_ev, eps_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=None):
    """
    Return (d_n, delta_n) arrays for n=0..n_max-1.
    d_n = ε(iξ_n),  Δ_n = (d_n − 1)/(d_n + 1).
    Used for error propagation: ∂A/∂d_n and σ(ξ_n).
    """
    if n_max is None:
        n_max = N_MATSUBARA
    xi_ev = np.asarray(xi_ev)
    eps_xi = np.asarray(eps_xi)
    order = np.argsort(xi_ev)
    xi_ev = xi_ev[order]
    eps_xi = eps_xi[order]
    if len(xi_ev) >= 4:
        interp = interp1d(xi_ev, eps_xi, kind="cubic",
                          bounds_error=False, fill_value=(eps_xi[0], 1.0))
    else:
        interp = interp1d(xi_ev, eps_xi, kind="linear",
                          bounds_error=False, fill_value=(eps_xi[0], 1.0))
    d_n = np.zeros(n_max)
    delta_n = np.zeros(n_max)
    d_n[0] = float(interp(0.0))
    delta_n[0] = (d_n[0] - 1.0) / (d_n[0] + 1.0)
    for n in range(1, n_max):
        xi_n = n * xi_step_ev
        d_n[n] = float(interp(xi_n))
        delta_n[n] = (d_n[n] - 1.0) / (d_n[n] + 1.0)
    return d_n, delta_n

OUT_DIR = "analysis_error_params"
os.makedirs(OUT_DIR, exist_ok=True)

K_SIGMA = 2.5  # band multiplier used throughout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigma_A_from_bc(xi_ev, eps_xi, b, c, alpha, k_sigma=K_SIGMA,
                    n_max=500, xi_step_ev=XI_STEP_EV, t_k=T_K):
    """
    sigma_A = sqrt( sum_n (dA/d_eps_n)^2 * (k_sigma * sigma(xi_n))^2 )
    where sigma(xi_n) = b / (xi_n + c)^alpha.

    k_sigma scales the per-frequency error (default 2.5 for a conservative band).
    Returns sigma_A in zJ.
    """
    d_n, delta_n = get_matsubara_sequences(xi_ev, eps_xi, t_k=t_k,
                                           xi_step_ev=xi_step_ev, n_max=n_max)
    pref = 1.5 * K_B_J * t_k
    var_a = 0.0
    for n in range(n_max):
        xi_n = n * xi_step_ev
        den = d_n[n] + 1.0
        if abs(den) < 1e-15:
            continue
        dA_ddn = pref * (2.0 if n == 0 else 4.0) * delta_n[n] / (den ** 2)
        s_n = k_sigma * b / ((xi_n + c) ** alpha)
        var_a += (dA_ddn * s_n) ** 2
    return float(np.sqrt(var_a) * ZJ_PER_J)


def fit_bc_oracle(residuals, xi_ev, alpha_val):
    """
    Fit sigma(xi) = b/(xi+c)^alpha to a per-crystal residual curve.
    Optimises in log-space for numerical stability.
    Returns (b*, c*) or None on failure.
    """
    mask = residuals > 1e-9
    if mask.sum() < 5:
        return None
    xi_m, r_m = xi_ev[mask], residuals[mask]

    def loss_fn(params):
        b_ = np.exp(params[0])
        c_ = np.exp(params[1])
        sigma_ = b_ / (xi_m + c_) ** alpha_val
        return np.mean((np.log1p(sigma_) - np.log1p(r_m)) ** 2)

    best_val, best_res = np.inf, None
    for log_b0, log_c0 in [(np.log(1.0), np.log(0.5)), (np.log(5.0), np.log(1.0)),
                            (np.log(20.0), np.log(2.0)), (np.log(50.0), np.log(0.1))]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(loss_fn, [log_b0, log_c0], method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-7})
        if res.fun < best_val:
            best_val, best_res = res.fun, res
    if best_res is None or (not best_res.success and best_res.fun > 1.0):
        return None
    return float(np.exp(best_res.x[0])), float(np.exp(best_res.x[1]))


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


def print_stats(name, arr, unit=""):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"\n  {name}{unit}:")
    print(f"    n           = {len(arr)}")
    print(f"    mean +- std = {arr.mean():.4f} +- {arr.std():.4f}")
    print(f"    median      = {np.median(arr):.4f}")
    print(f"    min / max   = {arr.min():.4f} / {arr.max():.4f}")
    print(f"    p5/25/50/75/95 = "
          f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f} / {p[3]:.3f} / {p[4]:.3f}")


def hist_cdf(arr, xlabel, title, save_path, color="C0", vlines=None):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(arr, bins=50, alpha=0.8, color=color, edgecolor="k", linewidth=0.3)
    axes[1].plot(np.sort(arr), np.linspace(0, 1, len(arr)), color=color, lw=1.5)
    for ax in axes:
        ax.axvline(arr.mean(), color="k", ls="--", lw=1.3, label=f"mean={arr.mean():.3g}")
        ax.axvline(np.median(arr), color="r", ls=":", lw=1.3, label=f"med={np.median(arr):.3g}")
        if vlines:
            for v, lbl in vlines:
                ax.axvline(v, color="grey", ls="-.", lw=1.0, label=lbl)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("CDF")
    axes[0].set_title(f"{title} — histogram")
    axes[1].set_title(f"{title} — CDF")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/mace_kkr_with_abc_v3_stage2.pt")
    parser.add_argument("--lmdb_path", default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k_sigma", type=float, default=K_SIGMA,
                        help="Multiplier on sigma for the uncertainty band (default 2.5)")
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0,
                        help="Crystals with eps_true at lowest xi > this are flagged as metals "
                             "and excluded. Set to inf to keep all. (default 50)")
    parser.add_argument("--max_oracle_b", type=float, default=1e5,
                        help="Oracle b* fits above this are treated as failed and excluded "
                             "(runaway fits on near-metallic crystals). (default 1e5)")
    args = parser.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(ckpt, device)

    use_log = ckpt.get("use_log", False)
    alpha = ckpt.get("alpha", 3.0)
    sigma_transform = ckpt.get("sigma_transform", "none")
    target_components = ckpt.get("target_components", ("xx",))

    print(f"Model: {ckpt.get('model_name', args.model_path)}")
    print(f"  use_log={use_log}  alpha={alpha}  k_sigma={args.k_sigma}")

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True)
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    test_idx = ckpt.get("test_idx")
    if test_idx is None:
        rng = np.random.RandomState(42)
        n = len(dataset)
        idx = rng.permutation(n)
        train_sz, val_sz = int(n * 0.8), int(n * 0.1)
        test_idx = idx[train_sz + val_sz:].tolist()

    loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_pool)

    criterion = SpectrumAndPowerLawABCLoss(
        xi_array=dataset.frequency_array, alpha=alpha,
        lambda_eps=ckpt.get("lambda_eps", 1.0),
        lambda_sigma=ckpt.get("lambda_sigma", 1.0),
        eps_reduction="mae", sigma_reduction="mse",
        sigma_transform=sigma_transform,
    ).to(device)
    criterion.eps_pred_is_log = use_log

    # -----------------------------------------------------------------------
    # Step 1 — inference: collect eps_pred, eps_true, b_hat, c_hat
    # -----------------------------------------------------------------------
    sep = "=" * 62
    print(f"\n{sep}")
    print("Step 1: model inference on test set")
    print(sep)

    total_loss = total_loss_eps = total_loss_sigma = n_total = 0
    eps_pred_all, eps_true_all, b_hat_all, c_hat_all = [], [], [], []

    with torch.no_grad():
        for inputs, target, _ in loader:
            af, nf, nfi, cai = inputs
            af = af.to(device); nf = nf.to(device)
            nfi = nfi.to(device)
            cai = [x.to(device) for x in cai]
            target = target.to(device)

            eps_out, bc_out = model(af, nf, nfi, cai)
            loss, parts = criterion(eps_out, bc_out, target)

            bs = target.shape[0]
            total_loss += loss.item() * bs
            total_loss_eps += parts["loss_eps"] * bs
            total_loss_sigma += parts["loss_sigma"] * bs
            n_total += bs

            eps_np = (torch.exp(eps_out).cpu().numpy() if use_log
                      else eps_out.cpu().numpy())
            eps_pred_all.append(eps_np)
            eps_true_all.append(target.cpu().numpy())
            bc_np = bc_out.cpu().numpy()
            b_hat_all.append(bc_np[:, 0])
            c_hat_all.append(bc_np[:, 1])

    eps_pred_all = np.concatenate(eps_pred_all, axis=0)   # (N, 1, F)
    eps_true_all = np.concatenate(eps_true_all, axis=0)
    b_hat = np.concatenate(b_hat_all)   # (N,)
    c_hat = np.concatenate(c_hat_all)

    print(f"  Test loss total    = {total_loss/n_total:.6f}")
    print(f"  Test loss eps part = {total_loss_eps/n_total:.6f}")
    print(f"  Test loss sig part = {total_loss_sigma/n_total:.6f}")
    print(f"  N test crystals    = {n_total}")

    # ---- metal filter ----
    # eps_true[:, 0, 0] is ε at the lowest imaginary frequency.
    # For metals ε(iξ→0) → ∞ (Drude divergence); insulators are finite.
    eps_low = eps_true_all[:, 0, 0]
    insulator_mask = eps_low <= args.metal_eps_threshold
    n_metals = int((~insulator_mask).sum())
    n_insulators = int(insulator_mask.sum())
    print(f"\n  Metal filter (eps_true at lowest xi > {args.metal_eps_threshold}):")
    print(f"    Metals excluded : {n_metals}")
    print(f"    Insulators kept : {n_insulators}")

    # apply mask — all downstream analysis uses only insulators
    eps_pred_all  = eps_pred_all[insulator_mask]
    eps_true_all  = eps_true_all[insulator_mask]
    b_hat         = b_hat[insulator_mask]
    c_hat         = c_hat[insulator_mask]
    test_idx_ins  = [test_idx[i] for i in range(n_total) if insulator_mask[i]]
    n_total       = n_insulators

    # -----------------------------------------------------------------------
    # Step 2 — oracle fit: (b*, c*) per crystal
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("Step 2: oracle (b*, c*) fit to actual per-crystal residuals")
    print(f"  sigma(xi) = b/(xi+c)^alpha,  alpha={alpha:.3f}")
    print(f"  Runaway fits with b* > {args.max_oracle_b:.0e} also excluded")
    print(sep)

    b_star_list, c_star_list = [], []
    valid_mask = np.zeros(n_total, dtype=bool)
    n_runaway = 0

    for i in range(n_total):
        res = np.abs(eps_pred_all[i, 0, :] - eps_true_all[i, 0, :])
        result = fit_bc_oracle(res, xi, alpha)
        if result is not None and result[0] <= args.max_oracle_b:
            b_star_list.append(result[0])
            c_star_list.append(result[1])
            valid_mask[i] = True
        else:
            if result is not None and result[0] > args.max_oracle_b:
                n_runaway += 1
            b_star_list.append(np.nan)
            c_star_list.append(np.nan)

    b_star = np.array(b_star_list)
    c_star = np.array(c_star_list)
    n_ok = valid_mask.sum()
    n_fit_failed = n_total - n_ok - n_runaway
    print(f"  Fitted successfully : {n_ok} / {n_total}")
    print(f"  Runaway fits (b*>{args.max_oracle_b:.0e}) excluded: {n_runaway}")
    print(f"  Convergence failures: {n_fit_failed}")

    # -----------------------------------------------------------------------
    # Step 3 — b and c prediction error
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("Step 3: b and c prediction error  (b_hat - b*,  c_hat - c*)")
    print(sep)

    bh = b_hat[valid_mask]
    ch = c_hat[valid_mask]
    bs = b_star[valid_mask]
    cs = c_star[valid_mask]

    b_err = bh - bs
    c_err = ch - cs

    for name, err, unit in [("b_hat - b*", b_err, ""),
                             ("c_hat - c*", c_err, " eV")]:
        print_stats(name, err, unit)
        print(f"    MAE  = {np.mean(np.abs(err)):.4f}{unit}")
        print(f"    RMSE = {np.sqrt(np.mean(err**2)):.4f}{unit}")
        print(f"    bias = {np.mean(err):.4f}{unit}  "
              f"({'over' if np.mean(err)>0 else 'under'}-predicted)")

    corr_b = np.corrcoef(bh, bs)[0, 1]
    corr_c = np.corrcoef(ch, cs)[0, 1]
    print(f"\n  Pearson r(b_hat, b*) = {corr_b:.4f}")
    print(f"  Pearson r(c_hat, c*) = {corr_c:.4f}")

    # scatter: predicted vs oracle
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pred, fit, label, unit in zip(axes, [bh, ch], [bs, cs], ["b", "c"], ["", " (eV)"]):
        ax.scatter(fit, pred, s=8, alpha=0.35, color="C0")
        lo = min(fit.min(), pred.min()) * 0.9
        hi = max(fit.max(), pred.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
        ax.set_xlabel(f"Oracle {label}*{unit}")
        ax.set_ylabel(f"Model {label}_hat{unit}")
        ax.set_title(f"{label}  MAE={np.mean(np.abs(pred-fit)):.3f}{unit}  r={corr_b if label=='b' else corr_c:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs oracle (b*, c*) per test crystal", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bc_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUT_DIR}/bc_scatter.png")

    # error distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, err, label, unit in zip(axes, [b_err, c_err],
                                    ["b_hat - b*", "c_hat - c*"], ["", " (eV)"]):
        ax.hist(err, bins=50, alpha=0.8, color="C3", edgecolor="k", linewidth=0.3)
        ax.axvline(0, color="k", ls="--", lw=1.2, label="zero error")
        ax.axvline(np.mean(err), color="r", ls=":", lw=1.5, label=f"mean={np.mean(err):.3f}")
        ax.set_xlabel(f"{label}{unit}")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {label}{unit}")
        ax.legend(fontsize=8)
    fig.suptitle("b and c prediction error distributions", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bc_error_dist.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/bc_error_dist.png")

    # -----------------------------------------------------------------------
    # Step 4 — sigma curve error vs actual residuals
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print("Step 4: sigma curve calibration vs actual |eps_pred - eps_true|")
    print(f"  Comparing sigma_pred(xi)={args.k_sigma}*b_hat/(xi+c_hat)^a "
          f"and sigma_oracle(xi)={args.k_sigma}*b*/(xi+c*)^a to actual residuals")
    print(sep)

    rmse_pred_list, rmse_oracle_list, coverage_pred_list, coverage_oracle_list = [], [], [], []

    for i in np.where(valid_mask)[0]:
        res_i = np.abs(eps_pred_all[i, 0, :] - eps_true_all[i, 0, :])
        sig_pred_i = args.k_sigma * b_hat[i] / (xi + c_hat[i]) ** alpha
        sig_oracle_i = args.k_sigma * b_star[i] / (xi + c_star[i]) ** alpha

        rmse_pred_list.append(np.sqrt(np.mean((sig_pred_i - res_i) ** 2)))
        rmse_oracle_list.append(np.sqrt(np.mean((sig_oracle_i - res_i) ** 2)))
        coverage_pred_list.append(float(np.mean(res_i <= sig_pred_i)))
        coverage_oracle_list.append(float(np.mean(res_i <= sig_oracle_i)))

    rmse_pred = np.array(rmse_pred_list)
    rmse_oracle = np.array(rmse_oracle_list)
    cov_pred = np.array(coverage_pred_list)
    cov_oracle = np.array(coverage_oracle_list)

    print_stats("RMSE(sigma_pred vs residuals)", rmse_pred)
    print_stats("RMSE(sigma_oracle vs residuals)", rmse_oracle)
    print(f"\n  Coverage (fraction of xi where residual <= {args.k_sigma}*sigma):")
    print(f"    Predicted  — mean={cov_pred.mean():.3f}  median={np.median(cov_pred):.3f}")
    print(f"    Oracle     — mean={cov_oracle.mean():.3f}  median={np.median(cov_oracle):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rmse_pred, bins=50, alpha=0.6, color="C0", label="predicted (b_hat,c_hat)", edgecolor="k", lw=0.3)
    axes[0].hist(rmse_oracle, bins=50, alpha=0.6, color="C1", label="oracle (b*,c*)", edgecolor="k", lw=0.3)
    axes[0].set_xlabel("RMSE of sigma curve vs actual residuals")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Sigma curve RMSE  (k={args.k_sigma})")
    axes[0].legend(fontsize=8)
    axes[1].hist(cov_pred, bins=50, alpha=0.6, color="C0", label="predicted", edgecolor="k", lw=0.3)
    axes[1].hist(cov_oracle, bins=50, alpha=0.6, color="C1", label="oracle", edgecolor="k", lw=0.3)
    axes[1].axvline(1.0, color="k", ls="--", lw=1)
    axes[1].set_xlabel(f"Coverage fraction (residual <= {args.k_sigma}*sigma)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-crystal coverage")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sigma_calibration.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/sigma_calibration.png")

    # -----------------------------------------------------------------------
    # Step 5 — sigma_A with k_sigma multiplier (predicted and oracle)
    # -----------------------------------------------------------------------
    print(f"\n{sep}")
    print(f"Step 5: sigma_A with {args.k_sigma}x multiplier  (uncorrelated propagation)")
    print(f"  sigma_A^2 = sum_n (dA/d_eps_n)^2 * ({args.k_sigma}*sigma(xi_n))^2")
    print(f"  n_matsubara = {args.n_matsubara}")
    print(sep)

    sA_pred_list, sA_oracle_list = [], []
    valid_indices = np.where(valid_mask)[0]

    for j, i in enumerate(valid_indices):
        eps_i = eps_pred_all[i, 0, :]

        sA_p = sigma_A_from_bc(xi, eps_i, b_hat[i], c_hat[i], alpha,
                                k_sigma=args.k_sigma, n_max=args.n_matsubara)
        sA_o = sigma_A_from_bc(xi, eps_i, b_star[i], c_star[i], alpha,
                                k_sigma=args.k_sigma, n_max=args.n_matsubara)
        sA_pred_list.append(sA_p)
        sA_oracle_list.append(sA_o)

        if (j + 1) % 200 == 0:
            print(f"  {j+1}/{len(valid_indices)} crystals done...")

    sA_pred = np.array(sA_pred_list)
    sA_oracle = np.array(sA_oracle_list)
    ratio = sA_pred / (sA_oracle + 1e-30)

    print_stats(f"sigma_A predicted  ({args.k_sigma}x band)", sA_pred, " zJ")
    print_stats(f"sigma_A oracle     ({args.k_sigma}x band)", sA_oracle, " zJ")
    print_stats("ratio sigma_A_pred / sigma_A_oracle", ratio)

    hist_cdf(sA_pred, f"sigma_A (zJ)", f"sigma_A from predicted (b_hat,c_hat)  k={args.k_sigma}",
             os.path.join(OUT_DIR, "sigmaA_pred.png"), color="C0")
    hist_cdf(sA_oracle, f"sigma_A (zJ)", f"sigma_A from oracle (b*,c*)  k={args.k_sigma}",
             os.path.join(OUT_DIR, "sigmaA_oracle.png"), color="C1")
    hist_cdf(ratio, "sigma_A_pred / sigma_A_oracle", "Ratio of sigma_A (predicted / oracle)",
             os.path.join(OUT_DIR, "sigmaA_ratio.png"), color="C3")

    # overlay comparison
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(sA_pred, bins=50, alpha=0.55, color="C0", label="predicted (b_hat,c_hat)", edgecolor="k", lw=0.3)
    axes[0].hist(sA_oracle, bins=50, alpha=0.55, color="C1", label="oracle (b*,c*)", edgecolor="k", lw=0.3)
    axes[0].set_xlabel(f"sigma_A (zJ)  [k={args.k_sigma}]")
    axes[0].set_ylabel("Count")
    axes[0].set_title("sigma_A distribution comparison")
    axes[0].legend(fontsize=9)

    axes[1].scatter(sA_oracle, sA_pred, s=8, alpha=0.35, color="C4")
    lim = max(sA_oracle.max(), sA_pred.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    axes[1].set_xlabel(f"sigma_A oracle (zJ)")
    axes[1].set_ylabel(f"sigma_A predicted (zJ)")
    axes[1].set_title(f"sigma_A: predicted vs oracle  (r={np.corrcoef(sA_pred, sA_oracle)[0,1]:.3f})")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sigmaA_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR}/sigmaA_comparison.png")

    dataset.close()
    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
