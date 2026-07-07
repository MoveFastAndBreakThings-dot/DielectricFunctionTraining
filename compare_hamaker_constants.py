"""
Compute Hamaker constant A from ε(iξ) (KKR imaginary-axis dielectric data)
and compare A from model predictions vs ground truth.

Finite-temperature Matsubara Lifshitz formula:

    A = (3/2) k_B T [ Δ_0²/2 + Σ_{n≥1} Δ_n² ]     (J)
    Δ_n = (ε(iξ_n) − 1) / (ε(iξ_n) + 1)
    ξ_n = n · 2π k_B T   (eV at T=300 K by default)

ε(iξ) is interpolated from the model's KKR grid onto Matsubara frequencies.

Usage:
    python3 compare_hamaker_constants.py --model_path models/mace_kkr_v1.pt
    python3 compare_hamaker_constants.py --model_path models/mace_kkr_v1.pt --split test
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward
from model_loader import load_model

T_K = 300.0
K_B_J = 1.380649e-23
K_B_EV = 8.617333e-5
XI_STEP_EV = 2 * np.pi * K_B_EV * T_K
ZJ_PER_J = 1e21
OUT_DIR = "hamaker_comparison"


def get_matsubara_sequences(xi_ev, eps_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=500):
    xi_ev = np.asarray(xi_ev, dtype=np.float64)
    eps_xi = np.asarray(eps_xi, dtype=np.float64)
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


def hamaker_constant_j(xi_ev, eps_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=500):
    """Hamaker constant in joules."""
    _, delta_n = get_matsubara_sequences(
        xi_ev, eps_xi, t_k=t_k, xi_step_ev=xi_step_ev, n_max=n_max,
    )
    pref = 1.5 * K_B_J * t_k
    a = 0.5 * pref * delta_n[0] ** 2
    for n in range(1, n_max):
        a += pref * delta_n[n] ** 2
    return float(a)


def hamaker_constant_zj(xi_ev, eps_xi, **kwargs):
    return hamaker_constant_j(xi_ev, eps_xi, **kwargs) * ZJ_PER_J


def hamaker_pair_constant_j(xi_ev, eps1_xi, eps2_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=500):
    """
    Pairwise Hamaker constant A_12 (joules) across vacuum at finite T.

    A_12 = (3/2) k_B T [ (1/2) Δ_1(0)Δ_2(0) + Σ_{n≥1} Δ_1(n)Δ_2(n) ]
    Δ_k(n) = (ε_k(iξ_n) − 1) / (ε_k(iξ_n) + 1)
    """
    _, delta1 = get_matsubara_sequences(
        xi_ev, eps1_xi, t_k=t_k, xi_step_ev=xi_step_ev, n_max=n_max,
    )
    _, delta2 = get_matsubara_sequences(
        xi_ev, eps2_xi, t_k=t_k, xi_step_ev=xi_step_ev, n_max=n_max,
    )
    pref = 1.5 * K_B_J * t_k
    a = 0.5 * pref * delta1[0] * delta2[0]
    a += pref * np.sum(delta1[1:] * delta2[1:])
    return float(a)


def hamaker_pair_constant_zj(xi_ev, eps1_xi, eps2_xi, **kwargs):
    return hamaker_pair_constant_j(xi_ev, eps1_xi, eps2_xi, **kwargs) * ZJ_PER_J


def matsubara_term_contributions_j(delta_n, t_k=T_K):
    """Per-mode contributions to A (joules). sum(terms) == hamaker_constant_j."""
    delta_n = np.asarray(delta_n, dtype=np.float64)
    pref = 1.5 * K_B_J * t_k
    terms = np.empty_like(delta_n)
    terms[0] = 0.5 * pref * delta_n[0] ** 2
    terms[1:] = pref * delta_n[1:] ** 2
    return terms


def terms_from_eps(xi_ev, eps_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=500):
    """Return (terms_j, xi_matsubara_ev) for n = 0 .. n_max-1."""
    _, delta_n = get_matsubara_sequences(
        xi_ev, eps_xi, t_k=t_k, xi_step_ev=xi_step_ev, n_max=n_max,
    )
    terms = matsubara_term_contributions_j(delta_n, t_k=t_k)
    xi_m = np.arange(n_max, dtype=np.float64) * xi_step_ev
    return terms, xi_m


def train_val_test_split(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return idx[:t].tolist(), idx[t:t + v].tolist(), idx[t + v:].tolist()


def get_split_indices(ckpt, n_samples, split, seed=42):
    key = f"{split}_idx"
    if key in ckpt:
        return ckpt[key]
    train_idx, val_idx, test_idx = train_val_test_split(n_samples, seed=seed)
    return {"train": train_idx, "val": val_idx, "test": test_idx}[split]


def print_stats(name, arr, unit=""):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        print(f"\n  {name}{unit}: (no finite values)")
        return
    p = np.percentile(arr, [5, 25, 50, 75, 95])
    print(f"\n  {name}{unit}:")
    print(f"    n           = {len(arr)}")
    print(f"    mean +- std = {arr.mean():.4f} +- {arr.std():.4f}")
    print(f"    median      = {np.median(arr):.4f}")
    print(f"    min / max   = {arr.min():.4f} / {arr.max():.4f}")
    print(f"    p5/25/50/75/95 = "
          f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f} / {p[3]:.3f} / {p[4]:.3f}")


def report_hamaker_group(group_label, a_true, a_pred, metal_eps_threshold):
    """Print Hamaker comparison stats for one subset (metals, insulators, or all)."""
    a_true = np.asarray(a_true, dtype=float)
    a_pred = np.asarray(a_pred, dtype=float)
    n = len(a_true)
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {group_label}  (n = {n})")
    print(sep)
    if n == 0:
        print("  (empty subset — skipped)")
        return

    diff = a_pred - a_true
    rel_diff = diff / (np.abs(a_true) + 1e-30)
    rel_pct = 100.0 * rel_diff

    print_stats("A_true (actual ε)", a_true, " zJ")
    print_stats("A_pred (predicted ε)", a_pred, " zJ")
    print_stats("ΔA = A_pred − A_true", diff, " zJ")
    print_stats("Relative error (A_pred−A_true)/A_true", rel_diff)
    print_stats("Relative error (%)", rel_pct, " %")

    if n > 1:
        r = np.corrcoef(a_pred, a_true)[0, 1]
    else:
        r = float("nan")
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    print(f"\n  Pearson r(A_pred, A_true) = {r:.4f}")
    print(f"  RMSE(ΔA)                = {rmse:.4f} zJ")
    print(f"  MAE(ΔA)                 = {mae:.4f} zJ")
    if "All crystals" in group_label:
        print(f"  (metal filter: ε_true at lowest ξ > {metal_eps_threshold} → metal)")


def load_eps_model(ckpt, device):
    model, model_type = load_model(ckpt, device)
    return model, model_type


def main():
    parser = argparse.ArgumentParser(description="Compare Hamaker A from predicted vs actual ε(iξ)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0,
                        help="Exclude crystals with ε_true at lowest ξ above this (Drude metals)")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    xi_step_ev = 2 * np.pi * K_B_EV * args.temperature_k

    ckpt = torch.load(args.model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_eps_model(ckpt, device)
    use_log = ckpt.get("use_log", False)
    use_mace = model_type == "MACE"
    target_components = ckpt.get("target_components", ("xx",))

    print(f"Model: {ckpt.get('model_name', args.model_path)} ({model_type})")
    print(f"  split={args.split}  use_log={use_log}  T={args.temperature_k} K")
    print(f"  n_matsubara={args.n_matsubara}  ξ_step={xi_step_ev:.4f} eV")

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt.get("mace_r_max"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    split_idx = get_split_indices(ckpt, len(dataset), args.split)
    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )

    hamaker_kw = dict(t_k=args.temperature_k, xi_step_ev=xi_step_ev, n_max=args.n_matsubara)

    mpids, formulas = [], []
    a_true_list, a_pred_list = [], []
    eps_low_list = []

    with torch.no_grad():
        batch_start = 0
        for inputs, target, _ in loader:
            target = target.to(device)
            eps_pred = model_forward(model, model_type, inputs, device, use_log=use_log)
            eps_pred = eps_pred.unsqueeze(1).cpu().numpy()
            eps_true = target.cpu().numpy()

            bs = eps_true.shape[0]
            for b in range(bs):
                global_i = split_idx[batch_start + b]
                info = dataset.get_sample_info(global_i)
                mpids.append(info["mpid"])
                formulas.append(info["formula"])

                eps_t = eps_true[b, 0, :]
                eps_p = eps_pred[b, 0, :]
                eps_low_list.append(float(eps_t[0]))
                a_true_list.append(hamaker_constant_zj(xi, eps_t, **hamaker_kw))
                a_pred_list.append(hamaker_constant_zj(xi, eps_p, **hamaker_kw))
            batch_start += bs

    dataset.close()

    a_true_all = np.array(a_true_list)
    a_pred_all = np.array(a_pred_list)
    eps_low = np.array(eps_low_list)
    insulator_mask = eps_low <= args.metal_eps_threshold
    metal_mask = ~insulator_mask

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"Hamaker constant comparison ({args.split} set)")
    print(sep)
    print(f"  Crystals total  : {len(a_true_all)}")
    print(f"  Insulators      : {int(insulator_mask.sum())}  "
          f"(ε_true at lowest ξ <= {args.metal_eps_threshold})")
    print(f"  Metals          : {int(metal_mask.sum())}  "
          f"(ε_true at lowest ξ >  {args.metal_eps_threshold})")

    report_hamaker_group(
        "All crystals (insulators + metals)",
        a_true_all, a_pred_all, args.metal_eps_threshold,
    )
    report_hamaker_group(
        "Insulators only",
        a_true_all[insulator_mask], a_pred_all[insulator_mask], args.metal_eps_threshold,
    )
    report_hamaker_group(
        "Metals only",
        a_true_all[metal_mask], a_pred_all[metal_mask], args.metal_eps_threshold,
    )

    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    prefix = os.path.join(args.out_dir, f"{model_stem}_{args.split}")

    a_true_ins = a_true_all[insulator_mask]
    a_pred_ins = a_pred_all[insulator_mask]
    diff_ins = a_pred_ins - a_true_ins
    rel_pct_ins = 100.0 * diff_ins / (np.abs(a_true_ins) + 1e-30)
    r_ins = (np.corrcoef(a_pred_ins, a_true_ins)[0, 1]
             if len(a_true_ins) > 1 else float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(a_true_all[metal_mask], a_pred_all[metal_mask],
                    s=10, alpha=0.35, color="C1", label="metal")
    axes[0].scatter(a_true_ins, a_pred_ins, s=10, alpha=0.35, color="C0", label="insulator")
    lim = max(a_true_all.max(), a_pred_all.max()) * 1.05
    axes[0].plot([0, lim], [0, lim], "k--", lw=1)
    axes[0].set_xlabel("A_true (zJ)")
    axes[0].set_ylabel("A_pred (zJ)")
    axes[0].set_title(f"Hamaker constant  (insulators r={r_ins:.3f})")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].hist(diff_ins, bins=50, color="C0", alpha=0.85, edgecolor="k", linewidth=0.3,
                 label="insulators")
    if metal_mask.any():
        diff_met = a_pred_all[metal_mask] - a_true_all[metal_mask]
        axes[1].hist(diff_met, bins=50, color="C1", alpha=0.55, edgecolor="k", linewidth=0.3,
                     label="metals")
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("ΔA = A_pred − A_true (zJ)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Hamaker prediction error")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    scatter_path = f"{prefix}_hamaker_scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rel_pct_ins, bins=50, color="C0", alpha=0.85, edgecolor="k", linewidth=0.3,
            label="insulators")
    if metal_mask.any():
        rel_met = 100.0 * (a_pred_all[metal_mask] - a_true_all[metal_mask]) / (
            np.abs(a_true_all[metal_mask]) + 1e-30)
        ax.hist(rel_met, bins=50, color="C1", alpha=0.55, edgecolor="k", linewidth=0.3,
                label="metals")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Relative error (%)")
    ax.set_ylabel("Count")
    ax.set_title("(A_pred − A_true) / A_true  (%)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    rel_path = f"{prefix}_hamaker_rel_error.png"
    plt.savefig(rel_path, dpi=150, bbox_inches="tight")
    plt.close()

    csv_path = f"{prefix}_hamaker_values.csv"
    with open(csv_path, "w") as f:
        f.write("mpid,formula,class,A_true_zJ,A_pred_zJ,delta_A_zJ,rel_error_pct,eps_lowest_xi\n")
        for mpid, formula, at, ap, el, is_ins in zip(
            mpids, formulas, a_true_all, a_pred_all, eps_low, insulator_mask,
        ):
            d = ap - at
            rp = 100.0 * d / (at + 1e-30)
            cls = "insulator" if is_ins else "metal"
            f.write(f"{mpid},{formula},{cls},{at:.6e},{ap:.6e},{d:.6e},{rp:.4f},{el:.6e}\n")

    print(f"\n  Saved: {scatter_path}")
    print(f"  Saved: {rel_path}")
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()
