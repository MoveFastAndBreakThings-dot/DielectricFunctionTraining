"""
Compare error in each Matsubara *term* of the Hamaker sum (not raw ε(iξ)).

    A = Σ_n  T_n
    T_0 = (3/2) k_B T · (1/2) · Δ_0²
    T_n = (3/2) k_B T · Δ_n²           (n ≥ 1)
    Δ_n = (ε(iξ_n) − 1) / (ε(iξ_n) + 1)
    ξ_n = n · 2π k_B T

For each crystal we compare T_n(pred) vs T_n(true), then aggregate over the test
split — with emphasis on low-n (low ξ) terms, especially for metals.

Usage:
    python3 compare_hamaker_term_errors.py --model_path models/mace_kkr_v1.pt
    python3 compare_hamaker_term_errors.py --model_path models/mace_kkr_v1.pt --n_low 15
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from compare_hamaker_constants import (
    K_B_J,
    K_B_EV,
    T_K,
    XI_STEP_EV,
    ZJ_PER_J,
    get_matsubara_sequences,
    get_split_indices,
    load_eps_model,
    print_stats,
)
from lmdb_dataset import LMDBCrystalDataset, collate_pool

OUT_DIR = "hamaker_term_comparison"


def matsubara_term_contributions_j(delta_n, t_k=T_K):
    """
    Per-mode contributions to A (joules).  sum(terms) == hamaker_constant_j.
    """
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


def report_term_group(label, terms_true, terms_pred, xi_m, n_low):
    """
    terms_* : (N_crystals, n_max) in zJ
  """
    n_crystals, n_max = terms_true.shape
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {label}  (n_crystals = {n_crystals}, n_matsubara = {n_max})")
    print(sep)

    if n_crystals == 0:
        print("  (empty — skipped)")
        return None

    diff = terms_pred - terms_true
    abs_diff = np.abs(diff)
    # relative error per term; mask tiny true terms
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / (np.abs(terms_true) + 1e-30)

    a_true = terms_true.sum(axis=1)
    a_pred = terms_pred.sum(axis=1)
    print(f"  Total A_true  mean={a_true.mean():.2f} zJ   "
          f"A_pred mean={a_pred.mean():.2f} zJ   "
          f"ΔA mean={(a_pred - a_true).mean():.2f} zJ")

    # --- per Matsubara index n (mean over crystals) ---
    mean_abs_per_n = abs_diff.mean(axis=0)
    mean_rel_per_n = np.nanmean(rel, axis=0)
    frac_of_a_true = terms_true.mean(axis=0) / (a_true.mean() + 1e-30)
    frac_of_abs_err = abs_diff.mean(axis=0) / (abs_diff.mean(axis=0).sum() + 1e-30)

    print(f"\n  Low-n Matsubara terms (n = 0 .. {n_low - 1}, ξ_n = n·2πk_BT):")
    print(f"  {'n':>4}  {'ξ(eV)':>8}  {'⟨T_true⟩':>10}  {'⟨|ΔT|⟩':>10}  "
          f"{'⟨rel err⟩':>10}  {'% of ⟨A⟩':>8}  {'% of ⟨|ΔT|⟩':>10}")
    for n in range(min(n_low, n_max)):
        print(f"  {n:4d}  {xi_m[n]:8.4f}  {terms_true[:, n].mean():10.3f}  "
              f"{mean_abs_per_n[n]:10.3f}  {mean_rel_per_n[n]:10.4f}  "
              f"{100 * frac_of_a_true[n]:7.2f}%  {100 * frac_of_abs_err[n]:9.2f}%")

    # --- low-n vs rest (error budget) ---
    low_slice = slice(0, min(n_low, n_max))
    err_low = abs_diff[:, low_slice].sum(axis=1)
    err_all = abs_diff.sum(axis=1)
    frac_low = err_low / (err_all + 1e-30)
    print(f"\n  Error budget: fraction of Σ_n |ΔT_n| from n < {n_low}:")
    print_stats("  fraction from low-n", frac_low)

    return {
        "mean_abs_per_n": mean_abs_per_n,
        "mean_rel_per_n": mean_rel_per_n,
        "frac_of_a_true": frac_of_a_true,
        "frac_of_abs_err": frac_of_abs_err,
        "abs_diff": abs_diff,
        "terms_true": terms_true,
        "terms_pred": terms_pred,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare per-term Matsubara errors in the Hamaker sum",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--n_low", type=int, default=15,
                        help="Print / highlight first n_low Matsubara terms (low ξ)")
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    xi_step_ev = 2 * np.pi * K_B_EV * args.temperature_k

    ckpt = torch.load(args.model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, has_abc = load_eps_model(ckpt, device)
    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))

    print(f"Model: {ckpt.get('model_name', args.model_path)}")
    print(f"  split={args.split}  use_log={use_log}  T={args.temperature_k} K")
    print(f"  n_matsubara={args.n_matsubara}  n_low={args.n_low}  ξ_step={xi_step_ev:.4f} eV")

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    split_idx = get_split_indices(ckpt, len(dataset), args.split)

    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pool,
    )

    term_kw = dict(t_k=args.temperature_k, xi_step_ev=xi_step_ev, n_max=args.n_matsubara)
    xi_m_ev = np.arange(args.n_matsubara, dtype=np.float64) * xi_step_ev

    terms_true_list, terms_pred_list, eps_low_list = [], [], []

    with torch.no_grad():
        batch_start = 0
        for inputs, target, _ in loader:
            af, nf, nfi, cai = inputs
            af = af.to(device)
            nf = nf.to(device)
            nfi = nfi.to(device)
            cai = [x.to(device) for x in cai]
            target = target.to(device)

            out = model(af, nf, nfi, cai)
            eps_raw = out[0] if has_abc else out
            eps_pred = torch.exp(eps_raw).cpu().numpy() if use_log else eps_raw.cpu().numpy()
            eps_true = target.cpu().numpy()

            bs = eps_true.shape[0]
            for b in range(bs):
                eps_t = eps_true[b, 0, :]
                eps_p = eps_pred[b, 0, :]
                eps_low_list.append(float(eps_t[0]))
                t_true, _ = terms_from_eps(xi, eps_t, **term_kw)
                t_pred, _ = terms_from_eps(xi, eps_p, **term_kw)
                terms_true_list.append(t_true * ZJ_PER_J)
                terms_pred_list.append(t_pred * ZJ_PER_J)
            batch_start += bs

    dataset.close()

    terms_true_all = np.stack(terms_true_list, axis=0)
    terms_pred_all = np.stack(terms_pred_list, axis=0)
    eps_low = np.array(eps_low_list)
    insulator_mask = eps_low <= args.metal_eps_threshold
    metal_mask = ~insulator_mask

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"Hamaker *term* errors ({args.split} set)")
    print(sep)
    print(f"  Each T_n is the n-th addend in A = Σ_n T_n  (units: zJ)")
    print(f"  Insulators: {insulator_mask.sum()}   Metals: {metal_mask.sum()}   "
          f"(ε_true at lowest ξ on grid > {args.metal_eps_threshold} → metal)")

    stats_all = report_term_group(
        "All crystals", terms_true_all, terms_pred_all, xi_m_ev, args.n_low,
    )
    stats_ins = report_term_group(
        "Insulators only",
        terms_true_all[insulator_mask], terms_pred_all[insulator_mask],
        xi_m_ev, args.n_low,
    )
    stats_met = report_term_group(
        "Metals only",
        terms_true_all[metal_mask], terms_pred_all[metal_mask],
        xi_m_ev, args.n_low,
    )

    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]
    prefix = os.path.join(args.out_dir, f"{model_stem}_{args.split}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    n_plot = min(args.n_matsubara, 120)
    x = xi_m_ev[:n_plot]

    for ax, stats, title, color in [
        (axes[0, 0], stats_all, "All", "C0"),
        (axes[0, 1], stats_ins, "Insulators", "C0"),
        (axes[1, 0], stats_met, "Metals", "C1"),
    ]:
        if stats is None:
            ax.set_visible(False)
            continue
        ax.plot(x, stats["mean_abs_per_n"][:n_plot], color=color, lw=1.5, label="⟨|ΔT_n|⟩")
        ax.axvline(xi_m_ev[min(args.n_low, n_plot - 1)], color="k", ls=":", alpha=0.6,
                   label=f"n_low={args.n_low}")
        ax.set_xlabel("ξ_n (eV)")
        ax.set_ylabel("Mean |T_pred − T_true| (zJ)")
        ax.set_title(f"{title}: per-term absolute error")
        ax.set_xscale("log")
        ax.legend(fontsize=7)

    ax = axes[1, 1]
    if stats_all is not None:
        ax.plot(x, 100 * stats_all["frac_of_abs_err"][:n_plot], "k-", lw=1.5, label="all")
    if stats_ins is not None:
        ax.plot(x, 100 * stats_ins["frac_of_abs_err"][:n_plot], "C0--", lw=1.2, label="insulator")
    if stats_met is not None:
        ax.plot(x, 100 * stats_met["frac_of_abs_err"][:n_plot], "C1--", lw=1.2, label="metal")
    ax.set_xlabel("ξ_n (eV)")
    ax.set_ylabel("% of mean |ΔT_n| across n")
    ax.set_title("Where Hamaker term error lives")
    ax.set_xscale("log")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = f"{prefix}_term_errors.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # cumulative |ΔT| fraction vs n
    fig, ax = plt.subplots(figsize=(8, 4))
    for stats, label, color in [
        (stats_all, "all", "k"),
        (stats_ins, "insulator", "C0"),
        (stats_met, "metal", "C1"),
    ]:
        if stats is None:
            continue
        cum = np.cumsum(stats["mean_abs_per_n"])
        cum_frac = cum / (cum[-1] + 1e-30)
        ax.plot(xi_m_ev, cum_frac, color=color, lw=1.5, label=label)
    ax.axvline(xi_m_ev[min(args.n_low, len(xi_m_ev) - 1)], color="gray", ls=":", label=f"n={args.n_low}")
    ax.set_xlabel("ξ_n (eV)  (cumulative up to this mode)")
    ax.set_ylabel("Fraction of total ⟨|ΔT_n|⟩")
    ax.set_title("Cumulative term-error budget")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.legend()
    plt.tight_layout()
    cum_path = f"{prefix}_term_error_cumulative.png"
    plt.savefig(cum_path, dpi=150, bbox_inches="tight")
    plt.close()

    csv_path = f"{prefix}_term_errors_per_n.csv"
    with open(csv_path, "w") as f:
        f.write("n,xi_ev,"
                "mean_T_true_all_zJ,mean_abs_dT_all_zJ,mean_rel_err_all,"
                "mean_T_true_ins_zJ,mean_abs_dT_ins_zJ,mean_rel_err_ins,"
                "mean_T_true_met_zJ,mean_abs_dT_met_zJ,mean_rel_err_met\n")
        for n in range(args.n_matsubara):
            row = [str(n), f"{xi_m_ev[n]:.6e}"]
            for stats in (stats_all, stats_ins, stats_met):
                if stats is None:
                    row.extend(["", "", ""])
                else:
                    row.append(f"{stats['terms_true'][:, n].mean():.6e}")
                    row.append(f"{stats['mean_abs_per_n'][n]:.6e}")
                    row.append(f"{stats['mean_rel_per_n'][n]:.6e}")
            f.write(",".join(row) + "\n")

    print(f"\n  Saved: {plot_path}")
    print(f"  Saved: {cum_path}")
    print(f"  Saved: {csv_path}")


if __name__ == "__main__":
    main()
