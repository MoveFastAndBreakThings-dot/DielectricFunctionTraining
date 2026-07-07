"""
Plot cumulative Hamaker sum vs Matsubara index / frequency.

    A(n) = Σ_{k=0}^{n} T_k        (partial sum through mode n)
    A(∞) = full Hamaker constant

At each n we compare DFT (ε_true) vs model (ε_pred):
  - two lines: cumulative A_true(n), cumulative A_pred(n)
  - gap = A_pred(n) − A_true(n)  (above / below DFT as the sum builds)

Outputs go to error_across_frequencies_hamaker_constant/
  - aggregate + convergence PNGs at top level
  - {model}_{split}_per_material/{insulator,metal}/*.png  — one plot per crystal

Usage:
    python3 plot_hamaker_cumulative_frequency.py --model_path models/mace_kkr_new_v1.pt
    python3 plot_hamaker_cumulative_frequency.py --model_path models/mace_kkr_new_v1.pt --no_per_material
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from compare_hamaker_constants import (
    K_B_EV,
    T_K,
    ZJ_PER_J,
    get_split_indices,
    load_eps_model,
    terms_from_eps,
)
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward_train

OUT_DIR = "error_across_frequencies_hamaker_constant"


def safe_filename(s, max_len=80):
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
    return out[:max_len] if len(out) > max_len else out


def cumulative_hamaker_zj(xi_ev, eps_xi, **term_kw):
    """Return (cum_zj, terms_zj, xi_m_ev) for modes 0..n_max-1."""
    terms_j, xi_m = terms_from_eps(xi_ev, eps_xi, **term_kw)
    terms_zj = terms_j * ZJ_PER_J
    cum = np.cumsum(terms_zj)
    return cum, terms_zj, xi_m


def collect_terms(loader, split_idx, dataset, model, model_type, use_log, device, xi, term_kw):
    """Return terms_true, terms_pred (N, n_max) zJ, eps_low, mpids, formulas."""
    terms_true_list, terms_pred_list = [], []
    eps_low_list, mpids, formulas = [], [], []

    with torch.no_grad():
        batch_start = 0
        for inputs, target, _ in loader:
            target = target.to(device)
            out = model_forward_train(model, model_type, inputs, device)
            eps_pred = torch.exp(out).cpu().numpy() if use_log else out.cpu().numpy()
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

                _, t_true, _ = cumulative_hamaker_zj(xi, eps_t, **term_kw)
                _, t_pred, _ = cumulative_hamaker_zj(xi, eps_p, **term_kw)
                terms_true_list.append(t_true)
                terms_pred_list.append(t_pred)
            batch_start += bs

    return (
        np.stack(terms_true_list),
        np.stack(terms_pred_list),
        np.array(eps_low_list),
        mpids,
        formulas,
    )


def plot_aggregate(cum_true, cum_pred, xi_m, mask, label, out_path, n_plot):
    """Mean cumulative A_true vs A_pred with p25–p75 band."""
    ct = cum_true[mask, :n_plot]
    cp = cum_pred[mask, :n_plot]
    if len(ct) == 0:
        return

    x = np.arange(n_plot)
    xi = xi_m[:n_plot]
    m_t, m_p = ct.mean(axis=0), cp.mean(axis=0)
    p25_t, p75_t = np.percentile(ct, [25, 75], axis=0)
    p25_p, p75_p = np.percentile(cp, [25, 75], axis=0)
    gap = m_p - m_t

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(x, p25_t, p75_t, color="C1", alpha=0.25, label="DFT p25–p75")
    ax.fill_between(x, p25_p, p75_p, color="C0", alpha=0.25, label="Model p25–p75")
    ax.plot(x, m_t, color="C1", lw=2, label="DFT mean cumulative A")
    ax.plot(x, m_p, color="C0", lw=2, ls="--", label="Model mean cumulative A")
    ax.set_ylabel("Cumulative Hamaker A (zJ)")
    ax.set_title(f"{label}  (n = {len(ct)}) — partial sum through Matsubara mode n")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(x, 0, gap, where=(gap >= 0), color="C0", alpha=0.4, interpolate=True,
                    label="Model above DFT")
    ax.fill_between(x, 0, gap, where=(gap < 0), color="C3", alpha=0.4, interpolate=True,
                    label="Model below DFT")
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(x, gap, color="k", lw=1.2, label="Mean ΔA(n) = A_pred(n) − A_true(n)")
    ax.set_ylabel("ΔA(n) (zJ)")
    ax.set_xlabel("Matsubara index n")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_idx = np.unique(np.linspace(0, n_plot - 1, min(8, n_plot)).astype(int))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([f"{xi[i]:.2f}" for i in tick_idx])
    ax2.set_xlabel("ξ_n (eV)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_single(cum_true, cum_pred, xi_m, mpid, formula, out_path, n_plot, material_class=None):
    x = np.arange(n_plot)
    xi = xi_m[:n_plot]
    gap = cum_pred[:n_plot] - cum_true[:n_plot]
    final_da = float(cum_pred[-1] - cum_true[-1])
    final_rel = 100.0 * final_da / (float(cum_true[-1]) + 1e-30)
    cls_tag = f"  [{material_class}]" if material_class else ""

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    ax.plot(x, cum_true[:n_plot], color="C1", lw=2, label="DFT (ε_true)")
    ax.plot(x, cum_pred[:n_plot], color="C0", lw=2, ls="--", label="Model (ε_pred)")
    ax.set_ylabel("Cumulative A (zJ)")
    ax.set_title(f"{mpid}  {formula}{cls_tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(x, 0, gap, where=(gap >= 0), color="C0", alpha=0.35)
    ax.fill_between(x, 0, gap, where=(gap < 0), color="C3", alpha=0.35)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(x, gap, "k-", lw=1.2)
    ax.set_ylabel("A_pred(n) − A_true(n) (zJ)")
    ax.set_xlabel("Matsubara index n")
    ax.set_title(f"Final ΔA = {final_da:.2f} zJ  ({final_rel:.2f}%)  [full sum n=0..{len(cum_true)-1}]")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_idx = np.unique(np.linspace(0, n_plot - 1, min(8, n_plot)).astype(int))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([f"{xi[i]:.2f}" for i in tick_idx])
    ax2.set_xlabel("ξ_n (eV)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return final_da, final_rel


def plot_convergence_fraction(cum_true, cum_pred, xi_m, mask, label, out_path, n_plot):
    """Fraction of final Hamaker reached by mode n (how fast the sum converges)."""
    ct = cum_true[mask, :n_plot]
    cp = cum_pred[mask, :n_plot]
    if len(ct) == 0:
        return
    final_t = ct[:, -1:]
    frac_t = ct / (final_t + 1e-30)
    frac_p = cp / (cp[:, -1:] + 1e-30)
    x = np.arange(n_plot)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, frac_t.mean(axis=0), color="C1", lw=2, label="DFT")
    ax.plot(x, frac_p.mean(axis=0), color="C0", lw=2, ls="--", label="Model")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Matsubara index n")
    ax.set_ylabel("A(n) / A(final)")
    ax.set_title(f"{label} — cumulative convergence (mean)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cumulative Hamaker vs Matsubara frequency: DFT vs model",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--n_plot", type=int, default=150,
                        help="Modes 0..n_plot-1 shown on x-axis (rest often plateau)")
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0)
    parser.add_argument("--mpid", type=str, default=None,
                        help="If set, only plot this mpid (must be in split); skips bulk per-material")
    parser.add_argument("--no_per_material", action="store_true",
                        help="Skip plotting every material (aggregate only)")
    parser.add_argument("--n_examples", type=int, default=3,
                        help="Copy worst |final ΔA| into aggregate/ (0 to disable)")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    xi_step_ev = 2 * np.pi * K_B_EV * args.temperature_k
    term_kw = dict(t_k=args.temperature_k, xi_step_ev=xi_step_ev, n_max=args.n_matsubara)

    ckpt = torch.load(args.model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_eps_model(ckpt, device)
    use_log = ckpt.get("use_log", False)
    use_mace = model_type == "MACE"
    target_components = ckpt.get("target_components", ("xx",))
    model_stem = os.path.splitext(os.path.basename(args.model_path))[0]

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt.get("mace_r_max"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    xi_m = np.arange(args.n_matsubara, dtype=np.float64) * xi_step_ev
    n_plot = min(args.n_plot, args.n_matsubara)

    split_idx = get_split_indices(ckpt, len(dataset), args.split)
    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )

    print(f"Model: {ckpt.get('model_name', args.model_path)} ({model_type})  split={args.split}")
    print(f"  n_matsubara={args.n_matsubara}  n_plot={n_plot}  → {args.out_dir}/")

    terms_true, terms_pred, eps_low, mpids, formulas = collect_terms(
        loader, split_idx, dataset, model, model_type, use_log, device, xi, term_kw,
    )
    dataset.close()

    cum_true = np.cumsum(terms_true, axis=1)
    cum_pred = np.cumsum(terms_pred, axis=1)
    insulator_mask = eps_low <= args.metal_eps_threshold
    metal_mask = ~insulator_mask

    prefix = os.path.join(args.out_dir, f"{model_stem}_{args.split}")

    plot_aggregate(cum_true, cum_pred, xi_m, np.ones(len(cum_true), dtype=bool),
                   "All crystals", f"{prefix}_cumulative_all.png", n_plot)
    plot_aggregate(cum_true, cum_pred, xi_m, insulator_mask,
                   "Insulators", f"{prefix}_cumulative_insulators.png", n_plot)
    plot_aggregate(cum_true, cum_pred, xi_m, metal_mask,
                   "Metals", f"{prefix}_cumulative_metals.png", n_plot)

    plot_convergence_fraction(cum_true, cum_pred, xi_m, insulator_mask,
                              "Insulators", f"{prefix}_convergence_insulators.png", n_plot)
    plot_convergence_fraction(cum_true, cum_pred, xi_m, metal_mask,
                              "Metals", f"{prefix}_convergence_metals.png", n_plot)

    final_gap = cum_pred[:, -1] - cum_true[:, -1]
    n_total = len(mpids)

    if args.mpid:
        if args.mpid not in mpids:
            print(f"  WARNING: {args.mpid} not in {args.split} split")
        else:
            i = mpids.index(args.mpid)
            safe = args.mpid.replace("/", "-")
            cls = "insulator" if insulator_mask[i] else "metal"
            out = f"{prefix}_single_{safe}.png"
            plot_single(cum_true[i], cum_pred[i], xi_m, args.mpid, formulas[i],
                        out, n_plot, material_class=cls)
            print(f"  Single: {args.mpid}  →  {out}")
    elif not args.no_per_material:
        per_root = os.path.join(args.out_dir, f"{model_stem}_{args.split}_per_material")
        for sub in ("insulator", "metal"):
            os.makedirs(os.path.join(per_root, sub), exist_ok=True)

        index_path = os.path.join(per_root, "index.csv")
        index_rows = []

        print(f"  Writing {n_total} per-material plots → {per_root}/")
        for i in range(n_total):
            cls = "insulator" if insulator_mask[i] else "metal"
            fname = f"{safe_filename(mpids[i])}_{safe_filename(formulas[i])}.png"
            rel_plot = os.path.join(cls, fname)
            out_path = os.path.join(per_root, rel_plot)
            da, rel_pct = plot_single(
                cum_true[i], cum_pred[i], xi_m, mpids[i], formulas[i],
                out_path, n_plot, material_class=cls,
            )
            index_rows.append([
                mpids[i], formulas[i], cls,
                f"{cum_true[i, -1]:.6e}", f"{cum_pred[i, -1]:.6e}",
                f"{da:.6e}", f"{rel_pct:.4f}", rel_plot,
            ])
            if (i + 1) % 200 == 0 or i + 1 == n_total:
                print(f"    {i + 1}/{n_total}")

        with open(index_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "mpid", "formula", "class", "A_true_zJ", "A_pred_zJ",
                "delta_A_zJ", "rel_error_pct", "plot_path",
            ])
            w.writerows(index_rows)
        print(f"  Index: {index_path}")

    if args.n_examples > 0 and not args.mpid:
        agg_dir = os.path.join(args.out_dir, f"{model_stem}_{args.split}_worst_examples")
        os.makedirs(agg_dir, exist_ok=True)
        worst = np.argsort(np.abs(final_gap))[::-1][: args.n_examples]
        for rank, i in enumerate(worst):
            safe = mpids[i].replace("/", "-")
            out = os.path.join(agg_dir, f"worst{rank+1}_{safe}.png")
            cls = "insulator" if insulator_mask[i] else "metal"
            plot_single(cum_true[i], cum_pred[i], xi_m, mpids[i], formulas[i],
                        out, n_plot, material_class=cls)
            print(f"  Worst #{rank + 1}: {mpids[i]}  ΔA = {final_gap[i]:.2f} zJ  →  {out}")

    print(f"\nDone. Plots in {args.out_dir}/")


if __name__ == "__main__":
    main()
