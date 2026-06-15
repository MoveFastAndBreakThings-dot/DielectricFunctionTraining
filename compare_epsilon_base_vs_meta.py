"""
Compare ε(iξ) prediction quality: base model vs meta-trained model.

Reports where meta training helped or hurt relative to the backbone:
  - per-ξ frequency (log & linear error)
  - low / mid / high ξ bands
  - Matsubara-relevant ξ_n
  - metals vs insulators, band-gap bins
  - per-material winners & losers (+ single-material Hamaker shift)

Usage:
    python3 compare_epsilon_base_vs_meta.py \\
        --base models/mace_kkr_v1.pt \\
        --meta models/mace_meta_hamaker_v1.pt

Outputs: epsilon_base_vs_meta/
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

from compare_hamaker_constants import T_K, get_split_indices, hamaker_constant_zj
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model

OUT_DIR = "epsilon_base_vs_meta"
K_B_EV = 8.617333e-5
XI_STEP_EV = 2 * np.pi * K_B_EV * T_K


def load_ckpt_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model, _ = load_model(ckpt, device)
    return ckpt, model


@torch.no_grad()
def predict_eps(model, loader, device, use_log):
    """Return pred_lin (N,F), true_lin (N,F), log_abs_err (N,F), lin_abs_err (N,F)."""
    pred_list, true_list, log_err_list = [], [], []
    for inputs, target, _ in loader:
        af, nf, nfi, cai = inputs
        af = af.to(device)
        nf = nf.to(device)
        nfi = nfi.to(device)
        cai = [x.to(device) for x in cai]
        target = target.to(device)
        out = model(af, nf, nfi, cai)[:, 0, :]
        true_ch = target[:, 0, :]
        if use_log:
            pred_lin = torch.exp(out)
            log_err = torch.abs(out - torch.log(true_ch))
        else:
            pred_lin = out
            log_err = torch.abs(
                torch.log(pred_lin.clamp_min(1e-12)) - torch.log(true_ch.clamp_min(1e-12))
            )
        pred_list.append(pred_lin.cpu().numpy())
        true_list.append(true_ch.cpu().numpy())
        log_err_list.append(log_err.cpu().numpy())
    pred = np.concatenate(pred_list)
    true = np.concatenate(true_list)
    log_err = np.concatenate(log_err_list)
    lin_abs_err = np.abs(pred - true)
    return pred, true, log_err, lin_abs_err


def collect_metadata(dataset, indices):
    mpids, formulas, gaps, eps_low = [], [], [], []
    for i in indices:
        info = dataset.get_sample_info(i)
        mpids.append(info["mpid"])
        formulas.append(info["formula"])
        gaps.append(float(info.get("opt_gap") or np.nan))
        _, tgt, _ = dataset[i]
        eps_low.append(float(tgt[0, 0]))
    return {
        "mpid": np.array(mpids),
        "formula": np.array(formulas),
        "opt_gap": np.array(gaps),
        "eps_low": np.array(eps_low),
    }


def material_masks(meta, metal_eps_threshold=50.0):
    metal = meta["eps_low"] > metal_eps_threshold
    insulator = ~metal
    gap = meta["opt_gap"]
    gap_bins = {
        "gap_unknown": ~np.isfinite(gap),
        "gap_metal (<0.1 eV)": np.isfinite(gap) & (gap < 0.1),
        "gap_small (0.1–1 eV)": np.isfinite(gap) & (gap >= 0.1) & (gap < 1.0),
        "gap_mid (1–3 eV)": np.isfinite(gap) & (gap >= 1.0) & (gap < 3.0),
        "gap_large (≥3 eV)": np.isfinite(gap) & (gap >= 3.0),
    }
    return {"metal": metal, "insulator": insulator, **gap_bins}


def xi_band_masks(xi, n_bands=3):
    """Split ξ grid into low / mid / high thirds by index (log-spaced grid)."""
    f = len(xi)
    edges = np.linspace(0, f, n_bands + 1, dtype=int)
    bands = {}
    names = ["low_xi", "mid_xi", "high_xi"]
    for k in range(n_bands):
        m = np.zeros(f, dtype=bool)
        m[edges[k]:edges[k + 1]] = True
        xi_lo, xi_hi = xi[edges[k]], xi[min(edges[k + 1] - 1, f - 1)]
        bands[names[k]] = (m, f"{names[k]}  ξ∈[{xi_lo:.3g}, {xi_hi:.3g}] eV")
    return bands


def mean_over(mask_samples, arr_2d, mask_freq=None):
    """Mean over selected crystals and optional frequency mask."""
    x = arr_2d[mask_samples]
    if mask_freq is not None:
        x = x[:, mask_freq]
    return float(x.mean())


def print_group_table(title, groups, err_base, err_meta, delta):
    print(f"\n{title}")
    print(f"  {'group':<28} {'n':>6} {'base':>10} {'meta':>10} {'Δ(meta-base)':>14} {'meta better':>12}")
    print(f"  {'-'*28} {'-'*6} {'-'*10} {'-'*10} {'-'*14} {'-'*12}")
    for label, mask in groups.items():
        if mask.sum() == 0:
            continue
        b = mean_over(mask, err_base)
        m = mean_over(mask, err_meta)
        d = m - b
        frac = float((delta[mask].mean(axis=1) < 0).mean()) * 100
        print(f"  {label:<28} {int(mask.sum()):>6} {b:10.4f} {m:10.4f} {d:+14.4f} {frac:11.1f}%")


def plot_freq_curves(xi, err_b, err_m, delta, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = np.arange(len(xi))

    ax = axes[0]
    ax.plot(x, err_b.mean(axis=0), color="C1", lw=2, label="base")
    ax.plot(x, err_m.mean(axis=0), color="C0", lw=2, ls="--", label="meta")
    ax.set_ylabel("|log ε err| mean")
    ax.set_title("Mean per-ξ error (log space)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    d_mean = delta.mean(axis=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(x, 0, d_mean, where=d_mean >= 0, color="C3", alpha=0.4, label="meta worse")
    ax.fill_between(x, 0, d_mean, where=d_mean < 0, color="C2", alpha=0.4, label="meta better")
    ax.plot(x, d_mean, "k-", lw=1)
    ax.set_ylabel("Δ mean |log ε err|")
    ax.set_title("meta − base  (negative = meta improved ε)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    frac_better = (delta < 0).mean(axis=0)
    ax.plot(x, 100 * frac_better, color="C2", lw=2)
    ax.axhline(50, color="k", ls=":", lw=0.8)
    ax.set_ylabel("% crystals meta better")
    ax.set_xlabel("ξ index")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax2 = axes[2].twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.unique(np.linspace(0, len(xi) - 1, min(8, len(xi))).astype(int))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{xi[i]:.2f}" for i in ticks])
    ax2.set_xlabel("ξ (eV)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_matsubara_overlay(xi, err_b, err_m, n_matsubara, out_path):
    xi_m = np.arange(n_matsubara) * XI_STEP_EV
    idx = np.searchsorted(xi, xi_m, side="right").clip(1, len(xi) - 1)
    xi_near = xi[idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xi_m, err_b.mean(axis=0)[idx], "o-", color="C1", ms=3, label="base @ nearest ξ")
    ax.plot(xi_m, err_m.mean(axis=0)[idx], "s--", color="C0", ms=3, label="meta @ nearest ξ")
    ax.set_xlabel("Matsubara ξ_n (eV)")
    ax.set_ylabel("mean |log ε err|")
    ax.set_title("Error at Matsubara frequencies (n = 0 … {})".format(n_matsubara - 1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_crystal_hist(delta_per_crystal, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delta_per_crystal, bins=60, color="C0", alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.axvline(np.median(delta_per_crystal), color="C3", ls="-", lw=1.5, label=f"median {np.median(delta_per_crystal):+.3f}")
    ax.set_xlabel("Δ per-crystal mean |log ε err|  (meta − base)")
    ax.set_ylabel("Count")
    ax.set_title(f"meta better: {(delta_per_crystal < 0).mean()*100:.1f}% of crystals")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ε(iξ) comparison: base vs meta-trained model")
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0)
    parser.add_argument("--n_matsubara", type=int, default=150)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_b, model_b = load_ckpt_model(args.base, device)
    ckpt_m, model_m = load_ckpt_model(args.meta, device)
    use_log = ckpt_b.get("use_log", False)
    target_components = tuple(ckpt_b.get("target_components", ("xx",)))

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt_b.get("metal_gap_threshold"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    split_idx = get_split_indices(ckpt_b, len(dataset), args.split)
    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pool,
    )

    print("=" * 64)
    print(f"Base : {args.base}  ({ckpt_b.get('model_name', '')})")
    print(f"Meta : {args.meta}  ({ckpt_m.get('model_name', '')})")
    print(f"Split: {args.split}  n={len(split_idx)}  use_log={use_log}")
    print("=" * 64)

    pred_b, true, log_err_b, lin_err_b = predict_eps(model_b, loader, device, use_log)
    pred_m, _, log_err_m, lin_err_m = predict_eps(model_m, loader, device, use_log)

    meta = collect_metadata(dataset, split_idx)
    dataset.close()

    delta_log = log_err_m - log_err_b
    delta_lin = lin_err_m - lin_err_b
    delta_per_crystal = delta_log.mean(axis=1)

    n_improved = int((delta_per_crystal < 0).sum())
    n_worse = int((delta_per_crystal > 0).sum())
    n_same = len(delta_per_crystal) - n_improved - n_worse

    print(f"\nOverall (mean |log ε error|)")
    print(f"  base : {log_err_b.mean():.4f}")
    print(f"  meta : {log_err_m.mean():.4f}")
    print(f"  Δ    : {delta_log.mean():+.4f}  (negative ⇒ meta better on ε)")
    print(f"  crystals: meta better {n_improved}  worse {n_worse}  tie~{n_same}")

    print(f"\nOverall (mean |ε_pred − ε_true| linear)")
    print(f"  base : {lin_err_b.mean():.4f}")
    print(f"  meta : {lin_err_m.mean():.4f}")
    print(f"  Δ    : {delta_lin.mean():+.4f}")

    groups = material_masks(meta, args.metal_eps_threshold)
    print_group_table("By material class (mean |log ε err|)", groups, log_err_b, log_err_m, delta_log)

    xi_bands = xi_band_masks(xi)
    print(f"\nBy ξ band (mean |log ε err| over ξ points in band)")
    print(f"  {'band':<32} {'base':>10} {'meta':>10} {'Δ':>12} {'% cr. meta better':>18}")
    print(f"  {'-'*32} {'-'*10} {'-'*10} {'-'*12} {'-'*18}")
    for key, (freq_mask, desc) in xi_bands.items():
        b = log_err_b[:, freq_mask].mean()
        m = log_err_m[:, freq_mask].mean()
        d = m - b
        frac = (delta_log[:, freq_mask].mean(axis=1) < 0).mean() * 100
        print(f"  {desc:<32} {b:10.4f} {m:10.4f} {d:+12.4f} {frac:17.1f}%")

    # Matsubara slice
    xi_m = np.arange(args.n_matsubara) * XI_STEP_EV
    m_idx = np.searchsorted(xi, xi_m, side="right").clip(1, len(xi) - 1)
    mb = log_err_b[:, m_idx].mean()
    mm = log_err_m[:, m_idx].mean()
    print(f"\nMatsubara-relevant ξ (n=0..{args.n_matsubara-1}, nearest grid point)")
    print(f"  base {mb:.4f}  meta {mm:.4f}  Δ {mm-mb:+.4f}")

    # Single-material Hamaker
    ham_kw = dict(t_k=T_K, n_max=500)
    a_true = np.array([hamaker_constant_zj(xi, true[i], **ham_kw) for i in range(len(true))])
    a_base = np.array([hamaker_constant_zj(xi, pred_b[i], **ham_kw) for i in range(len(pred_b))])
    a_meta = np.array([hamaker_constant_zj(xi, pred_m[i], **ham_kw) for i in range(len(pred_m))])
    h_err_b = np.abs(a_base - a_true)
    h_err_m = np.abs(a_meta - a_true)
    delta_h = h_err_m - h_err_b
    print(f"\nSingle-material Hamaker |ΔA| (zJ)")
    print(f"  base MAE : {h_err_b.mean():.2f}")
    print(f"  meta MAE : {h_err_m.mean():.2f}")
    print(f"  Δ MAE    : {delta_h.mean():+.2f}  (negative ⇒ meta better)")
    print(f"  crystals with better Hamaker: {(delta_h < 0).mean()*100:.1f}%")

    # Top-K materials
    order_worse = np.argsort(delta_per_crystal)[::-1]
    order_better = np.argsort(delta_per_crystal)

    print(f"\nTop {args.top_k} crystals where meta WORSENED ε (largest Δ log err)")
    print(f"  {'mpid':<14} {'formula':<20} {'class':<10} {'gap':>6} {'Δlog':>8} {'Δ|A| zJ':>10}")
    for i in order_worse[-args.top_k:][::-1]:
        cls = "metal" if groups["metal"][i] else "insulator"
        g = meta["opt_gap"][i]
        print(f"  {meta['mpid'][i]:<14} {str(meta['formula'][i])[:20]:<20} {cls:<10} "
              f"{g if np.isfinite(g) else float('nan'):>6.2f} {delta_per_crystal[i]:+8.3f} {delta_h[i]:+10.2f}")

    print(f"\nTop {args.top_k} crystals where meta IMPROVED ε (most negative Δ)")
    print(f"  {'mpid':<14} {'formula':<20} {'class':<10} {'gap':>6} {'Δlog':>8} {'Δ|A| zJ':>10}")
    for i in order_better[:args.top_k]:
        cls = "metal" if groups["metal"][i] else "insulator"
        g = meta["opt_gap"][i]
        print(f"  {meta['mpid'][i]:<14} {str(meta['formula'][i])[:20]:<20} {cls:<10} "
              f"{g if np.isfinite(g) else float('nan'):>6.2f} {delta_per_crystal[i]:+8.3f} {delta_h[i]:+10.2f}")

    # Worst ξ indices for meta regression
    d_freq = delta_log.mean(axis=0)
    worst_xi = np.argsort(d_freq)[-10:][::-1]
    best_xi = np.argsort(d_freq)[:10]
    print(f"\nξ points where meta hurt ε most (largest Δ mean |log err|)")
    for j in worst_xi:
        print(f"  idx={j:4d}  ξ={xi[j]:.4f} eV  base={log_err_b[:,j].mean():.4f}  "
              f"meta={log_err_m[:,j].mean():.4f}  Δ={d_freq[j]:+.4f}")
    print(f"\nξ points where meta helped ε most")
    for j in best_xi:
        print(f"  idx={j:4d}  ξ={xi[j]:.4f} eV  base={log_err_b[:,j].mean():.4f}  "
              f"meta={log_err_m[:,j].mean():.4f}  Δ={d_freq[j]:+.4f}")

    prefix = os.path.join(args.out_dir, f"{ckpt_b.get('model_name','base')}_vs_{ckpt_m.get('model_name','meta')}_{args.split}")
    plot_freq_curves(xi, log_err_b, log_err_m, delta_log, f"{prefix}_freq_log_err.png")
    plot_matsubara_overlay(xi, log_err_b, log_err_m, args.n_matsubara, f"{prefix}_matsubara_err.png")
    plot_per_crystal_hist(delta_per_crystal, f"{prefix}_per_crystal_delta.png")

    csv_path = f"{prefix}_per_crystal.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mpid", "formula", "class", "opt_gap_eV", "eps_lowest_xi",
            "log_err_base", "log_err_meta", "delta_log_err",
            "hamaker_abs_err_base_zJ", "hamaker_abs_err_meta_zJ", "delta_hamaker_abs_err_zJ",
        ])
        for i in range(len(split_idx)):
            cls = "metal" if groups["metal"][i] else "insulator"
            w.writerow([
                meta["mpid"][i], meta["formula"][i], cls, meta["opt_gap"][i], meta["eps_low"][i],
                log_err_b[i].mean(), log_err_m[i].mean(), delta_per_crystal[i],
                h_err_b[i], h_err_m[i], delta_h[i],
            ])

    print(f"\nSaved plots + {csv_path}")


if __name__ == "__main__":
    main()
