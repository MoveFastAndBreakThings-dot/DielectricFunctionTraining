"""
Compare base vs meta model: ε accuracy at Matsubara frequencies only.

Uses the same cubic-spline interpolation as Hamaker evaluation
(compare_hamaker_constants.get_matsubara_sequences):
  ε(iξ) on KKR grid → cubic interp1d → ε(iξ_n),  ξ_n = n · 2π k_B T

Usage:
    python3 compare_epsilon_base_vs_meta.py \\
        --base models/mace_kkr_v1.pt \\
        --meta models/mace_meta_hamaker_v1.pt
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

from compare_hamaker_constants import T_K, get_matsubara_sequences, get_split_indices
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward
from model_loader import load_model

OUT_DIR = "epsilon_base_vs_meta"


def load_ckpt_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model, model_type = load_model(ckpt, device)
    return ckpt, model, model_type


@torch.no_grad()
def predict_eps_linear(model, model_type, loader, device, use_log):
    pred_list, true_list = [], []
    for inputs, target, _ in loader:
        target = target.to(device)
        pred_lin = model_forward(model, model_type, inputs, device, use_log=use_log)
        pred_list.append(pred_lin.cpu().numpy())
        true_list.append(target[:, 0, :].cpu().numpy())
    return np.concatenate(pred_list), np.concatenate(true_list)


def eps_on_matsubara(xi, eps_curve, n_matsubara, temperature_k):
    """Cubic-spline ε(iξ_n) for n = 0 .. n_matsubara-1 (same as Hamaker code)."""
    d_n, _ = get_matsubara_sequences(
        xi, eps_curve,
        t_k=temperature_k,
        xi_step_ev=2 * np.pi * 8.617333e-5 * temperature_k,
        n_max=n_matsubara,
    )
    return d_n


def matsubara_log_errors(xi, pred, true, n_matsubara, temperature_k):
    """Per-crystal |log ε| error at each Matsubara point, shape (N, M)."""
    n = len(pred)
    errs = np.empty((n, n_matsubara), dtype=np.float64)
    for i in range(n):
        eps_t = eps_on_matsubara(xi, true[i], n_matsubara, temperature_k)
        eps_p = eps_on_matsubara(xi, pred[i], n_matsubara, temperature_k)
        errs[i] = np.abs(np.log(eps_p.clip(1e-30)) - np.log(eps_t.clip(1e-30)))
    return errs


def plot_matsubara_comparison(xi_m, err_b, err_m, delta, out_path):
    n = len(xi_m)
    x = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(x, err_b.mean(axis=0), "C1-", lw=2, label="base")
    ax.plot(x, err_m.mean(axis=0), "C0--", lw=2, label="meta")
    ax.set_ylabel("mean |log ε err|")
    ax.set_title("ε accuracy at Matsubara points (cubic spline from KKR grid)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    d = delta.mean(axis=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(x, 0, d, where=d >= 0, color="C3", alpha=0.4, label="meta worse")
    ax.fill_between(x, 0, d, where=d < 0, color="C2", alpha=0.4, label="meta better")
    ax.plot(x, d, "k-", lw=1.2)
    ax.set_ylabel("Δ mean |log ε err|")
    ax.set_title("meta − base  (negative = meta better)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(x, 100 * (delta < 0).mean(axis=0), "C2-", lw=2)
    ax.axhline(50, color="k", ls=":", lw=0.8)
    ax.set_ylabel("% crystals meta better")
    ax.set_xlabel("Matsubara index n")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.unique(np.linspace(0, n - 1, min(8, n)).astype(int))
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{xi_m[i]:.3f}" for i in ticks])
    ax2.set_xlabel("ξ_n (eV)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Base vs meta: ε error at Matsubara points (cubic spline)",
    )
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xi_step = 2 * np.pi * 8.617333e-5 * args.temperature_k
    xi_m = np.arange(args.n_matsubara, dtype=np.float64) * xi_step

    ckpt_b, model_b, model_type_b = load_ckpt_model(args.base, device)
    ckpt_m, model_m, model_type_m = load_ckpt_model(args.meta, device)
    use_log = ckpt_b.get("use_log", False)
    use_mace = ckpt_b.get("model_type", "CGCNN") == "MACE"
    target_components = tuple(ckpt_b.get("target_components", ("xx",)))

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt_b.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt_b.get("mace_r_max"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    split_idx = get_split_indices(ckpt_b, len(dataset), args.split)
    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )

    print("=" * 64)
    print(f"Base : {args.base}  ({ckpt_b.get('model_name', '')})  [{model_type_b}]")
    print(f"Meta : {args.meta}  ({ckpt_m.get('model_name', '')})  [{model_type_m}]")
    print(f"Split: {args.split}  n={len(split_idx)}  T={args.temperature_k} K")
    print(f"Matsubara: n = 0 .. {args.n_matsubara - 1}  (cubic spline, same as Hamaker)")
    print("=" * 64)

    pred_b, true = predict_eps_linear(model_b, model_type_b, loader, device, use_log)
    pred_m, _ = predict_eps_linear(model_m, model_type_m, loader, device, use_log)

    mpids = [dataset.get_sample_info(i)["mpid"] for i in split_idx]
    dataset.close()

    err_b = matsubara_log_errors(xi, pred_b, true, args.n_matsubara, args.temperature_k)
    err_m = matsubara_log_errors(xi, pred_m, true, args.n_matsubara, args.temperature_k)
    delta = err_m - err_b
    delta_per_crystal = delta.mean(axis=1)

    n_better = int((delta_per_crystal < 0).sum())
    n_worse = int((delta_per_crystal > 0).sum())

    print(f"\nMean |log ε error| at Matsubara points (cubic spline)")
    print(f"  base : {err_b.mean():.4f}")
    print(f"  meta : {err_m.mean():.4f}")
    print(f"  Δ    : {delta.mean():+.4f}  (negative ⇒ meta better)")
    print(f"  crystals: meta better {n_better}  worse {n_worse}")

    print(f"\nPer Matsubara index n  (mean |log ε err| over crystals)")
    print(f"  {'n':>4}  {'ξ_n (eV)':>10}  {'base':>10}  {'meta':>10}  {'Δ':>10}  {'% meta better':>14}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*14}")
    show_ns = list(range(min(15, args.n_matsubara)))
    if args.n_matsubara > 20:
        show_ns += list(range(args.n_matsubara - 5, args.n_matsubara))
    seen = set()
    for n in show_ns:
        if n in seen:
            continue
        seen.add(n)
        frac = 100 * (delta[:, n] < 0).mean()
        print(f"  {n:4d}  {xi_m[n]:10.4f}  {err_b[:, n].mean():10.4f}  "
              f"{err_m[:, n].mean():10.4f}  {delta[:, n].mean():+10.4f}  {frac:13.1f}%")

    d_n = delta.mean(axis=0)
    worst = np.argsort(d_n)[-5:][::-1]
    best = np.argsort(d_n)[:5]
    print(f"\nMatsubara n where meta hurt most")
    for n in worst:
        print(f"  n={n:3d}  ξ={xi_m[n]:.4f} eV  base={err_b[:, n].mean():.4f}  "
              f"meta={err_m[:, n].mean():.4f}  Δ={d_n[n]:+.4f}")
    print(f"Matsubara n where meta helped most")
    for n in best:
        print(f"  n={n:3d}  ξ={xi_m[n]:.4f} eV  base={err_b[:, n].mean():.4f}  "
              f"meta={err_m[:, n].mean():.4f}  Δ={d_n[n]:+.4f}")

    prefix = os.path.join(
        args.out_dir,
        f"{ckpt_b.get('model_name', 'base')}_vs_{ckpt_m.get('model_name', 'meta')}_{args.split}_matsubara",
    )
    plot_matsubara_comparison(xi_m, err_b, err_m, delta, f"{prefix}.png")

    csv_path = f"{prefix}_per_crystal.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mpid", "log_err_base", "log_err_meta", "delta_log_err"])
        for i, mpid in enumerate(mpids):
            w.writerow([mpid, err_b[i].mean(), err_m[i].mean(), delta_per_crystal[i]])

    print(f"\nSaved {prefix}.png")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
