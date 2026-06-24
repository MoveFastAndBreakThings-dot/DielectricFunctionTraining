"""
Plot ε(iξ) vs ξ for base vs meta models — one PNG per material on the test set.

Ranking is by per-crystal MAE on the KKR grid (meta − base vs DFT truth).
  negative ΔMAE → meta better   |   positive ΔMAE → meta worse

Outputs under --out_dir (default: epsilon_curves_base_vs_meta/):
  all/{mpid}.png             — every crystal in the split
  best_20/rank01_{mpid}.png  — 20 materials meta helped most
  worst_20/rank01_{mpid}.png — 20 materials meta hurt most
  ranking.csv                — full split ranking + plot paths

Usage:
    python3 plot_epsilon_base_vs_meta_curves.py \\
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

from compare_hamaker_constants import get_split_indices
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model

OUT_DIR = "epsilon_curves_base_vs_meta"
TOP_K = 20


def safe_name(s, max_len=80):
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))
    return out[:max_len] if len(out) > max_len else out


def load_ckpt_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model, _ = load_model(ckpt, device)
    return ckpt, model


@torch.no_grad()
def predict_eps(model, loader, device, use_log):
    pred_list, true_list = [], []
    for inputs, target, _ in loader:
        af, nf, nfi, cai = inputs
        af = af.to(device)
        nf = nf.to(device)
        nfi = nfi.to(device)
        cai = [x.to(device) for x in cai]
        target = target.to(device)
        out = model(af, nf, nfi, cai)[:, 0, :]
        true_ch = target[:, 0, :]
        pred_lin = torch.exp(out) if use_log else out
        pred_list.append(pred_lin.cpu().numpy())
        true_list.append(true_ch.cpu().numpy())
    return np.concatenate(pred_list), np.concatenate(true_list)


def per_crystal_mae(pred, true):
    return np.mean(np.abs(pred - true), axis=1)


def plot_single_material(xi, eps_true, eps_base, eps_meta, mpid, formula, mae_b, mae_m, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    delta = mae_m - mae_b
    for ax, log_x in zip(axes, [True, False]):
        ax.plot(xi, eps_true, color="C2", lw=2, ls="-", label="DFT")
        ax.plot(xi, eps_base, color="C1", lw=1.8, ls="--", label=f"Base  MAE={mae_b:.4f}")
        ax.plot(xi, eps_meta, color="C0", lw=1.8, ls="-.", label=f"Meta  MAE={mae_m:.4f}")
        if log_x:
            ax.set_xscale("log")
            ax.set_title("Log ξ")
        else:
            ax.set_title("Linear ξ")
        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel("ε(iξ)")
        ax.set_ylim(bottom=0.9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{mpid}  {formula}  |  ΔMAE(meta−base) = {delta:+.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_individual_plots(xi, indices, true, pred_b, pred_m, mpids, formulas, mae_b, mae_m,
                           out_dir, filename_fn):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    n = len(indices)
    for j, i in enumerate(indices):
        rel = filename_fn(j, i)
        out_path = os.path.join(out_dir, rel)
        plot_single_material(
            xi, true[i], pred_b[i], pred_m[i],
            mpids[i], formulas[i], mae_b[i], mae_m[i], out_path,
        )
        paths.append((i, rel))
        if (j + 1) % 50 == 0 or j + 1 == n:
            print(f"    {j + 1}/{n} → {out_dir}/")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Individual ε(iξ) plots: all test materials + best/worst 20",
    )
    parser.add_argument("--base", type=str, required=True, help="Base model checkpoint")
    parser.add_argument("--meta", type=str, required=True, help="Meta-trained checkpoint")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of best/worst materials")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_dir = os.path.join(args.out_dir, "all")
    best_dir = os.path.join(args.out_dir, "best_20")
    worst_dir = os.path.join(args.out_dir, "worst_20")

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

    name_b = ckpt_b.get("model_name", os.path.basename(args.base))
    name_m = ckpt_m.get("model_name", os.path.basename(args.meta))
    print("=" * 64)
    print(f"Base : {args.base}  ({name_b})")
    print(f"Meta : {args.meta}  ({name_m})")
    print(f"Split: {args.split}  n={len(split_idx)}  use_log={use_log}")
    print("=" * 64)

    pred_b, true = predict_eps(model_b, loader, device, use_log)
    pred_m, _ = predict_eps(model_m, loader, device, use_log)

    mpids = [dataset.get_sample_info(i)["mpid"] for i in split_idx]
    formulas = [dataset.get_sample_info(i)["formula"] for i in split_idx]
    dataset.close()

    mae_b = per_crystal_mae(pred_b, true)
    mae_m = per_crystal_mae(pred_m, true)
    delta_mae = mae_m - mae_b

    order_best = np.argsort(delta_mae)
    k = min(args.top_k, len(split_idx))
    best_idx = order_best[:k]
    worst_idx = order_best[::-1][:k]
    all_indices = np.arange(len(split_idx))

    print(f"\n(i) All materials — {len(all_indices)} individual plots")
    all_paths = write_individual_plots(
        xi, all_indices, true, pred_b, pred_m, mpids, formulas, mae_b, mae_m,
        all_dir,
        lambda _j, i: f"{safe_name(mpids[i])}_{safe_name(formulas[i])}.png",
    )

    print(f"\n(ii) Worst {k} — meta worse than base")
    worst_paths = write_individual_plots(
        xi, worst_idx, true, pred_b, pred_m, mpids, formulas, mae_b, mae_m,
        worst_dir,
        lambda rank, i: f"rank{rank + 1:02d}_{safe_name(mpids[i])}.png",
    )

    print(f"\n(iii) Best {k} — meta better than base")
    best_paths = write_individual_plots(
        xi, best_idx, true, pred_b, pred_m, mpids, formulas, mae_b, mae_m,
        best_dir,
        lambda rank, i: f"rank{rank + 1:02d}_{safe_name(mpids[i])}.png",
    )

    all_plot = {i: os.path.join("all", rel) for i, rel in all_paths}
    best_plot = {i: os.path.join("best_20", rel) for i, rel in best_paths}
    worst_plot = {i: os.path.join("worst_20", rel) for i, rel in worst_paths}

    csv_path = os.path.join(args.out_dir, "ranking.csv")
    rows = sorted(range(len(split_idx)), key=lambda i: delta_mae[i])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank_meta_vs_base", "mpid", "formula",
            "mae_base", "mae_meta", "delta_mae_meta_minus_base",
            "plot_all", "plot_best_20", "plot_worst_20",
        ])
        for rank, i in enumerate(rows, start=1):
            w.writerow([
                rank, mpids[i], formulas[i],
                f"{mae_b[i]:.6e}", f"{mae_m[i]:.6e}", f"{delta_mae[i]:.6e}",
                all_plot[i],
                best_plot.get(i, ""),
                worst_plot.get(i, ""),
            ])
    print(f"\nSaved {csv_path}")

    print(f"\nSummary (MAE on ε(iξ), {args.split} split):")
    print(f"  mean MAE base : {mae_b.mean():.4f}")
    print(f"  mean MAE meta : {mae_m.mean():.4f}")
    print(f"  mean ΔMAE     : {delta_mae.mean():+.4f}  (negative ⇒ meta better)")
    print(f"  meta better   : {(delta_mae < 0).sum()} / {len(delta_mae)}")


if __name__ == "__main__":
    main()
