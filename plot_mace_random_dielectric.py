#!/usr/bin/env python3
"""
Plot GT vs official MACE ε_xx(iξ) for a random sample of crystals.

Usage:
  python plot_mace_random_dielectric.py
  python plot_mace_random_dielectric.py \\
      --mace models_latest_opt/mace_kkr_latest_opt.pt \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb \\
      --n 2000
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch_load_compat  # noqa: F401

from compare_hamaker_constants import get_split_indices
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward
from model_loader import load_model

COLOR_GT = "#111111"
COLOR_MACE = "#d62728"
OUT_DIR = "mace_random_dielectric"


def safe_name(jid, mpid, formula, idx):
    raw = f"{jid or mpid or 'noid'}_{formula or 'unk'}_{idx}"
    return re.sub(r"[^\w\-.]+", "_", raw)[:120]


def configure_mace_dataset(dataset, ckpt):
    if ckpt.get("mace_atomic_numbers") is None:
        return
    from mace.tools import AtomicNumberTable

    dataset.set_mace_graph_config(
        AtomicNumberTable(zs=[int(z) for z in ckpt["mace_atomic_numbers"]]),
        ckpt.get("mace_r_max", dataset.mace_r_max),
    )


@torch.no_grad()
def predict_eps(model, model_type, loader, device, use_log):
    preds, trues = [], []
    for inputs, target, _ in loader:
        target = target.to(device)
        pred = model_forward(model, model_type, inputs, device, use_log=use_log)
        preds.append(pred.cpu().numpy())
        trues.append(target[:, 0, :].cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def plot_one(xi, eps_gt, eps_pred, title, out_path):
    mae = float(np.mean(np.abs(eps_pred - eps_gt)))
    log_mae = float(np.mean(np.abs(np.log(eps_pred.clip(1e-30)) - np.log(eps_gt.clip(1e-30)))))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, log_x in zip(axes, (True, False)):
        ax.plot(xi, eps_gt, color=COLOR_GT, lw=2.0, label="GT")
        ax.plot(xi, eps_pred, color=COLOR_MACE, lw=1.6, ls="--", label="MACE")
        if log_x:
            ax.set_xscale("log")
            ax.set_title("log ξ")
        else:
            ax.set_title("linear ξ")
        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel(r"$\varepsilon_{xx}(i\xi)$")
        ax.set_ylim(bottom=0.9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{title}\nMAE={mae:.4f}   |log ε|={log_mae:.4f}", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mae, log_mae


def main():
    parser = argparse.ArgumentParser(description="Random official-MACE ε(iξ) plots vs GT")
    parser.add_argument("--mace", default="models_latest_opt/mace_kkr_latest_opt.pt")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--split", default="all", choices=("all", "train", "val", "test"))
    parser.add_argument("--n", type=int, default=2000, help="Number of random crystals to plot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")

    ckpt = torch.load(args.mace, map_location=device)
    model, model_type = load_model(ckpt, device)
    if model_type != "MACE":
        raise SystemExit(
            f"{args.mace} loaded as {model_type}, not official MACE. "
            "Pass an OfficialFrequencyMACE checkpoint (model_type=MACE)."
        )
    use_log = bool(ckpt.get("use_log", False))
    foundation = ckpt.get("foundation_model")

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=tuple(ckpt.get("target_components", ("xx",))),
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
        use_mace=True,
        mace_r_max=ckpt.get("mace_r_max"),
    )
    configure_mace_dataset(dataset, ckpt)
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    if args.split == "all":
        pool = np.arange(len(dataset))
    else:
        pool = np.asarray(get_split_indices(ckpt, len(dataset), args.split))

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(pool))
    split_idx = rng.choice(pool, size=n, replace=False).tolist()

    print("=" * 64)
    print(f"MACE : {args.mace}")
    print(f"       type={model_type}  use_log={use_log}  foundation={foundation}")
    print(
        f"       irreps={ckpt.get('mace_hidden_irreps')}  "
        f"layers={ckpt.get('mace_num_interactions')}  "
        f"r_max={ckpt.get('mace_r_max')}"
    )
    print(f"LMDB : {args.lmdb_path}  n_total={len(dataset)}")
    print(f"Plot : split={args.split}  pool={len(pool)}  n={n}  seed={args.seed}  → {out_dir}/")
    print("=" * 64)

    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=True),
    )
    pred, true = predict_eps(model, model_type, loader, device, use_log)
    infos = [dataset.get_sample_info(i) for i in split_idx]
    dataset.close()

    rows = []
    for i, info in enumerate(tqdm(infos, desc="plot")):
        name = safe_name(info.get("jid"), info.get("mpid"), info.get("formula"), split_idx[i])
        title = f"{info.get('jid', '')}  {info.get('formula', '')}  ({info.get('mpid', '')})"
        mae, log_mae = plot_one(xi, true[i], pred[i], title, out_dir / f"{name}.png")
        rows.append((name, info.get("jid", ""), info.get("formula", ""), mae, log_mae))

    rank_path = out_dir / "ranking_by_abs_log.csv"
    with open(rank_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "jid", "formula", "mae", "abs_log"])
        for row in sorted(rows, key=lambda r: r[4], reverse=True):
            w.writerow([row[0], row[1], row[2], f"{row[3]:.6f}", f"{row[4]:.6f}"])

    maes = np.array([r[3] for r in rows])
    logs = np.array([r[4] for r in rows])
    print(f"\nSaved {len(rows)} PNGs → {out_dir}/")
    print(f"Saved {rank_path}")
    print(f"Mean MAE={maes.mean():.4f}   mean |log ε|={logs.mean():.4f}")


if __name__ == "__main__":
    main()
