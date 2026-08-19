#!/usr/bin/env python3
"""
Plot GT vs TSENN-KKR vs CGCNN/MACE ε_xx(iξ) on the shared OPT/MBJ test set.

Assumes test sets match (TSENN trained with --align_split_ckpt). Default uses
the CGCNN/MACE checkpoint's test_idx. Use --split_mode intersection if unsure.

Each PNG: log-ξ and linear-ξ panels with GT / TSENN / Ours overlaid.

Usage:
  python plot_tsenn_cgcnn_kkr_dielectric.py \\
      --tsenn models_latest_opt/tsenn_kkr_xx_latest_opt_best.torch \\
      --ours  models_latest_opt/cgcnn_kkr_latest_opt.pt \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb \\
      --limit 24
"""

from __future__ import annotations

import argparse
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
from compare_tsenn_kkr_vs_ours import (
    load_tsenn_kkr,
    predict_ours,
    predict_tsenn,
    read_lmdb_records,
    resolve_eval_indices,
)
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_loader import load_model
from train_tsenn_kkr import DEFAULT_DTYPE, TSENN_ROOT

COLOR_GT = "#111111"
COLOR_TSENN = "#1f77b4"
COLOR_OURS = "#d62728"
OUT_DIR = "tsenn_cgcnn_kkr_dielectric"


def safe_name(jid, mpid, formula, idx):
    raw = f"{jid or mpid or 'noid'}_{formula or 'unk'}_{idx}"
    return re.sub(r"[^\w\-.]+", "_", raw)[:120]


def plot_one(xi, eps_gt, eps_t, eps_o, title, out_path):
    mae_t = float(np.mean(np.abs(eps_t - eps_gt)))
    mae_o = float(np.mean(np.abs(eps_o - eps_gt)))
    log_t = float(np.mean(np.abs(np.log(eps_t.clip(1e-30)) - np.log(eps_gt.clip(1e-30)))))
    log_o = float(np.mean(np.abs(np.log(eps_o.clip(1e-30)) - np.log(eps_gt.clip(1e-30)))))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, log_x in zip(axes, (True, False)):
        ax.plot(xi, eps_gt, color=COLOR_GT, lw=2.0, label="GT")
        ax.plot(xi, eps_t, color=COLOR_TSENN, lw=1.6, label="TSENN")
        ax.plot(xi, eps_o, color=COLOR_OURS, lw=1.6, ls="--", label="CGCNN")
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

    fig.suptitle(
        f"{title}\n"
        f"MAE  TSENN={mae_t:.4f}  CGCNN={mae_o:.4f}   |   "
        f"|log ε|  TSENN={log_t:.4f}  CGCNN={log_o:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mae_t, mae_o, log_t, log_o


def main():
    parser = argparse.ArgumentParser(
        description="Plot GT / TSENN-KKR / CGCNN ε(iξ) on shared test set",
    )
    parser.add_argument("--tsenn", required=True)
    parser.add_argument("--ours", required=True, help="CGCNN or MACE .pt")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--split_mode",
        default="ours",
        choices=("ours", "intersection", "tsenn"),
        help="Default 'ours': assume aligned test sets (use CGCNN test_idx). "
        "Use 'intersection' if splits may differ.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_default_dtype(torch.float32)
    print(f"device={device}")

    ckpt_t_meta = torch.load(args.tsenn, map_location="cpu")
    ckpt_o = torch.load(args.ours, map_location=device)
    model_o, model_type_o = load_model(ckpt_o, device)
    use_log_o = bool(ckpt_o.get("use_log", False))
    use_mace = model_type_o == "MACE"
    target_components = tuple(ckpt_o.get("target_components", ("xx",)))

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt_o.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt_o.get("mace_r_max"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    if int(ckpt_t_meta["n_frequencies"]) != len(xi):
        raise ValueError(
            f"TSENN n_frequencies={ckpt_t_meta['n_frequencies']} != LMDB F={len(xi)}"
        )

    if args.split_mode == "ours":
        split_idx = list(get_split_indices(ckpt_o, len(dataset), args.split))
        print(
            f"Using {args.split} from ours (aligned-split assumption): n={len(split_idx)}"
        )
    else:
        split_idx, ours_idx, tsenn_idx = resolve_eval_indices(
            ckpt_o, ckpt_t_meta, dataset, args.split, args.split_mode
        )
        print(
            f"split_mode={args.split_mode}  eval_n={len(split_idx)}  "
            f"(ours={len(ours_idx)}, tsenn={len(tsenn_idx)})"
        )

    if args.limit is not None:
        split_idx = split_idx[: args.limit]

    print("=" * 64)
    print(f"TSENN : {args.tsenn}")
    print(f"Ours  : {args.ours}  [{model_type_o}]")
    print(f"Plot  : n={len(split_idx)}  → {out_dir}/")
    print("=" * 64)

    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )
    pred_o, true = predict_ours(model_o, model_type_o, loader, device, use_log_o)
    del model_o
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = read_lmdb_records(args.lmdb_path, split_idx)
    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    torch.set_default_dtype(DEFAULT_DTYPE)
    model_t, ckpt_t = load_tsenn_kkr(args.tsenn, device)
    pred_t = predict_tsenn(
        model_t,
        ckpt_t,
        records,
        device,
        args.batch_size,
        onehot_dir,
        refine=not args.no_refine,
    )
    torch.set_default_dtype(torch.float32)

    infos = [dataset.get_sample_info(i) for i in split_idx]
    dataset.close()

    rows = []
    for i, info in enumerate(tqdm(infos, desc="plot")):
        name = safe_name(info.get("jid"), info.get("mpid"), info.get("formula"), split_idx[i])
        out_path = out_dir / f"{name}.png"
        title = f"{info.get('jid', '')}  {info.get('formula', '')}  ({info.get('mpid', '')})"
        mae_t, mae_o, log_t, log_o = plot_one(
            xi, true[i], pred_t[i], pred_o[i], title, out_path
        )
        rows.append((name, mae_t, mae_o, log_t, log_o))

    rank_path = out_dir / "ranking_by_abs_log.csv"
    with open(rank_path, "w") as f:
        f.write("name,mae_tsenn,mae_cgcnn,abs_log_tsenn,abs_log_cgcnn\n")
        for name, mae_t, mae_o, log_t, log_o in rows:
            f.write(f"{name},{mae_t:.6f},{mae_o:.6f},{log_t:.6f},{log_o:.6f}\n")

    print(f"\nSaved {len(rows)} PNGs → {out_dir}/")
    print(f"Saved {rank_path}")
    print(
        f"Mean |log ε|: TSENN={np.mean([r[3] for r in rows]):.4f}  "
        f"CGCNN={np.mean([r[4] for r in rows]):.4f}"
    )


if __name__ == "__main__":
    main()
