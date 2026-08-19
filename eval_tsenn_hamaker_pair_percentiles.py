#!/usr/bin/env python3
"""
TSENN-KKR equivalent of eval_hamaker_pair_percentiles.py:

  1. Core task — ε_xx(iξ) MAE (log space if checkpoint was trained with --use_log)
  2. Downstream — pairwise Hamaker A_12 relative-error percentiles on the pair split

Usage:
  python eval_tsenn_hamaker_pair_percentiles.py \\
      --model_path models_latest_opt/tsenn_kkr_xx_latest_opt_best.torch \\
      --pairs_dir meta_hamaker_pairs_opt \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb

  python eval_tsenn_hamaker_pair_percentiles.py --model_path ... --skip_core
  python eval_tsenn_hamaker_pair_percentiles.py --model_path ... --skip_pairs --limit 64
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import torch_load_compat  # noqa: F401

from compare_tsenn_kkr_vs_ours import (
    TSENN_SPLIT_KEY,
    load_tsenn_kkr,
    predict_tsenn,
    read_lmdb_records,
)
from eval_hamaker_pair_percentiles import (
    PERCENTILES,
    predict_pair_hamaker_zj,
    print_percentile_table,
)
from hamaker_pair_dataset import load_pair_splits
from lmdb_dataset import LMDBCrystalDataset
from train_tsenn_kkr import DEFAULT_DTYPE, TSENN_ROOT

K_B_EV = 8.617333e-5


def core_lmdb_indices(ckpt, crystal_ds, split):
    """Map TSENN row-split → LMDB indices."""
    local = ckpt.get(TSENN_SPLIT_KEY[split])
    if local is None:
        raise ValueError(f"TSENN checkpoint missing {TSENN_SPLIT_KEY[split]}")
    lmdb_indices = ckpt.get("lmdb_indices")
    if lmdb_indices is not None:
        return [int(lmdb_indices[i]) for i in local]

    jids = ckpt.get("jids")
    if not jids:
        raise ValueError("TSENN checkpoint missing jids / lmdb_indices")
    jid_to_lmdb = {}
    for i in range(len(crystal_ds)):
        jid_to_lmdb[crystal_ds.get_sample_info(i)["jid"]] = i
    out = []
    missing = 0
    for li in local:
        jid = jids[li]
        if jid not in jid_to_lmdb:
            missing += 1
            continue
        out.append(jid_to_lmdb[jid])
    if missing:
        print(f"warning: {missing} TSENN {split} jids not in LMDB")
    if not out:
        raise RuntimeError(f"No LMDB indices resolved for TSENN {split} split")
    return out


def load_gt_eps_xx(crystal_ds, indices):
    """Return (N, F) GT ε_xx(iξ) from LMDBCrystalDataset."""
    out = []
    for i in indices:
        _, target, _ = crystal_ds[i]
        out.append(target[0].numpy())
    return np.stack(out, axis=0)


@torch.no_grad()
def predict_eps_for_indices(
    model, ckpt, lmdb_path, indices, device, batch_size, onehot_dir, refine
):
    """Predict ε_xx(iξ) for LMDB indices; returns dict idx -> (F,) and stacked array order."""
    unique = sorted(set(int(i) for i in indices))
    records = read_lmdb_records(lmdb_path, unique)
    preds = predict_tsenn(
        model, ckpt, records, device, batch_size, onehot_dir, refine=refine
    )
    return {idx: preds[j] for j, idx in enumerate(unique)}, unique, preds


def main():
    parser = argparse.ArgumentParser(
        description="TSENN-KKR: core ε(iξ) + pairwise Hamaker A_12 percentiles",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pairs_dir", type=str, default="meta_hamaker_pairs_opt")
    parser.add_argument(
        "--pair_split", type=str, default="test", choices=("train", "val", "test")
    )
    parser.add_argument(
        "--core_split",
        type=str,
        default="test",
        choices=("train", "val", "test"),
        help="TSENN crystal split for core ε(iξ) loss",
    )
    parser.add_argument("--lmdb_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_matsubara", type=int, default=None)
    parser.add_argument("--skip_core", action="store_true")
    parser.add_argument("--skip_pairs", action="store_true")
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap pairs / core crystals")
    args = parser.parse_args()

    meta_path = os.path.join(args.pairs_dir, "metadata.json")
    with open(meta_path) as f:
        pair_meta = json.load(f)

    lmdb_path = args.lmdb_path or pair_meta["lmdb_path"]
    n_matsubara = args.n_matsubara or pair_meta["n_matsubara"]
    temperature_k = pair_meta["temperature_k"]
    target_components = tuple(pair_meta.get("target_components", ("xx",)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_default_dtype(DEFAULT_DTYPE)
    print(f"device={device}")

    model, ckpt = load_tsenn_kkr(args.model_path, device)
    use_log = bool(ckpt.get("use_log", False))
    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    refine = not args.no_refine

    crystal_ds = LMDBCrystalDataset(
        lmdb_path,
        target_components=target_components,
        use_kkr=True,
        use_mace=False,
    )
    xi_np = np.asarray(crystal_ds.kkr_xi_array, dtype=np.float64)
    if int(ckpt["n_frequencies"]) != len(xi_np):
        raise ValueError(
            f"TSENN n_frequencies={ckpt['n_frequencies']} != LMDB ξ length={len(xi_np)}"
        )

    print("=" * 60)
    print(f"Model      : {args.model_path}")
    print(f"  type     : TSENN-KKR xx")
    print(f"  use_log  : {use_log}")
    print(f"  layers   : {ckpt.get('layers')}")
    print(f"LMDB       : {lmdb_path}  F={len(xi_np)}")
    print("=" * 60)

    needed = set()

    if not args.skip_core:
        core_idx = core_lmdb_indices(ckpt, crystal_ds, args.core_split)
        if args.limit is not None:
            core_idx = core_idx[: args.limit]
        needed.update(core_idx)
    else:
        core_idx = []

    pair_idx_a = pair_idx_b = a_true_all = None
    if not args.skip_pairs:
        splits = load_pair_splits(args.pairs_dir)
        pairs_npz = os.path.join(args.pairs_dir, "pairs.npz")
        data = np.load(pairs_npz)
        pair_indices = np.asarray(splits[args.pair_split], dtype=np.int64)
        if args.limit is not None:
            pair_indices = pair_indices[: args.limit]
        pair_idx_a = data["idx_a"][pair_indices].astype(np.int64)
        pair_idx_b = data["idx_b"][pair_indices].astype(np.int64)
        a_true_all = data["a_true_zj"][pair_indices].astype(np.float64)
        needed.update(pair_idx_a.tolist())
        needed.update(pair_idx_b.tolist())

    if not needed:
        crystal_ds.close()
        return

    print(f"\nPredicting TSENN ε_xx for {len(needed)} unique crystals …")
    eps_map, _, _ = predict_eps_for_indices(
        model,
        ckpt,
        lmdb_path,
        needed,
        device,
        args.batch_size,
        onehot_dir,
        refine,
    )

    if not args.skip_core:
        space = "log(ε)" if use_log else "ε"
        print(f"\n--- Core ε(iξ) loss (TSENN training objective) ---")
        print(f"  MAE on {space}  |  split={args.core_split}  n={len(core_idx)}")

        gt = load_gt_eps_xx(crystal_ds, core_idx)
        pred = np.stack([eps_map[i] for i in core_idx], axis=0)
        if use_log:
            pred_sp = np.log(np.clip(pred, 1e-12, None))
            gt_sp = np.log(np.clip(gt, 1e-12, None))
        else:
            pred_sp, gt_sp = pred, gt
        per_sample = np.mean(np.abs(pred_sp - gt_sp), axis=1)
        core_mean = float(per_sample.mean())
        print(f"\n  Test loss (mean MAE): {core_mean:.6f}")
        print_percentile_table(
            f"Per-crystal core loss ({space})",
            per_sample,
            PERCENTILES,
        )

    if not args.skip_pairs:
        print(f"\n--- Pairwise Hamaker A_12 (downstream) ---")
        print(f"  pairs: {args.pairs_dir}  split={args.pair_split}  n={len(a_true_all)}")
        print(f"  T={temperature_k} K  n_matsubara={n_matsubara}")

        xi_grid = torch.tensor(xi_np, dtype=torch.float64, device=device)
        xi_matsubara = (
            torch.arange(n_matsubara, dtype=torch.float64, device=device)
            * (2.0 * np.pi * K_B_EV * temperature_k)
        )

        a_pred_list = []
        bs = max(args.batch_size, 1)
        n_pairs = len(a_true_all)
        for start in tqdm(range(0, n_pairs, bs), desc="hamaker-pairs"):
            end = min(start + bs, n_pairs)
            ea = torch.tensor(
                np.stack([eps_map[int(i)] for i in pair_idx_a[start:end]], axis=0),
                dtype=torch.float64,
                device=device,
            )
            eb = torch.tensor(
                np.stack([eps_map[int(i)] for i in pair_idx_b[start:end]], axis=0),
                dtype=torch.float64,
                device=device,
            )
            a_pred = predict_pair_hamaker_zj(
                xi_grid, ea, eb, xi_matsubara, temperature_k
            )
            a_pred_list.append(a_pred.detach().cpu().numpy())

        a_pred = np.concatenate(a_pred_list)
        a_true = a_true_all
        rel_pct = 100.0 * (a_pred - a_true) / (a_true + 1e-30)
        abs_rel_pct = np.abs(rel_pct)

        mae_zj = float(np.mean(np.abs(a_pred - a_true)))
        rmse_zj = float(np.sqrt(np.mean((a_pred - a_true) ** 2)))
        r = float(np.corrcoef(a_pred, a_true)[0, 1]) if len(a_true) > 1 else float("nan")

        print(f"\n  MAE(ΔA)   = {mae_zj:.4f} zJ")
        print(f"  RMSE(ΔA)  = {rmse_zj:.4f} zJ")
        print(f"  Pearson r = {r:.4f}")

        print_percentile_table(
            "Signed relative error  (A_pred − A_true) / A_true × 100",
            rel_pct,
            PERCENTILES,
            unit="%",
        )
        print_percentile_table(
            "Absolute relative error  |A_pred − A_true| / |A_true| × 100",
            abs_rel_pct,
            PERCENTILES,
            unit="%",
        )

    crystal_ds.close()


if __name__ == "__main__":
    main()
