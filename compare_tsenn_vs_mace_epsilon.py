#!/usr/bin/env python3
"""
TSENN vs MACE on OPT: same ε(iξ) and self-Hamaker A errors as
compare_hamaker_constants.py / compare_tsenn_kkr_vs_ours.py.

Usage:
  python compare_tsenn_vs_mace_epsilon.py
  python compare_tsenn_vs_mace_epsilon.py \\
      --tsenn models_latest_opt/tsenn_kkr_xx_latest_opt_best.torch \\
      --mace  models_latest_opt/mace_kkr_latest_opt.pt
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import torch_load_compat  # noqa: F401

from compare_hamaker_constants import (
    K_B_EV,
    T_K,
    hamaker_constant_zj,
    report_hamaker_group,
)
from compare_tsenn_kkr_vs_ours import (
    load_tsenn_kkr,
    predict_ours,
    predict_tsenn,
    read_lmdb_records,
    resolve_eval_indices,
    summarize_eps,
    summarize_hamaker,
)
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_loader import load_model
from train_tsenn_kkr import DEFAULT_DTYPE, TSENN_ROOT


def configure_mace_dataset(dataset, ckpt):
    if ckpt.get("mace_atomic_numbers") is None:
        return
    from mace.tools import AtomicNumberTable

    dataset.set_mace_graph_config(
        AtomicNumberTable(zs=[int(z) for z in ckpt["mace_atomic_numbers"]]),
        ckpt.get("mace_r_max", dataset.mace_r_max),
    )


def main():
    parser = argparse.ArgumentParser(description="TSENN vs MACE: ε and Hamaker A on OPT")
    parser.add_argument("--tsenn", default="models_latest_opt/tsenn_kkr_xx_latest_opt_best.torch")
    parser.add_argument("--mace", default="models_latest_opt/mace_kkr_latest_opt.pt")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--split_mode", default="intersection", choices=("intersection", "ours", "tsenn"),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_csv", default="tsenn_vs_mace_per_crystal.csv")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_default_dtype(torch.float32)
    print(f"device={device}")

    ckpt_m = torch.load(args.mace, map_location=device)
    model_m, model_type = load_model(ckpt_m, device)
    use_log_m = bool(ckpt_m.get("use_log", False))
    use_mace = model_type == "MACE"
    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=tuple(ckpt_m.get("target_components", ("xx",))),
        use_kkr=True,
        metal_gap_threshold=ckpt_m.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt_m.get("mace_r_max"),
    )
    if use_mace:
        configure_mace_dataset(dataset, ckpt_m)
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    ckpt_t_meta = torch.load(args.tsenn, map_location="cpu")
    split_idx, mace_split, tsenn_split = resolve_eval_indices(
        ckpt_m, ckpt_t_meta, dataset, args.split, args.split_mode,
    )
    if args.limit is not None:
        split_idx = split_idx[: args.limit]

    print("=" * 64)
    print(f"TSENN : {args.tsenn}")
    print(f"MACE  : {args.mace}  [{model_type}]  use_log={use_log_m}")
    print(f"LMDB  : {args.lmdb_path}")
    print(
        f"Split : {args.split}  mode={args.split_mode}  n={len(split_idx)}  "
        f"(mace={len(mace_split)}, tsenn={len(tsenn_split)})"
    )
    print("=" * 64)

    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )
    pred_m, true = predict_ours(model_m, model_type, loader, device, use_log_m)
    del model_m
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = read_lmdb_records(args.lmdb_path, split_idx)
    torch.set_default_dtype(DEFAULT_DTYPE)
    model_t, ckpt_t = load_tsenn_kkr(args.tsenn, device)
    pred_t = predict_tsenn(
        model_t, ckpt_t, records, device, args.batch_size,
        args.onehot_dir or str(TSENN_ROOT / "onehot_data"),
        refine=not args.no_refine,
    )
    torch.set_default_dtype(torch.float32)
    infos = [dataset.get_sample_info(i) for i in split_idx]
    dataset.close()

    summarize_eps("TSENN-KKR", pred_t, true)
    summarize_eps("MACE", pred_m, true)

    hamaker_kw = dict(
        t_k=T_K,
        xi_step_ev=2.0 * np.pi * K_B_EV * T_K,
        n_max=args.n_matsubara,
    )
    a_true = np.array([hamaker_constant_zj(xi, true[i], **hamaker_kw) for i in range(len(true))])
    a_t = np.array([hamaker_constant_zj(xi, pred_t[i], **hamaker_kw) for i in range(len(true))])
    a_m = np.array([hamaker_constant_zj(xi, pred_m[i], **hamaker_kw) for i in range(len(true))])
    eps_low = true[:, 0]
    insulator = eps_low <= args.metal_eps_threshold

    print("\n" + "=" * 64)
    print("Hamaker A (self, vacuum gap)")
    print("=" * 64)
    summarize_hamaker("TSENN-KKR", a_t, a_true)
    summarize_hamaker("MACE", a_m, a_true)
    report_hamaker_group("Insulators only — TSENN-KKR", a_true[insulator], a_t[insulator], args.metal_eps_threshold)
    report_hamaker_group("Insulators only — MACE", a_true[insulator], a_m[insulator], args.metal_eps_threshold)

    log_t = np.abs(np.log(np.clip(pred_t, 1e-30, None)) - np.log(np.clip(true, 1e-30, None))).mean(axis=1)
    log_m = np.abs(np.log(np.clip(pred_m, 1e-30, None)) - np.log(np.clip(true, 1e-30, None))).mean(axis=1)
    print("\nHead-to-head (lower error wins)")
    print(f"  ε |log| : TSENN {(log_t < log_m).sum()}/{len(true)}   MACE {(log_m < log_t).sum()}/{len(true)}")
    print(f"  Hamaker : TSENN {(np.abs(a_t - a_true) < np.abs(a_m - a_true)).sum()}/{len(true)}   "
          f"MACE {(np.abs(a_m - a_true) < np.abs(a_t - a_true)).sum()}/{len(true)}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jid", "formula", "eps_low", "log_err_tsenn", "log_err_mace",
                    "a_true_zj", "a_tsenn_zj", "a_mace_zj"])
        for i, info in enumerate(infos):
            w.writerow([
                info.get("jid", ""), info.get("formula", ""), float(eps_low[i]),
                float(log_t[i]), float(log_m[i]),
                float(a_true[i]), float(a_t[i]), float(a_m[i]),
            ])
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
