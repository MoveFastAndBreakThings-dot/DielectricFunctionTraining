#!/usr/bin/env python3
"""
Fair TSENN vs OPT DFT eval on a *truncated* 0–ω_max eV window (default 30 eV).

TSENN predicts ε₂(ω) on 0–30 eV. We:
  1) Truncate JARVIS OPT ε₂(ω) to the same window
  2) KKR both → ε(iξ) on the OPT LMDB ξ grid
  3) Report metrics analogous to eval_hamaker_pair_percentiles.py:
       - core MAE on ε(iξ) (xx)
       - self-Hamaker A percentiles
       - optional pairwise A_12 (recomputed from truncated DFT, not pairs.npz A)

Offline / sbatch-safe.

Usage:
  python eval_tsenn_opt_truncated.py
  python eval_tsenn_opt_truncated.py --omega_max 30 --component xx
  python eval_tsenn_opt_truncated.py --pairs_dir meta_hamaker_pairs_opt
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
from tqdm import tqdm

from compare_hamaker_constants import (
    T_K,
    hamaker_constant_zj,
    hamaker_pair_constant_zj,
)
from eval_hamaker_pair_percentiles import PERCENTILES, print_percentile_table
from preprocess_dataset import apply_kkr
from train_mace_frequency import train_val_test_split


def load_pred_lmdb(path):
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    preds = {}
    with env.begin() as txn:
        for key, val in txn.cursor():
            if key == b"__metadata__":
                continue
            rec = pickle.loads(val)
            preds[rec["jid"]] = rec
    env.close()
    return preds


def load_opt_by_jid(lmdb_path):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    by_jid = {}
    xi = None
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        xi = np.asarray(meta["kkr_xi_array"], dtype=np.float64)
        n = meta["total_samples"]
        for i in range(n):
            raw = txn.get(f"{i}".encode("utf-8"))
            if raw is None:
                break
            rec = pickle.loads(raw)
            by_jid[rec["jid"]] = (i, rec)
    env.close()
    return by_jid, xi


def imag_component(rec, component):
    """DFT ε₂(ω) from OPT LMDB record."""
    if component == "iso":
        xx = np.asarray(rec["targets"]["imag_xx"], dtype=np.float64)
        yy = np.asarray(rec["targets"]["imag_yy"], dtype=np.float64)
        zz = np.asarray(rec["targets"]["imag_zz"], dtype=np.float64)
        return (xx + yy + zz) / 3.0
    return np.asarray(rec["targets"][f"imag_{component}"], dtype=np.float64)


def tsenn_imag_component(pred, component):
    """TSENN ε₂ tensor (F, 3, 3) → 1D spectrum."""
    eps2 = np.asarray(pred["eps2"], dtype=np.float64)
    if component == "iso":
        return (eps2[:, 0, 0] + eps2[:, 1, 1] + eps2[:, 2, 2]) / 3.0
    idx = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yz": (1, 2), "zx": (2, 0)}
    i, j = idx[component]
    return eps2[:, i, j]


def truncate_spectrum(omega, eps2, omega_max):
    omega = np.asarray(omega, dtype=np.float64)
    eps2 = np.asarray(eps2, dtype=np.float64)
    mask = (omega >= 0.0) & (omega <= omega_max)
    if mask.sum() < 4:
        raise ValueError(f"too few points after truncate (n={mask.sum()})")
    return omega[mask], eps2[mask]


def kkr_eps_ixi(omega, eps2, xi):
    _, eps_kkr = apply_kkr(omega, np.clip(eps2, 0.0, None), xi_points=xi)
    return np.asarray(eps_kkr, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(
        description="TSENN vs OPT DFT on truncated ω window → ε(iξ) + Hamaker",
    )
    parser.add_argument("--pred_lmdb", default="tsenn_opt/predictions.lmdb")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--omega_max", type=float, default=30.0)
    parser.add_argument(
        "--component",
        default="xx",
        choices=("xx", "yy", "zz", "xy", "yz", "zx", "iso"),
    )
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "train", "val", "test"),
        help="Crystal split via seed=42 80/10/10 on full OPT LMDB indices",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs_dir", default=None, help="Optional pair dir for A_12 metrics")
    parser.add_argument("--pair_split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out_json", default="tsenn_opt/eval_truncated.json")
    args = parser.parse_args()

    preds = load_pred_lmdb(args.pred_lmdb)
    opt_by_jid, xi = load_opt_by_jid(args.lmdb_path)
    print(f"TSENN preds: {len(preds)}  OPT LMDB: {len(opt_by_jid)}  ξ points: {len(xi)}")
    print(f"Truncation: ω ∈ [0, {args.omega_max}] eV  component={args.component}")

    # Optional crystal split on OPT LMDB index space (same recipe as training)
    n_opt = max(i for i, _ in opt_by_jid.values()) + 1 if opt_by_jid else 0
    allowed_idx = None
    if args.split != "all":
        train_idx, val_idx, test_idx = train_val_test_split(
            n_opt, train_ratio=0.8, val_ratio=0.1, seed=args.seed
        )
        allowed_idx = {
            "train": set(train_idx),
            "val": set(val_idx),
            "test": set(test_idx),
        }[args.split]
        print(f"Split={args.split}  n_allowed_idx={len(allowed_idx)}")

    matched = []
    for jid, pred in preds.items():
        if jid not in opt_by_jid:
            continue
        idx, opt_rec = opt_by_jid[jid]
        if allowed_idx is not None and idx not in allowed_idx:
            continue
        matched.append((jid, idx, pred, opt_rec))
    matched.sort(key=lambda x: x[1])
    if args.limit is not None:
        matched = matched[: args.limit]
    print(f"Matched for eval: {len(matched)}")

    core_mae = []
    a_true_list, a_pred_list = [], []
    jid_to_eps = {}  # truncated-KKR ε(iξ) for pairs
    n_fail = 0

    hamaker_kw = dict(t_k=args.temperature_k, n_max=args.n_matsubara)

    for jid, idx, pred, opt_rec in tqdm(matched, desc="eval"):
        try:
            omega_dft = np.asarray(opt_rec["opt_en"], dtype=np.float64)
            eps2_dft = imag_component(opt_rec, args.component)
            w_dft, e_dft = truncate_spectrum(omega_dft, eps2_dft, args.omega_max)
            eps_true = kkr_eps_ixi(w_dft, e_dft, xi)

            omega_p = np.asarray(pred["omega"], dtype=np.float64)
            eps2_p = tsenn_imag_component(pred, args.component)
            w_p, e_p = truncate_spectrum(omega_p, eps2_p, args.omega_max)
            eps_pred = kkr_eps_ixi(w_p, e_p, xi)
        except Exception:
            n_fail += 1
            continue

        core_mae.append(float(np.mean(np.abs(eps_pred - eps_true))))
        a_true_list.append(hamaker_constant_zj(xi, eps_true, **hamaker_kw))
        a_pred_list.append(hamaker_constant_zj(xi, eps_pred, **hamaker_kw))
        jid_to_eps[jid] = (eps_true, eps_pred)

    core_mae = np.asarray(core_mae, dtype=np.float64)
    a_true = np.asarray(a_true_list, dtype=np.float64)
    a_pred = np.asarray(a_pred_list, dtype=np.float64)
    rel_pct = 100.0 * (a_pred - a_true) / (a_true + 1e-30)
    abs_rel_pct = np.abs(rel_pct)

    print("\n" + "=" * 60)
    print("TSENN vs OPT DFT  (truncated-ω KKR, fair window)")
    print(f"  ω_max={args.omega_max} eV  component={args.component}  n={len(core_mae)}  fail={n_fail}")
    print("=" * 60)

    print("\n--- Core ε(iξ) MAE (truncated DFT vs TSENN→KKR) ---")
    print(f"  mean |ε_pred − ε_true| over ξ points")
    print(f"  Test mean: {core_mae.mean():.6f}" if len(core_mae) else "  (empty)")
    print_percentile_table("Per-crystal core MAE (ε)", core_mae, PERCENTILES)

    print("\n--- Self-Hamaker A (from truncated-KKR ε) ---")
    if len(a_true):
        mae_zj = float(np.mean(np.abs(a_pred - a_true)))
        rmse_zj = float(np.sqrt(np.mean((a_pred - a_true) ** 2)))
        r = float(np.corrcoef(a_pred, a_true)[0, 1]) if len(a_true) > 1 else float("nan")
        print(f"  MAE(ΔA)   = {mae_zj:.4f} zJ")
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
    else:
        mae_zj = rmse_zj = r = float("nan")

    pair_stats = None
    if args.pairs_dir:
        pairs_dir = Path(args.pairs_dir)
        splits = json.loads((pairs_dir / "splits.json").read_text())
        pair_idx = splits[args.pair_split]
        data = np.load(pairs_dir / "pairs.npz")
        # Build index → jid for OPT LMDB
        idx_to_jid = {i: jid for jid, (i, _) in opt_by_jid.items()}
        a12_true, a12_pred = [], []
        n_skip = 0
        for p in tqdm(pair_idx, desc="pairs"):
            ia = int(data["idx_a"][p])
            ib = int(data["idx_b"][p])
            ja, jb = idx_to_jid.get(ia), idx_to_jid.get(ib)
            if ja not in jid_to_eps or jb not in jid_to_eps:
                n_skip += 1
                continue
            eps_ta, eps_pa = jid_to_eps[ja]
            eps_tb, eps_pb = jid_to_eps[jb]
            a12_true.append(hamaker_pair_constant_zj(xi, eps_ta, eps_tb, **hamaker_kw))
            a12_pred.append(hamaker_pair_constant_zj(xi, eps_pa, eps_pb, **hamaker_kw))

        a12_true = np.asarray(a12_true, dtype=np.float64)
        a12_pred = np.asarray(a12_pred, dtype=np.float64)
        print(f"\n--- Pairwise Hamaker A_12 (truncated KKR; split={args.pair_split}) ---")
        print(f"  evaluated={len(a12_true)}  skipped(missing TSENN)={n_skip}")
        if len(a12_true):
            rel12 = 100.0 * (a12_pred - a12_true) / (a12_true + 1e-30)
            abs_rel12 = np.abs(rel12)
            mae12 = float(np.mean(np.abs(a12_pred - a12_true)))
            rmse12 = float(np.sqrt(np.mean((a12_pred - a12_true) ** 2)))
            r12 = float(np.corrcoef(a12_pred, a12_true)[0, 1]) if len(a12_true) > 1 else float("nan")
            print(f"  MAE(ΔA)   = {mae12:.4f} zJ")
            print(f"  RMSE(ΔA)  = {rmse12:.4f} zJ")
            print(f"  Pearson r = {r12:.4f}")
            print_percentile_table(
                "Signed relative error  (A_pred − A_true) / A_true × 100",
                rel12,
                PERCENTILES,
                unit="%",
            )
            print_percentile_table(
                "Absolute relative error  |A_pred − A_true| / |A_true| × 100",
                abs_rel12,
                PERCENTILES,
                unit="%",
            )
            pair_stats = {
                "n": len(a12_true),
                "n_skip": n_skip,
                "mae_zj": mae12,
                "rmse_zj": rmse12,
                "pearson_r": r12,
            }

    out = {
        "omega_max": args.omega_max,
        "component": args.component,
        "n_matched": len(core_mae),
        "n_fail": n_fail,
        "split": args.split,
        "core_mae_mean": float(core_mae.mean()) if len(core_mae) else None,
        "core_mae_median": float(np.median(core_mae)) if len(core_mae) else None,
        "hamaker_mae_zj": mae_zj,
        "hamaker_rmse_zj": rmse_zj,
        "hamaker_pearson_r": r,
        "pairs": pair_stats,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
