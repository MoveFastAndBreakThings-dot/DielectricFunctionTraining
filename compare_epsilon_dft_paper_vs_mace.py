"""
Compare ε(iξ) errors vs DFT (KKR ground truth):

  1. Gudarzi & Aboutalebi (2021) modified harmonic oscillator — bandgap (+ formula for B) only
  2. FrequencyMACE checkpoint

Prints mean and percentile table; optional CSV.

Usage:
    python3 compare_epsilon_dft_paper_vs_mace.py --model_path models/mace_kkr_v1.pt
    python3 compare_epsilon_dft_paper_vs_mace.py \\
        --model_path models/mace_kkr_v1.pt --meta_path models/mace_meta_hamaker_v1.pt
"""

import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from compare_hamaker_constants import get_split_indices
from gudarzi_mho import eps_mho_from_bandgap
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model

PERCENTILES = [5, 25, 50, 75, 95]


def load_mace(path, device):
    ckpt = torch.load(path, map_location=device)
    model, _ = load_model(ckpt, device)
    return ckpt, model


@torch.no_grad()
def predict_mace(model, loader, device, use_log):
    pred_list, true_list = [], []
    for inputs, target, _ in loader:
        af, nf, nfi, cai = inputs
        af = af.to(device)
        nf = nf.to(device)
        nfi = nfi.to(device)
        cai = [x.to(device) for x in cai]
        target = target.to(device)
        out = model(af, nf, nfi, cai)[:, 0, :]
        pred_lin = torch.exp(out) if use_log else out
        pred_list.append(pred_lin.cpu().numpy())
        true_list.append(target[:, 0, :].cpu().numpy())
    return np.concatenate(pred_list), np.concatenate(true_list)


def per_crystal_mae(pred, true):
    return np.mean(np.abs(pred - true), axis=1)


def per_crystal_abs_rel_pct(pred, true):
    return 100.0 * np.mean(np.abs(pred - true) / (np.abs(true) + 1e-30), axis=1)


def per_crystal_log_mae(pred, true):
    lp = np.log(np.clip(pred, 1e-30, None))
    lt = np.log(np.clip(true, 1e-30, None))
    return np.mean(np.abs(lp - lt), axis=1)


def summarize(name, values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"name": name, "n": 0}
    pct = np.percentile(values, PERCENTILES)
    return {
        "name": name,
        "n": len(values),
        "mean": float(values.mean()),
        "percentiles": {p: float(v) for p, v in zip(PERCENTILES, pct)},
    }


def print_table(rows, metric_label):
    cols = [r["name"] for r in rows if r.get("n", 0) > 0]
    if not cols:
        print(f"\n{metric_label}: no valid samples")
        return
    w = max(28, max(len(c) for c in cols) + 2)
    print(f"\n{metric_label}")
    print(f"  {'stat':<12}" + "".join(f"{c:>{w}}" for c in cols))
    print(f"  {'-'*12}" + "-" * w * len(cols))
    print(f"  {'mean':<12}" + "".join(f"{r['mean']:>{w}.4f}" for r in rows if r.get("n")))
    for p in PERCENTILES:
        line = f"  {f'p{p}':<12}"
        for r in rows:
            if not r.get("n"):
                continue
            line += f"{r['percentiles'][p]:>{w}.4f}"
        print(line)
    print(f"  {'n':<12}" + "".join(f"{r['n']:>{w}d}" for r in rows if r.get("n")))


def main():
    parser = argparse.ArgumentParser(
        description="ε(iξ) error vs DFT: Gudarzi MHO (bandgap) vs MACE",
    )
    parser.add_argument("--model_path", type=str, required=True, help="MACE checkpoint")
    parser.add_argument("--meta_path", type=str, default=None, help="Optional meta checkpoint")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gap_field", type=str, default="opt_gap", choices=("opt_gap", "mbj_gap"))
    parser.add_argument("--metal_gap_threshold", type=float, default=0.1,
                        help="Skip materials with gap below this (MHO is for insulators/semiconductors)")
    parser.add_argument("--alpha", type=float, default=1.7, help="MHO power exponent")
    parser.add_argument("--a_pref", type=float, default=20.0, help="MHO C_UV prefactor (A in paper)")
    parser.add_argument("--out_csv", type=str, default=None, help="Save per-material errors to CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model = load_mace(args.model_path, device)
    use_log = ckpt.get("use_log", False)
    target_components = tuple(ckpt.get("target_components", ("xx",)))

    meta_pred = None
    meta_name = None
    if args.meta_path:
        ckpt_m, model_m = load_mace(args.meta_path, device)
        meta_name = ckpt_m.get("model_name", os.path.basename(args.meta_path))

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

    model_name = ckpt.get("model_name", os.path.basename(args.model_path))
    print("=" * 72)
    print(f"DFT reference : KKR ε(iξ) from {args.lmdb_path}")
    print(f"Paper model   : Gudarzi & Aboutalebi MHO (bandgap + formula → B)")
    print(f"  gap field={args.gap_field}  α={args.alpha}  A={args.a_pref}")
    print(f"  exclude gap < {args.metal_gap_threshold} eV")
    print(f"MACE          : {args.model_path} ({model_name})")
    if args.meta_path:
        print(f"MACE meta     : {args.meta_path} ({meta_name})")
    print(f"Split         : {args.split}  n={len(split_idx)}")
    print("=" * 72)

    pred_mace, true = predict_mace(model, loader, device, use_log)
    if args.meta_path:
        meta_pred, _ = predict_mace(model_m, loader, device, ckpt_m.get("use_log", use_log))

    mpids, formulas, gaps = [], [], []
    pred_paper_list, valid_mask = [], []

    for j, global_i in enumerate(split_idx):
        info = dataset.get_sample_info(global_i)
        mpids.append(info["mpid"])
        formulas.append(info["formula"])
        gap_raw = info.get(args.gap_field)
        gap = float(gap_raw) if gap_raw is not None else float("nan")
        gaps.append(gap)

        ok = np.isfinite(gap) and gap >= args.metal_gap_threshold
        if ok:
            try:
                pred_paper_list.append(
                    eps_mho_from_bandgap(
                        xi, gap, formula=info["formula"],
                        alpha=args.alpha, a_pref=args.a_pref,
                    )
                )
                valid_mask.append(True)
            except ValueError:
                valid_mask.append(False)
        else:
            valid_mask.append(False)

    dataset.close()
    valid_mask = np.array(valid_mask, dtype=bool)
    n_valid = int(valid_mask.sum())
    print(f"\nMHO evaluated on {n_valid}/{len(split_idx)} crystals (finite gap ≥ threshold)")

    pred_paper = np.stack(pred_paper_list) if n_valid else None
    true_v = true[valid_mask]
    pred_mace_v = pred_mace[valid_mask]
    pred_meta_v = meta_pred[valid_mask] if meta_pred is not None else None

    methods = [("Paper MHO (Eg)", pred_paper)]
    methods.append((f"MACE ({model_name})", pred_mace_v))
    if pred_meta_v is not None:
        methods.append((f"MACE meta ({meta_name})", pred_meta_v))

    mae_rows, rel_rows, log_rows = [], [], []
    for label, pred in methods:
        if pred is None:
            continue
        mae_rows.append(summarize(label, per_crystal_mae(pred, true_v)))
        rel_rows.append(summarize(label, per_crystal_abs_rel_pct(pred, true_v)))
        log_rows.append(summarize(label, per_crystal_log_mae(pred, true_v)))

    print_table(mae_rows, "MAE on ε(iξ)  (lower is better)")
    print_table(rel_rows, "Mean |pred − DFT| / |DFT| × 100  (%)")
    print_table(log_rows, "Mean |log ε_pred − log ε_DFT|  (lower is better)")

    if args.out_csv and n_valid:
        csv_path = args.out_csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            header = [
                "mpid", "formula", "gap_ev",
                "mae_paper", "rel_pct_paper", "log_mae_paper",
                f"mae_{model_name}", f"rel_pct_{model_name}", f"log_mae_{model_name}",
            ]
            if pred_meta_v is not None:
                header.extend([
                    f"mae_{meta_name}", f"rel_pct_{meta_name}", f"log_mae_{meta_name}",
                ])
            w.writerow(header)
            vi = 0
            for i in range(len(split_idx)):
                if not valid_mask[i]:
                    continue
                row = [
                    mpids[i], formulas[i], gaps[i],
                    per_crystal_mae(pred_paper[vi:vi+1], true_v[vi:vi+1])[0],
                    per_crystal_abs_rel_pct(pred_paper[vi:vi+1], true_v[vi:vi+1])[0],
                    per_crystal_log_mae(pred_paper[vi:vi+1], true_v[vi:vi+1])[0],
                    per_crystal_mae(pred_mace_v[vi:vi+1], true_v[vi:vi+1])[0],
                    per_crystal_abs_rel_pct(pred_mace_v[vi:vi+1], true_v[vi:vi+1])[0],
                    per_crystal_log_mae(pred_mace_v[vi:vi+1], true_v[vi:vi+1])[0],
                ]
                if pred_meta_v is not None:
                    row.extend([
                        per_crystal_mae(pred_meta_v[vi:vi+1], true_v[vi:vi+1])[0],
                        per_crystal_abs_rel_pct(pred_meta_v[vi:vi+1], true_v[vi:vi+1])[0],
                        per_crystal_log_mae(pred_meta_v[vi:vi+1], true_v[vi:vi+1])[0],
                    ])
                w.writerow(row)
                vi += 1
        print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
