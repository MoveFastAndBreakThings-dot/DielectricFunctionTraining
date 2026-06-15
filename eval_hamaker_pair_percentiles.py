"""
Evaluate pairwise Hamaker A_12 relative error on the pair-dataset test split.

Works with any FrequencyMACE checkpoint (base or meta-trained).

Relative error (%):
    rel = 100 * (A_pred - A_true) / A_true

Reports signed and absolute relative-error percentiles p5, p10, …, p95.

Usage:
    python3 eval_hamaker_pair_percentiles.py --model_path models/mace_kkr_v1.pt
    python3 eval_hamaker_pair_percentiles.py --model_path models/mace_meta_hamaker_v1.pt --split test
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from hamaker_pair_dataset import HamakerPairDataset, collate_hamaker_pairs, load_pair_splits
from hamaker_torch import ZJ_PER_J, hamaker_pair_j_from_eps
from lmdb_dataset import LMDBCrystalDataset
from model_loader import load_model

K_B_EV = 8.617333e-5
PERCENTILES = list(range(5, 100, 5))


@torch.no_grad()
def predict_pair_hamaker_zj(xi_grid, eps1, eps2, xi_matsubara, temperature_k):
    a_j = hamaker_pair_j_from_eps(
        xi_grid, eps1, eps2, xi_matsubara, t_k=temperature_k,
    )
    return a_j * ZJ_PER_J


@torch.no_grad()
def run_pairs(model, inputs_a, inputs_b, device, use_log):
    af_a, nf_a, nfi_a, cai_a = inputs_a
    af_b, nf_b, nfi_b, cai_b = inputs_b
    af_a = af_a.to(device)
    nf_a = nf_a.to(device)
    nfi_a = nfi_a.to(device)
    cai_a = [x.to(device) for x in cai_a]
    af_b = af_b.to(device)
    nf_b = nf_b.to(device)
    nfi_b = nfi_b.to(device)
    cai_b = [x.to(device) for x in cai_b]

    out_a = model(af_a, nf_a, nfi_a, cai_a)[:, 0, :]
    out_b = model(af_b, nf_b, nfi_b, cai_b)[:, 0, :]
    if use_log:
        return torch.exp(out_a), torch.exp(out_b)
    return out_a, out_b


@torch.no_grad()
def collect_pair_errors(model, loader, xi_grid, xi_matsubara, temperature_k, device, use_log):
    a_true_list, a_pred_list = [], []
    for inputs_a, inputs_b, a_true in loader:
        a_true = a_true.to(device)
        eps_a, eps_b = run_pairs(model, inputs_a, inputs_b, device, use_log)
        a_pred = predict_pair_hamaker_zj(
            xi_grid, eps_a, eps_b, xi_matsubara, temperature_k,
        )
        a_true_list.append(a_true.cpu().numpy())
        a_pred_list.append(a_pred.cpu().numpy())

    a_true = np.concatenate(a_true_list)
    a_pred = np.concatenate(a_pred_list)
    rel_pct = 100.0 * (a_pred - a_true) / (a_true + 1e-30)
    abs_rel_pct = np.abs(rel_pct)
    return a_true, a_pred, rel_pct, abs_rel_pct


def print_percentile_table(name, values, percentiles):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"\n{name}: no finite values")
        return {}
    qs = np.percentile(values, percentiles)
    print(f"\n{name}  (n = {len(values)})")
    print(f"  {'pct':>4}  {'value (%)':>12}")
    print(f"  {'-'*4}  {'-'*12}")
    out = {}
    for p, v in zip(percentiles, qs):
        print(f"  p{p:>3}  {v:12.4f}")
        out[p] = float(v)
    print(f"  mean  {values.mean():12.4f}")
    print(f"  median{np.median(values):12.4f}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise Hamaker relative-error percentiles on pair-dataset test split",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pairs_dir", type=str, default="meta_hamaker_pairs")
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--lmdb_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_matsubara", type=int, default=None)
    args = parser.parse_args()

    meta_path = os.path.join(args.pairs_dir, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    lmdb_path = args.lmdb_path or meta["lmdb_path"]
    n_matsubara = args.n_matsubara or meta["n_matsubara"]
    temperature_k = meta["temperature_k"]
    target_components = tuple(meta["target_components"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device)
    use_log = ckpt.get("use_log", False)
    model, _ = load_model(ckpt, device)

    crystal_ds = LMDBCrystalDataset(
        lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
    )
    splits = load_pair_splits(args.pairs_dir)
    pairs_npz = os.path.join(args.pairs_dir, "pairs.npz")
    pair_ds = HamakerPairDataset(crystal_ds, pairs_npz, splits[args.split])
    loader = DataLoader(
        pair_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_hamaker_pairs,
    )

    xi_grid = torch.tensor(crystal_ds.kkr_xi_array, dtype=torch.float32, device=device)
    xi_step = 2.0 * np.pi * K_B_EV * temperature_k
    xi_matsubara = torch.arange(n_matsubara, dtype=torch.float32, device=device) * xi_step

    model_name = ckpt.get("model_name", os.path.basename(args.model_path))
    meta_obj = ckpt.get("meta_objective", "epsilon")

    print("=" * 56)
    print(f"Model      : {args.model_path}")
    print(f"  name     : {model_name}")
    print(f"  objective: {meta_obj}")
    print(f"  use_log  : {use_log}")
    print(f"Pairs      : {args.pairs_dir}  split={args.split}  n={len(pair_ds)}")
    print(f"Hamaker    : T={temperature_k} K  n_matsubara={n_matsubara}")
    print("=" * 56)

    a_true, a_pred, rel_pct, abs_rel_pct = collect_pair_errors(
        model, loader, xi_grid, xi_matsubara, temperature_k, device, use_log,
    )
    crystal_ds.close()

    mae_zj = float(np.mean(np.abs(a_pred - a_true)))
    rmse_zj = float(np.sqrt(np.mean((a_pred - a_true) ** 2)))
    r = float(np.corrcoef(a_pred, a_true)[0, 1]) if len(a_true) > 1 else float("nan")

    print(f"\nSummary (zJ)")
    print(f"  MAE(ΔA)   = {mae_zj:.4f}")
    print(f"  RMSE(ΔA)  = {rmse_zj:.4f}")
    print(f"  Pearson r = {r:.4f}")

    print_percentile_table("Signed relative error  (A_pred − A_true) / A_true × 100", rel_pct, PERCENTILES)
    print_percentile_table("Absolute relative error  |A_pred − A_true| / |A_true| × 100", abs_rel_pct, PERCENTILES)


if __name__ == "__main__":
    main()
