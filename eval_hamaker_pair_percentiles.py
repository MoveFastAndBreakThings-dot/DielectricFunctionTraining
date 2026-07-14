"""
Evaluate a FrequencyMACE checkpoint on:
  1. Core task — ε(iξ) loss from main training (FrequencyWeightedMSELoss)
  2. Downstream — pairwise Hamaker A_12 relative error on the pair-dataset test split

Works with base or meta-trained checkpoints.

Usage:
    python3 eval_hamaker_pair_percentiles.py --model_path models/mace_kkr_v1.pt
    python3 eval_hamaker_pair_percentiles.py --model_path models/mace_meta_hamaker_v1.pt
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from compare_hamaker_constants import get_split_indices
from hamaker_pair_dataset import HamakerPairDataset, get_hamaker_collate, load_pair_splits
from hamaker_torch import ZJ_PER_J, hamaker_pair_j_from_eps
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from losses import FrequencyWeightedMSELoss
from model_forward import model_forward_train
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
def run_pairs(model, model_type, inputs_a, inputs_b, device, use_log):
    out_a = model_forward_train(model, model_type, inputs_a, device)[:, 0, :]
    out_b = model_forward_train(model, model_type, inputs_b, device)[:, 0, :]
    if use_log:
        return torch.exp(out_a), torch.exp(out_b)
    return out_a, out_b


@torch.no_grad()
def eval_core_loss(model, model_type, loader, criterion, device, use_log):
    """Return (mean loss, per-sample losses) using main-training objective."""
    per_sample = []
    total, n = 0.0, 0
    for inputs, target, is_metal in loader:
        target = target.to(device)
        if use_log:
            target = torch.log(target)
        if is_metal is not None:
            is_metal = is_metal.to(device)

        pred = model_forward_train(model, model_type, inputs, device)
        loss = criterion(pred, target, is_metal)
        bs = target.shape[0]
        total += loss.item() * bs
        n += bs

        abs_err = torch.abs(pred - target)
        if is_metal is not None and criterion.n_low_freq > 0:
            w = torch.ones(bs, 1, pred.shape[2], device=device)
            w[is_metal, :, :criterion.n_low_freq] = criterion.metal_low_freq_weight
            abs_err = abs_err * w
        per_sample.append(abs_err.mean(dim=(1, 2)).cpu().numpy())

    mean_loss = total / max(n, 1)
    return mean_loss, np.concatenate(per_sample)


@torch.no_grad()
def collect_pair_errors(model, model_type, loader, xi_grid, xi_matsubara, temperature_k, device, use_log):
    a_true_list, a_pred_list = [], []
    for inputs_a, inputs_b, a_true in loader:
        a_true = a_true.to(device)
        eps_a, eps_b = run_pairs(model, model_type, inputs_a, inputs_b, device, use_log)
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


def print_percentile_table(name, values, percentiles, unit=""):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        print(f"\n{name}: no finite values")
        return
    qs = np.percentile(values, percentiles)
    print(f"\n{name}  (n = {len(values)})")
    print(f"  {'pct':>4}  {'value':>12}{unit}")
    print(f"  {'-'*4}  {'-'*12}")
    for p, v in zip(percentiles, qs):
        print(f"  p{p:>3}  {v:12.4f}{unit}")
    print(f"  mean  {values.mean():12.4f}{unit}")
    print(f"  median{np.median(values):12.4f}{unit}")


def main():
    parser = argparse.ArgumentParser(
        description="Core ε(iξ) loss + pairwise Hamaker percentiles for model comparison",
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pairs_dir", type=str, default="meta_hamaker_pairs")
    parser.add_argument("--pair_split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--core_split", type=str, default="test", choices=("train", "val", "test"),
                        help="Crystal split for core ε(iξ) loss")
    parser.add_argument("--lmdb_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_matsubara", type=int, default=None)
    parser.add_argument("--skip_core", action="store_true")
    parser.add_argument("--skip_pairs", action="store_true")
    args = parser.parse_args()

    meta_path = os.path.join(args.pairs_dir, "metadata.json")
    with open(meta_path) as f:
        pair_meta = json.load(f)

    lmdb_path = args.lmdb_path or pair_meta["lmdb_path"]
    n_matsubara = args.n_matsubara or pair_meta["n_matsubara"]
    temperature_k = pair_meta["temperature_k"]
    target_components = tuple(pair_meta["target_components"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device)
    use_log = ckpt.get("use_log", False)
    model, model_type = load_model(ckpt, device)
    use_mace = model_type == "MACE"

    crystal_ds = LMDBCrystalDataset(
        lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt.get("mace_r_max"),
    )
    # Foundation / MP-0 finetunes store their own element table — must match encoder.
    if use_mace and ckpt.get("mace_atomic_numbers") is not None:
        from mace.tools import AtomicNumberTable
        crystal_ds.set_mace_graph_config(
            AtomicNumberTable(zs=[int(z) for z in ckpt["mace_atomic_numbers"]]),
            ckpt.get("mace_r_max", crystal_ds.mace_r_max),
        )

    model_name = ckpt.get("model_name", os.path.basename(args.model_path))
    meta_obj = ckpt.get("meta_objective", "epsilon")

    print("=" * 60)
    print(f"Model      : {args.model_path}")
    print(f"  name     : {model_name}")
    print(f"  type     : {model_type}")
    print(f"  objective: {meta_obj}")
    print(f"  use_log  : {use_log}")
    print("=" * 60)

    if not args.skip_core:
        n_low_freq = ckpt.get("n_low_freq", 0)
        metal_low_freq_weight = ckpt.get("metal_low_freq_weight", 0.1)
        core_idx = get_split_indices(ckpt, len(crystal_ds), args.core_split)
        core_loader = DataLoader(
            Subset(crystal_ds, core_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=get_collate_fn(use_mace=use_mace),
        )
        criterion = FrequencyWeightedMSELoss(
            crystal_ds.frequency_array,
            n_low_freq=n_low_freq,
            metal_low_freq_weight=metal_low_freq_weight,
        ).to(device)

        space = "log(ε)" if use_log else "ε"
        print(f"\n--- Core ε(iξ) loss (main training objective) ---")
        print(f"  Loss: FrequencyWeightedMSELoss  (= weighted MAE on {space})")
        print(f"  mean |pred − target| over all channels & ξ points")
        if n_low_freq > 0:
            print(f"  metal low-ξ discount: first {n_low_freq} pts × {metal_low_freq_weight}")
        print(f"  split: {args.core_split}  n_crystals={len(core_idx)}")

        core_mean, core_per_sample = eval_core_loss(
            model, model_type, core_loader, criterion, device, use_log,
        )
        ckpt_val = ckpt.get("val_loss")
        val_note = f"  (checkpoint val_loss={ckpt_val:.6f})" if ckpt_val is not None else ""
        print(f"\n  Test loss (batch mean): {core_mean:.6f}{val_note}")
        print_percentile_table(
            f"Per-crystal core loss ({space})",
            core_per_sample,
            PERCENTILES,
        )

    if not args.skip_pairs:
        splits = load_pair_splits(args.pairs_dir)
        pairs_npz = os.path.join(args.pairs_dir, "pairs.npz")
        pair_ds = HamakerPairDataset(crystal_ds, pairs_npz, splits[args.pair_split])
        pair_loader = DataLoader(
            pair_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=get_hamaker_collate(use_mace=use_mace),
        )

        xi_grid = torch.tensor(crystal_ds.kkr_xi_array, dtype=torch.float32, device=device)
        xi_matsubara = (
            torch.arange(n_matsubara, dtype=torch.float32, device=device)
            * (2.0 * np.pi * K_B_EV * temperature_k)
        )

        print(f"\n--- Pairwise Hamaker A_12 (downstream) ---")
        print(f"  pairs: {args.pairs_dir}  split={args.pair_split}  n={len(pair_ds)}")
        print(f"  T={temperature_k} K  n_matsubara={n_matsubara}")

        a_true, a_pred, rel_pct, abs_rel_pct = collect_pair_errors(
            model, model_type, pair_loader, xi_grid, xi_matsubara, temperature_k, device, use_log,
        )

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
