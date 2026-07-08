"""
Batch structural sensitivity stats for official MACE (autograd).

For each crystal in a split:
  1. Per-material sensitivity summaries at low / mid / high xi
  2. Population percentile tables across materials
  3. Correlation of sensitivity vs core epsilon prediction error

Usage:
    python sensitivity_stats_mace.py \\
        --model_path models/official_mace_kkr_v1.pt \\
        --split test \\
        --max_samples 100
"""

import argparse
import os

import numpy as np
import torch

from compare_hamaker_constants import get_split_indices
from lmdb_dataset import LMDBCrystalDataset
from mace_data import structure_arrays_to_atomic_data, z_table_from_atom_init
from mace.tools.torch_geometric.batch import Batch
from sensitivity_utils import (
    load_checkpoint_model,
    load_sample_arrays,
    material_sensitivity_summary,
    plot_stats,
    print_percentile_table,
    print_sensitivity_correlations,
    save_stats_csv,
)


def forward_eps_at_freq(model, batch, use_log, channel, freq_idx, wrt_linear):
    out = model(batch)[0, channel]
    if use_log and wrt_linear:
        return torch.exp(out[freq_idx])
    if use_log and not wrt_linear:
        return out[freq_idx]
    return out[freq_idx]


def core_mae(pred, target):
    return float(torch.abs(pred - target).mean().cpu())


def analyze_crystal(
    model, device, raw, info, z_table, r_max, use_log, channel,
    freq_indices, wrt_linear, target_components,
):
    graph = structure_arrays_to_atomic_data(
        raw["positions"],
        raw["cell"],
        raw["atomic_numbers"],
        raw["pbc"],
        z_table,
        r_max,
    )
    target_parts = [raw["kkr_targets"][f"kkr_{comp}"] for comp in target_components]
    target_lin = torch.as_tensor(np.stack(target_parts, axis=0), dtype=torch.float32, device=device)
    if use_log:
        target_model = torch.log(target_lin.clamp(min=1e-30))
    else:
        target_model = target_lin

    batch_eval = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        pred = model(batch_eval)[0]
        mae = core_mae(pred, target_model)

    freq_rows = []
    for freq_idx in freq_indices:
        graph_grad = structure_arrays_to_atomic_data(
            raw["positions"],
            raw["cell"],
            raw["atomic_numbers"],
            raw["pbc"],
            z_table,
            r_max,
        )
        graph_grad.positions = graph_grad.positions.to(device).detach().requires_grad_(True)
        batch_grad = Batch.from_data_list([graph_grad]).to(device)

        target_eps = forward_eps_at_freq(
            model, batch_grad, use_log, channel, freq_idx, wrt_linear
        )
        grad = torch.autograd.grad(target_eps, batch_grad.positions)[0]
        per_atom = grad.detach().cpu().numpy()
        per_atom_norm = np.linalg.norm(per_atom, axis=1)
        summary = material_sensitivity_summary(per_atom_norm)

        freq_rows.append({
            "freq_idx": freq_idx,
            "eps_pred": float(target_eps.detach().cpu()),
            **summary,
        })

    return {
        "mpid": info["mpid"],
        "formula": info["formula"],
        "n_atoms": int(raw["n_atoms"]),
        "opt_gap": info.get("opt_gap"),
        "core_mae": mae,
        "freq_rows": freq_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch MACE sensitivity stats")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lmdb_path", default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--freq_low", type=int, default=10)
    parser.add_argument("--freq_mid", type=int, default=100)
    parser.add_argument("--freq_high", type=int, default=400)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--wrt", choices=("linear", "log"), default="linear")
    parser.add_argument("--out_dir", default="structural_sensitivity/stats")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, model_type = load_checkpoint_model(args.model_path, device)
    if model_type != "MACE":
        raise SystemExit("This script requires an official MACE checkpoint.")

    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))
    r_max = ckpt.get("mace_r_max", ckpt.get("radius", 8.0))
    model_name = ckpt.get("model_name", os.path.basename(args.model_path))
    wrt_linear = args.wrt == "linear"
    freq_indices = [args.freq_low, args.freq_mid, args.freq_high]

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        use_mace=True,
        mace_r_max=r_max,
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    z_table = z_table_from_atom_init()

    indices = get_split_indices(ckpt, len(dataset), args.split)
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    os.makedirs(args.out_dir, exist_ok=True)
    records = []

    print("=" * 60)
    print(f"Model : {args.model_path} ({model_name})")
    print(f"Split : {args.split}  n={len(indices)}")
    print(f"Freq indices (low/mid/high): {freq_indices}")
    print(f"xi (eV): {[float(xi[i]) for i in freq_indices]}")
    print("=" * 60)

    for n_done, idx in enumerate(indices, 1):
        info = dataset.get_sample_info(idx)
        raw = load_sample_arrays(dataset, idx)
        rec = analyze_crystal(
            model, device, raw, info, z_table, r_max,
            use_log, args.channel, freq_indices, wrt_linear, target_components,
        )
        records.append(rec)
        if n_done % 25 == 0 or n_done == len(indices):
            print(f"  processed {n_done}/{len(indices)}")

    csv_path = os.path.join(args.out_dir, f"sensitivity_{args.split}.csv")
    save_stats_csv(csv_path, records, xi)
    plot_stats(args.out_dir, records, freq_indices, xi, model_name)

    space = "log(eps)" if use_log else "eps"
    print(f"\n--- Per-material core epsilon MAE ({space}) ---")
    print_percentile_table("core_mae", [rec["core_mae"] for rec in records])

    band_names = ["low", "mid", "high"]
    for band, freq_idx in zip(band_names, freq_indices):
        rows = [
            row
            for rec in records
            for row in rec["freq_rows"]
            if row["freq_idx"] == freq_idx
        ]
        print(f"\n--- {band.upper()} xi (idx={freq_idx}, xi={xi[freq_idx]:.4g} eV) ---")
        for metric in ("max_sens", "mean_sens", "p90_sens", "concentration"):
            print_percentile_table(metric, [row[metric] for row in rows])

    print_sensitivity_correlations(records, freq_indices, xi)

    print(f"\nSaved CSV  : {csv_path}")
    print(f"Saved plots: {args.out_dir}/sensitivity_distributions.png")
    print(f"             {args.out_dir}/sensitivity_vs_mae.png")

    dataset.close()


if __name__ == "__main__":
    main()
