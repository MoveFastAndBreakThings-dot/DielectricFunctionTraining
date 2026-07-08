"""
Batch structural sensitivity stats for CGCNN / quasi-MACE (finite differences).

Same outputs as sensitivity_stats_mace.py but uses central differences because
the CGCNN graph is rebuilt from perturbed coordinates.

WARNING: very slow — ~ (1 + 3 freqs × 6 × N_atoms) forward passes per crystal.
Use --max_samples (default 50).

Usage:
    python sensitivity_stats_cgcnn.py \\
        --model_path models/mace_kkr_v2.pt \\
        --split test \\
        --max_samples 50
"""

import argparse
import os

import numpy as np
import torch

from compare_hamaker_constants import get_split_indices
from lmdb_dataset import LMDBCrystalDataset, collate_crystal_graphs
from preprocess_dataset import GaussianDistance, load_atom_init
from sensitivity_utils import (
    load_checkpoint_model,
    load_sample_arrays,
    material_sensitivity_summary,
    plot_stats,
    print_percentile_table,
    print_sensitivity_correlations,
    save_stats_csv,
)
from structural_sensitivity_cgcnn import (
    cgcnn_graph_from_structure,
    finite_difference_grad,
    structure_from_arrays,
)


@torch.no_grad()
def predict_spectrum(model, device, atom_fea, nbr_fea, nbr_fea_idx, channel):
    inputs = collate_crystal_graphs([(atom_fea, nbr_fea, nbr_fea_idx)])
    af, nf, nfi, cai = inputs
    af = af.to(device)
    nf = nf.to(device)
    nfi = nfi.to(device)
    cai = [x.to(device) for x in cai]
    return model(af, nf, nfi, cai)[0, channel]


def core_mae(pred, target):
    return float(torch.abs(pred - target).mean().cpu())


def analyze_crystal(
    model,
    device,
    raw,
    info,
    atom_init,
    radius,
    max_num_nbr,
    gdf,
    use_log,
    channel,
    freq_indices,
    delta,
    target_components,
):
    if "positions" not in raw:
        raise RuntimeError(
            f"{info['mpid']} lacks geometry in LMDB — re-run preprocess_dataset.py"
        )

    structure = structure_from_arrays(
        raw["positions"], raw["cell"], raw["atomic_numbers"], raw["pbc"],
    )
    atom_fea, nbr_fea, nbr_fea_idx = cgcnn_graph_from_structure(
        structure, atom_init, radius, max_num_nbr, gdf,
    )

    target_parts = [raw["kkr_targets"][f"kkr_{comp}"] for comp in target_components]
    target_lin = torch.as_tensor(np.stack(target_parts, axis=0), dtype=torch.float32, device=device)
    if use_log:
        target_model = torch.log(target_lin.clamp(min=1e-30))
    else:
        target_model = target_lin

    pred = predict_spectrum(model, device, atom_fea, nbr_fea, nbr_fea_idx, channel)
    mae = core_mae(pred, target_model[0])

    n_atoms = int(raw["n_atoms"])
    atom_indices = list(range(n_atoms))
    freq_rows = []
    for freq_idx in freq_indices:
        grad, eps_pred = finite_difference_grad(
            model,
            device,
            raw["positions"],
            raw["cell"],
            raw["atomic_numbers"],
            raw["pbc"],
            atom_init,
            radius,
            max_num_nbr,
            gdf,
            use_log,
            channel,
            freq_idx,
            delta,
            atom_indices,
        )
        per_atom_norm = np.linalg.norm(grad, axis=1)
        freq_rows.append({
            "freq_idx": freq_idx,
            "eps_pred": eps_pred,
            **material_sensitivity_summary(per_atom_norm),
        })

    return {
        "mpid": info["mpid"],
        "formula": info["formula"],
        "n_atoms": n_atoms,
        "opt_gap": info.get("opt_gap"),
        "core_mae": mae,
        "freq_rows": freq_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch CGCNN sensitivity stats (finite diff)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lmdb_path", default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--atom_init", default="atom_init.json")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--freq_low", type=int, default=10)
    parser.add_argument("--freq_mid", type=int, default=100)
    parser.add_argument("--freq_high", type=int, default=400)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.01, help="Perturbation size (Å)")
    parser.add_argument("--out_dir", default="structural_sensitivity/stats_cgcnn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, model_type = load_checkpoint_model(args.model_path, device)
    if model_type != "CGCNN":
        raise SystemExit("This script requires a CGCNN / quasi-MACE checkpoint.")

    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))
    model_name = ckpt.get("model_name", os.path.basename(args.model_path))
    freq_indices = [args.freq_low, args.freq_mid, args.freq_high]

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    radius = dataset.radius
    max_num_nbr = dataset.max_num_nbr
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)
    atom_init = load_atom_init(args.atom_init)

    indices = get_split_indices(ckpt, len(dataset), args.split)
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    os.makedirs(args.out_dir, exist_ok=True)
    records = []

    print("=" * 60)
    print(f"Model : {args.model_path} ({model_name})")
    print(f"Split : {args.split}  n={len(indices)}")
    print(f"Method: CGCNN finite differences  delta={args.delta} A")
    print(f"Freq indices (low/mid/high): {freq_indices}")
    print(f"xi (eV): {[float(xi[i]) for i in freq_indices]}")
    print("=" * 60)

    for n_done, idx in enumerate(indices, 1):
        info = dataset.get_sample_info(idx)
        raw = load_sample_arrays(dataset, idx)
        n_atoms = int(raw["n_atoms"])
        n_fwd = 1 + len(freq_indices) * (6 * n_atoms)
        print(f"  [{n_done}/{len(indices)}] {info['mpid']}  n_atoms={n_atoms}  ~{n_fwd} forwards")
        rec = analyze_crystal(
            model, device, raw, info, atom_init, radius, max_num_nbr, gdf,
            use_log, args.channel, freq_indices, args.delta, target_components,
        )
        records.append(rec)

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
