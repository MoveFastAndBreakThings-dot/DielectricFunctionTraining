"""
Structural sensitivity of ε(iξ) w.r.t. atomic positions via autograd (official MACE).

Requires a MACE checkpoint (--model mace) and geometry in LMDB.

Usage:
    python structural_sensitivity_mace.py \\
        --model_path models/official_mace_kkr_v1.pt \\
        --mpid mp-149 \\
        --freq_idx 10
"""

import argparse
import os

import numpy as np
import torch

from lmdb_dataset import LMDBCrystalDataset
from mace_data import structure_arrays_to_atomic_data, z_table_from_atom_init
from mace.tools.torch_geometric.batch import Batch
from sensitivity_utils import (
    find_mpid_index,
    load_checkpoint_model,
    load_sample_arrays,
    plot_sensitivity,
    save_sensitivity_npz,
)


def forward_eps_at_freq(model, batch, use_log, channel, freq_idx, wrt_linear):
    out = model(batch)[0, channel]
    if use_log and wrt_linear:
        return torch.exp(out[freq_idx])
    return out[freq_idx]


def main():
    parser = argparse.ArgumentParser(
        description="Autograd structural sensitivity for official MACE models"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lmdb_path", default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--mpid", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--freq_idx", type=int, default=10)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument(
        "--wrt",
        choices=("linear", "log"),
        default="linear",
        help="Differentiate linear ε or log ε (match --wrt log if model uses --use_log)",
    )
    parser.add_argument("--out_dir", default="structural_sensitivity/mace")
    args = parser.parse_args()

    if (args.mpid is None) == (args.index is None):
        raise SystemExit("Provide exactly one of --mpid or --index")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, model_type = load_checkpoint_model(args.model_path, device)
    if model_type != "MACE":
        raise SystemExit(
            f"Checkpoint is model_type={model_type!r}. "
            "Use structural_sensitivity_cgcnn.py for CGCNN checkpoints."
        )

    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))
    r_max = ckpt.get("mace_r_max", ckpt.get("radius", 8.0))

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        use_mace=True,
        mace_r_max=r_max,
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    if args.mpid is not None:
        idx = find_mpid_index(args.lmdb_path, dataset, args.mpid)
        if idx is None:
            raise SystemExit(f"MPID {args.mpid!r} not found in {args.lmdb_path}")
    else:
        idx = args.index

    info = dataset.get_sample_info(idx)
    raw = load_sample_arrays(dataset, idx)
    z_table = z_table_from_atom_init()

    graph = structure_arrays_to_atomic_data(
        raw["positions"],
        raw["cell"],
        raw["atomic_numbers"],
        raw["pbc"],
        z_table,
        r_max,
    )
    graph.positions = graph.positions.to(device).detach().requires_grad_(True)
    batch = Batch.from_data_list([graph]).to(device)

    wrt_linear = args.wrt == "linear"
    target = forward_eps_at_freq(
        model, batch, use_log, args.channel, args.freq_idx, wrt_linear
    )
    grad = torch.autograd.grad(target, batch.positions, retain_graph=False)[0]
    grad_np = grad.detach().cpu().numpy()
    target_val = float(target.detach().cpu())

    os.makedirs(args.out_dir, exist_ok=True)
    safe_mpid = info["mpid"].replace("/", "-")
    base = f"{safe_mpid}_freq{args.freq_idx}"
    npz_path = os.path.join(args.out_dir, f"{base}_grad.npz")
    plot_path = os.path.join(args.out_dir, f"{base}_sensitivity.png")

    save_sensitivity_npz(
        npz_path,
        info["mpid"],
        info["formula"],
        xi,
        args.freq_idx,
        grad_np,
        target_val,
    )
    plot_sensitivity(
        plot_path,
        info["mpid"],
        info["formula"],
        np.linalg.norm(grad_np, axis=1),
        args.freq_idx,
        xi[args.freq_idx],
        "MACE autograd",
    )

    per_atom = np.linalg.norm(grad_np, axis=1)
    top = np.argsort(per_atom)[::-1][:10]
    print(f"Model : {args.model_path} ({model_type})")
    print(f"Sample: {info['mpid']} {info['formula']} (index {idx}, {raw['n_atoms']} atoms)")
    print(f"Target: ε(iξ) at idx {args.freq_idx}, xi={xi[args.freq_idx]:.4g} eV -> {target_val:.6f}")
    print(f"Saved : {npz_path}")
    print(f"Plot  : {plot_path}")
    print("Top atoms by |grad|:")
    for rank, atom in enumerate(top, 1):
        print(f"  {rank:2d}. atom {atom:4d}  |grad|={per_atom[atom]:.6e}")

    dataset.close()


if __name__ == "__main__":
    main()
