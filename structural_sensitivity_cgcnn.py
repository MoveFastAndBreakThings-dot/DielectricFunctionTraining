"""
Structural sensitivity of ε(iξ) w.r.t. atomic positions via finite differences (CGCNN).

For unofficial FrequencyCGCNN / FrequencyMACE checkpoints. Rebuilds the distance
graph from perturbed coordinates (pymatgen) at each evaluation.

Usage:
    python structural_sensitivity_cgcnn.py \\
        --model_path models/mace_kkr_v2.pt \\
        --mpid mp-149 \\
        --freq_idx 10 \\
        --delta 0.01
"""

import argparse
import os

import numpy as np
import torch
from pymatgen.core import Element, Lattice, Structure

from lmdb_dataset import LMDBCrystalDataset, collate_crystal_graphs
from preprocess_dataset import GaussianDistance, get_atom_features, get_neighbor_features, load_atom_init
from sensitivity_utils import (
    find_mpid_index,
    load_checkpoint_model,
    load_sample_arrays,
    parse_atom_indices,
    plot_sensitivity,
    save_sensitivity_npz,
)


def structure_from_arrays(positions, cell, atomic_numbers, pbc):
    lattice = Lattice(cell, pbc=tuple(bool(x) for x in pbc))
    species = [Element.from_Z(int(z)) for z in atomic_numbers]
    return Structure(lattice, species, positions, coords_are_cartesian=True)


def cgcnn_graph_from_structure(structure, atom_init, radius, max_num_nbr, gdf):
    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(structure, radius, max_num_nbr, gdf)
    return atom_fea, nbr_fea, nbr_fea_idx


@torch.no_grad()
def predict_eps(model, device, atom_fea, nbr_fea, nbr_fea_idx, use_log, channel, freq_idx):
    inputs = collate_crystal_graphs([(atom_fea, nbr_fea, nbr_fea_idx)])
    af, nf, nfi, cai = inputs
    af = af.to(device)
    nf = nf.to(device)
    nfi = nfi.to(device)
    cai = [x.to(device) for x in cai]
    out = model(af, nf, nfi, cai)[0, channel]
    if use_log:
        out = torch.exp(out)
    return float(out[freq_idx].cpu())


def finite_difference_grad(
    model,
    device,
    positions,
    cell,
    atomic_numbers,
    pbc,
    atom_init,
    radius,
    max_num_nbr,
    gdf,
    use_log,
    channel,
    freq_idx,
    delta,
    atom_indices,
):
    positions = np.array(positions, dtype=np.float64, copy=True)
    n_atoms = positions.shape[0]
    grad = np.zeros((n_atoms, 3), dtype=np.float64)

    structure = structure_from_arrays(positions, cell, atomic_numbers, pbc)
    atom_fea, nbr_fea, nbr_fea_idx = cgcnn_graph_from_structure(
        structure, atom_init, radius, max_num_nbr, gdf
    )
    base = predict_eps(
        model, device, atom_fea, nbr_fea, nbr_fea_idx, use_log, channel, freq_idx
    )

    for atom in atom_indices:
        for dim in range(3):
            pos_plus = positions.copy()
            pos_minus = positions.copy()
            pos_plus[atom, dim] += delta
            pos_minus[atom, dim] -= delta

            s_plus = structure_from_arrays(pos_plus, cell, atomic_numbers, pbc)
            af_p, nf_p, idx_p = cgcnn_graph_from_structure(
                s_plus, atom_init, radius, max_num_nbr, gdf
            )
            val_plus = predict_eps(
                model, device, af_p, nf_p, idx_p, use_log, channel, freq_idx
            )

            s_minus = structure_from_arrays(pos_minus, cell, atomic_numbers, pbc)
            af_m, nf_m, idx_m = cgcnn_graph_from_structure(
                s_minus, atom_init, radius, max_num_nbr, gdf
            )
            val_minus = predict_eps(
                model, device, af_m, nf_m, idx_m, use_log, channel, freq_idx
            )

            grad[atom, dim] = (val_plus - val_minus) / (2.0 * delta)

    return grad, base


def main():
    parser = argparse.ArgumentParser(
        description="Finite-difference structural sensitivity for CGCNN models"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lmdb_path", default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--atom_init", default="atom_init.json")
    parser.add_argument("--mpid", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--freq_idx", type=int, default=10)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.01, help="Cartesian perturbation (Å)")
    parser.add_argument(
        "--atoms",
        default="all",
        help='Comma-separated atom indices or "all" (default: all atoms)',
    )
    parser.add_argument("--out_dir", default="structural_sensitivity/cgcnn")
    args = parser.parse_args()

    if (args.mpid is None) == (args.index is None):
        raise SystemExit("Provide exactly one of --mpid or --index")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model, model_type = load_checkpoint_model(args.model_path, device)
    if model_type != "CGCNN":
        raise SystemExit(
            f"Checkpoint is model_type={model_type!r}. "
            "Use structural_sensitivity_mace.py for official MACE checkpoints."
        )

    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))

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

    if args.mpid is not None:
        idx = find_mpid_index(args.lmdb_path, dataset, args.mpid)
        if idx is None:
            raise SystemExit(f"MPID {args.mpid!r} not found in {args.lmdb_path}")
    else:
        idx = args.index

    info = dataset.get_sample_info(idx)
    raw = load_sample_arrays(dataset, idx)
    if "positions" not in raw:
        raise SystemExit(
            "LMDB sample lacks positions/cell/atomic_numbers. Re-run preprocess_dataset.py."
        )

    atom_indices = parse_atom_indices(args.atoms, raw["n_atoms"])
    n_evals = len(atom_indices) * 3 * 2
    print(f"Finite-difference evals: {n_evals} forward passes")

    grad, base = finite_difference_grad(
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
        args.channel,
        args.freq_idx,
        args.delta,
        atom_indices,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    safe_mpid = info["mpid"].replace("/", "-")
    base_name = f"{safe_mpid}_freq{args.freq_idx}"
    npz_path = os.path.join(args.out_dir, f"{base_name}_grad.npz")
    plot_path = os.path.join(args.out_dir, f"{base_name}_sensitivity.png")

    save_sensitivity_npz(
        npz_path,
        info["mpid"],
        info["formula"],
        xi,
        args.freq_idx,
        grad.astype(np.float32),
        base,
    )
    per_atom = np.linalg.norm(grad, axis=1)
    plot_sensitivity(
        plot_path,
        info["mpid"],
        info["formula"],
        per_atom,
        args.freq_idx,
        xi[args.freq_idx],
        f"CGCNN finite diff (delta={args.delta} A)",
    )

    computed = np.array(atom_indices, dtype=int)
    top = computed[np.argsort(per_atom[computed])[::-1][:10]]
    print(f"Model : {args.model_path} ({model_type})")
    print(f"Sample: {info['mpid']} {info['formula']} (index {idx}, {raw['n_atoms']} atoms)")
    print(f"Target: ε(iξ) at idx {args.freq_idx}, xi={xi[args.freq_idx]:.4g} eV -> {base:.6f}")
    print(f"Saved : {npz_path}")
    print(f"Plot  : {plot_path}")
    print("Top atoms by |grad| (among computed indices):")
    for rank, atom in enumerate(top, 1):
        print(f"  {rank:2d}. atom {atom:4d}  |grad|={per_atom[atom]:.6e}")

    dataset.close()


if __name__ == "__main__":
    main()
