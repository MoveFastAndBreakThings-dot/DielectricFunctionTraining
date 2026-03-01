"""
Test script: run dielectric model on benzene (1 molecule), graphene, and diamond crystal.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.build import molecule, bulk
from pymatgen.io.ase import AseAtomsAdaptor

from preprocess_dataset import (
    load_atom_init,
    get_atom_features,
    get_neighbor_features,
    GaussianDistance,
)
from model_loader import load_model
from lmdb_dataset import collate_pool


def graphene(formula="C2", a=2.460, thickness=0.0, size=(1, 1, 1), vacuum=None):
    """Create a graphene monolayer structure.

    Parameters
    ----------
    formula : str
        Chemical formula for the unit cell (default 'C2' = 2 C atoms).
    a : float
        In-plane hexagonal lattice constant (Å). C–C ≈ 1.42 Å gives a ≈ 2.46.
    thickness : float
        Layer thickness; used for buckled structures like silicene (0 = flat).
    size : tuple
        Repeat (n1, n2, n3) of the unit cell.
    vacuum : float, optional
        If set, add this much vacuum (Å) in the z direction and center the layer.
    """
    cell = [[a, 0, 0], [-a / 2, a * 3**0.5 / 2, 0], [0, 0, 0]]
    basis = [[0, 0, -0.5 * thickness], [2 / 3, 1 / 3, 0.5 * thickness]]
    atoms = Atoms(formula, cell=cell, pbc=(1, 1, 0))
    atoms.set_scaled_positions(basis)
    if vacuum is not None:
        atoms.center(vacuum, axis=2)
    atoms = atoms.repeat(size)
    return atoms


def ase_to_pymatgen(ase_atoms, cell_size=None):
    """ASE Atoms -> pymatgen Structure. If cell_size given, set cubic cell and PBC."""
    adaptor = AseAtomsAdaptor()
    if cell_size is not None:
        ase_atoms = ase_atoms.copy()
        ase_atoms.set_cell([cell_size, cell_size, cell_size])
        ase_atoms.set_pbc([True, True, True])
        ase_atoms.center()
    return adaptor.get_structure(ase_atoms)


def run_model_on_structure(structure, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr):
    """Run model on a pymatgen Structure; return prediction array and xi."""
    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(structure, radius, max_num_nbr, gdf)

    n_frequencies = checkpoint.get("n_frequencies", checkpoint.get("n_xi_points", 500))
    n_output_channels = checkpoint.get("n_output_channels", 1)
    max_idx = len(atom_fea) - 1
    if nbr_fea_idx.max() > max_idx:
        nbr_fea_idx = np.clip(nbr_fea_idx, 0, max_idx)

    dummy_target = np.zeros((n_output_channels, n_frequencies), dtype=np.float32)
    dataset_item = ((atom_fea, nbr_fea, nbr_fea_idx), dummy_target, 0)
    inputs, _, _ = collate_pool([dataset_item])
    atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx = inputs

    atom_fea_t = atom_fea_t.to(device)
    nbr_fea_t = nbr_fea_t.to(device)
    nbr_fea_idx_t = nbr_fea_idx_t.to(device)
    crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

    model.eval()
    with torch.no_grad():
        pred = model(atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx)

    use_log = checkpoint.get("use_log", False)
    pred_real = torch.exp(pred) if use_log else pred
    prediction = pred_real[0, 0].cpu().numpy()

    if "kkr_xi_array" in checkpoint or "xi_array" in checkpoint:
        xi = checkpoint.get("kkr_xi_array", checkpoint.get("xi_array"))
    else:
        xi = np.logspace(np.log10(0.01), np.log10(100.0), len(prediction))

    return prediction, xi


def save_plot(xi, predictions, title, filename, model_type=""):
    """Save ε(iξ) plot. predictions: array."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, predictions, "b-", linewidth=2, label="Predicted ε(iξ)")
    ax.set_xlabel("ξ (eV)", fontsize=12)
    ax.set_ylabel("ε(iξ)", fontsize=12)
    ax.set_title(f"{title}" + (f" – {model_type}" if model_type else ""), fontsize=14)
    ax.set_xlim(0, 40)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {filename}")


def main():
    model_path = "models/mace_kkr_v2.pt"
    if not os.path.exists(model_path):
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
            if model_files:
                model_path = os.path.join(models_dir, model_files[0])
            else:
                print("No .pt model in models/")
                return
        else:
            print("models/ not found")
            return

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_model(checkpoint, device)
    atom_init = load_atom_init("atom_init.json")
    radius = checkpoint.get("radius", 8.0)
    max_num_nbr = checkpoint.get("max_num_nbr", 12)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)

    print("=" * 60)
    print("Benzene + graphene + diamond + silicon crystal test")
    print("=" * 60)

    # Benzene (1 molecule in large cell)
    print("\n--- Benzene C6H6 ---")
    try:
        benzene = molecule("C6H6")
        struct_bz = ase_to_pymatgen(benzene, cell_size=50.0)
        pred_bz, xi_bz = run_model_on_structure(
            struct_bz, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
        )
        print(f"  ε(0) ≈ {pred_bz[0]:.4f}, mean ε(iξ) = {pred_bz.mean():.4f}")
        save_plot(xi_bz, pred_bz, "Benzene C6H6 ε(iξ)", "molecule_test_benzene.png", model_type)
    except Exception as e:
        print(f"  Failed: {e}")

    # Graphene (2D monolayer)
    print("\n--- Graphene ---")
    try:
        graphene_atoms = graphene(a=2.46, vacuum=15.0)
        graphene_atoms.set_pbc([True, True, True])  # 3D PBC for neighbor search
        struct_gr = ase_to_pymatgen(graphene_atoms, cell_size=None)
        pred_gr, xi_gr = run_model_on_structure(
            struct_gr, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
        )
        print(f"  ε(0) ≈ {pred_gr[0]:.4f}, mean ε(iξ) = {pred_gr.mean():.4f}")
        save_plot(xi_gr, pred_gr, "Graphene ε(iξ)", "molecule_test_graphene.png", model_type)
    except Exception as e:
        print(f"  Failed: {e}")

    # Diamond crystal
    print("\n--- C diamond (bulk) ---")
    try:
        c_bulk = bulk("C", "diamond", a=3.57)
        struct_bulk = ase_to_pymatgen(c_bulk, cell_size=None)
        pred_bulk, xi_bulk = run_model_on_structure(
            struct_bulk, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
        )
        print(f"  ε(0) ≈ {pred_bulk[0]:.4f}, mean ε(iξ) = {pred_bulk.mean():.4f}")
        save_plot(xi_bulk, pred_bulk, "C diamond ε(iξ)", "molecule_test_c_diamond.png", model_type)
    except Exception as e:
        print(f"  Failed: {e}")

    # Silicon (diamond cubic)
    print("\n--- Si silicon (bulk) ---")
    try:
        si_bulk = bulk("Si", "diamond", a=5.43)
        struct_si = ase_to_pymatgen(si_bulk, cell_size=None)
        pred_si, xi_si = run_model_on_structure(
            struct_si, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
        )
        print(f"  ε(0) ≈ {pred_si[0]:.4f}, mean ε(iξ) = {pred_si.mean():.4f}")
        save_plot(xi_si, pred_si, "Si silicon ε(iξ)", "molecule_test_silicon.png", model_type)
    except Exception as e:
        print(f"  Failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
