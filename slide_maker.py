"""
Create 1 slide with 6 panels: 2 best + 2 worst from test set (by loss),
and 2 randomly chosen ASE-built structures. Shows predictions vs actual
for test materials and prediction-only for ASE (easy to use on new structures).
"""

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from ase.build import molecule, bulk

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model
from losses import FrequencyWeightedMSELoss
from preprocess_dataset import load_atom_init, get_atom_features, get_neighbor_features, GaussianDistance
from molecule_test import ase_to_pymatgen, run_model_on_structure, graphene

# Model and data paths
MODEL_NAME = "cgcnn_kkr_v2"
LMDB_PATH = "processed_data/crystal_graphs.lmdb"
ATOM_INIT_PATH = "atom_init.json"
OUTPUT_PATH = "slide_maker_slide.png"
SEED = 42


def get_test_results(dataset, model, checkpoint, criterion, device, use_log):
    """Run model on test set; return list of {idx, mpid, formula, loss, actual, predicted} sorted by loss."""
    if "test_idx" in checkpoint:
        test_indices = np.array(checkpoint["test_idx"])
    else:
        np.random.seed(SEED)
        n = len(dataset)
        idx = np.random.permutation(n)
        train_sz = int(n * 0.8)
        val_sz = int(n * 0.1)
        test_indices = idx[train_sz + val_sz :]

    results = []
    for idx in test_indices:
        info = dataset.get_sample_info(idx)
        inputs, tgt, _ = collate_pool([dataset[idx]])
        atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx = inputs
        with torch.no_grad():
            pred = model(
                atom_fea_t.to(device),
                nbr_fea_t.to(device),
                nbr_fea_idx_t.to(device),
                [c.to(device) for c in crystal_atom_idx],
            )
        pred_real = torch.exp(pred) if use_log else pred
        loss = criterion(pred_real.to(device), tgt.to(device)).item()
        actual = tgt[0, 0].numpy()
        predicted = pred_real[0, 0].cpu().numpy()
        results.append({
            "idx": idx,
            "mpid": info["mpid"],
            "formula": info["formula"],
            "loss": loss,
            "actual": actual,
            "predicted": predicted,
        })
    results.sort(key=lambda x: x["loss"])
    return results, test_indices


def pick_best_worst(results, n_best=2, n_worst=2, pool_size=10):
    """Randomly pick n_best from top pool_size and n_worst from bottom pool_size."""
    random.seed(SEED)
    np.random.seed(SEED)
    top_pool = results[: min(pool_size, len(results))]
    bot_pool = results[-min(pool_size, len(results)) :][::-1]
    k_best = min(n_best, len(top_pool))
    k_worst = min(n_worst, len(bot_pool))
    best = list(random.sample(top_pool, k_best))
    worst = list(random.sample(bot_pool, k_worst))
    return best, worst


def build_ase_structures():
    """Return list of (label, ase_atoms, cell_size). cell_size=None means use ASE cell."""
    options = [
        ("Benzene C6H6 (ASE)", molecule("C6H6"), 50.0),
        ("C diamond (ASE)", bulk("C", "diamond", a=3.57), None),
        ("Si silicon (ASE)", bulk("Si", "diamond", a=5.43), None),
    ]
    g = graphene(a=2.46, vacuum=15.0)
    g.set_pbc([True, True, True])
    options.append(("Graphene (ASE)", g, None))
    return options


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model_path = f"models/{MODEL_NAME}.pt"
    if not os.path.exists(model_path):
        if os.path.exists("models"):
            pts = [f for f in os.listdir("models") if f.endswith(".pt")]
            if pts:
                model_path = os.path.join("models", pts[0])
                print(f"Using {model_path}")
            else:
                print("No .pt model in models/")
                return
        else:
            print("models/ not found")
            return

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_model(checkpoint, device)
    use_log = checkpoint.get("use_log", False)

    dataset = LMDBCrystalDataset(LMDB_PATH, use_kkr=True)
    xi = dataset.kkr_xi_array
    criterion = FrequencyWeightedMSELoss(dataset.frequency_array).to(device)

    print("Loading test set and computing losses...")
    results, _ = get_test_results(dataset, model, checkpoint, criterion, device, use_log)
    dataset.close()

    best, worst = pick_best_worst(results, n_best=2, n_worst=2, pool_size=15)
    panels_test = [("Best", r) for r in best] + [("Worst", r) for r in worst]

    atom_init = load_atom_init(ATOM_INIT_PATH)
    radius = checkpoint.get("radius", 8.0)
    max_num_nbr = checkpoint.get("max_num_nbr", 12)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)

    ase_list = build_ase_structures()
    chosen = [ase_list[i] for i in np.random.choice(len(ase_list), size=2, replace=False)]
    panels_ase = []
    for label, ase_atoms, cell_size in chosen:
        struct = ase_to_pymatgen(ase_atoms, cell_size)
        pred, xi_ase = run_model_on_structure(
            struct, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
        )
        panels_ase.append((label, pred, xi_ase))

    # Build slide: 2 rows x 3 — row0: Best1, Best2, Worst1; row1: Worst2, ASE1, ASE2
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Dielectric ε(iξ): test set best/worst + ASE-built structures (easy to run on new crystals)", fontsize=12)

    positions_test = [(0, 0), (0, 1), (0, 2), (1, 0)]
    for (row, col), (tag, r) in zip(positions_test, panels_test):
        ax = axes[row, col]
        ax.plot(xi, r["actual"], "b-", lw=2, label="Actual")
        ax.plot(xi, r["predicted"], "r--", lw=1.5, alpha=0.9, label="Predicted")
        ax.set_xscale("log")
        ax.set_xlim(xi.min(), min(40, xi.max()))
        title = f"{tag}: {r['formula']}\n{r['mpid']}  loss={r['loss']:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("ε(iξ)")
        ax.set_xlabel("ξ (eV)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for col, (label, pred, xi_ase) in enumerate(panels_ase):
        ax = axes[1, col + 1]
        ax.plot(xi_ase, pred, "g-", lw=2, label="Predicted")
        ax.set_xscale("log")
        ax.set_xlim(xi_ase.min(), min(40, xi_ase.max()))
        ax.set_title(f"ASE build: {label}", fontsize=9)
        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel("ε(iξ)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Slide saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
