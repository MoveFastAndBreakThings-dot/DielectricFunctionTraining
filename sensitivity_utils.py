"""Shared helpers for structural sensitivity scripts."""

import json
import os
import pickle

import numpy as np
import torch

from model_loader import load_model


def find_mpid_index(lmdb_path, dataset, mpid):
    cache_path = lmdb_path + ".mpid_index.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            mapping = json.load(f)
    else:
        mapping = {}
        for i in range(len(dataset)):
            mapping[dataset.get_sample_info(i)["mpid"]] = i
        with open(cache_path, "w") as f:
            json.dump(mapping, f)
    return mapping.get(mpid)


def load_sample_arrays(dataset, idx):
    key = f"{idx}".encode("utf-8")
    return pickle.loads(dataset._txn.get(key))


def load_checkpoint_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model, model_type = load_model(ckpt, device)
    model.eval()
    return ckpt, model, model_type


def parse_atom_indices(text, n_atoms):
    if text is None or text.lower() == "all":
        return list(range(n_atoms))
    indices = [int(x.strip()) for x in text.split(",") if x.strip()]
    for i in indices:
        if i < 0 or i >= n_atoms:
            raise ValueError(f"Atom index {i} out of range [0, {n_atoms})")
    return indices


def save_sensitivity_npz(out_path, mpid, formula, xi, freq_idx, grad_pos, target_value):
    per_atom = np.linalg.norm(grad_pos, axis=1)
    np.savez(
        out_path,
        mpid=mpid,
        formula=formula,
        xi=xi,
        freq_idx=freq_idx,
        target_value=target_value,
        grad_positions=grad_pos,
        per_atom_norm=per_atom,
    )


def plot_sensitivity(out_path, mpid, formula, per_atom, freq_idx, xi_val, method):
    import matplotlib.pyplot as plt

    order = np.argsort(per_atom)[::-1]
    top = order[: min(30, len(order))]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(top)), per_atom[top], color="C0")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels([str(i) for i in top], rotation=90, fontsize=8)
    ax.set_xlabel("Atom index")
    ax.set_ylabel(r"$|\partial \varepsilon / \partial \mathbf{r}|$")
    ax.set_title(
        f"{mpid} {formula} | {method} | freq idx {freq_idx}, "
        f"xi={xi_val:.4g} eV"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
