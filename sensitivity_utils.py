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


def material_sensitivity_summary(per_atom_norm):
    per_atom_norm = np.asarray(per_atom_norm, dtype=np.float64)
    total = per_atom_norm.sum()
    return {
        "max_sens": float(per_atom_norm.max()),
        "mean_sens": float(per_atom_norm.mean()),
        "p90_sens": float(np.percentile(per_atom_norm, 90)),
        "concentration": float(per_atom_norm.max() / (total + 1e-30)),
    }


PERCENTILES = list(range(5, 100, 5))


def print_percentile_table(name, values, percentiles=None, unit=""):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if percentiles is None:
        percentiles = PERCENTILES
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


def save_stats_csv(path, records, xi):
    fieldnames = [
        "mpid", "formula", "n_atoms", "opt_gap", "core_mae",
        "freq_idx", "xi_ev", "eps_pred",
        "max_sens", "mean_sens", "p90_sens", "concentration",
    ]
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            for row in rec["freq_rows"]:
                writer.writerow({
                    "mpid": rec["mpid"],
                    "formula": rec["formula"],
                    "n_atoms": rec["n_atoms"],
                    "opt_gap": rec["opt_gap"],
                    "core_mae": rec["core_mae"],
                    "freq_idx": row["freq_idx"],
                    "xi_ev": float(xi[row["freq_idx"]]),
                    "eps_pred": row["eps_pred"],
                    "max_sens": row["max_sens"],
                    "mean_sens": row["mean_sens"],
                    "p90_sens": row["p90_sens"],
                    "concentration": row["concentration"],
                })


def plot_stats(out_dir, records, freq_indices, xi, model_name):
    import matplotlib.pyplot as plt

    labels = {freq_indices[0]: "low", freq_indices[1]: "mid", freq_indices[2]: "high"}
    colors = {"low": "C0", "mid": "C1", "high": "C2"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ("max_sens", "mean_sens", "concentration")):
        for freq_idx in freq_indices:
            vals = [
                row[metric]
                for rec in records
                for row in rec["freq_rows"]
                if row["freq_idx"] == freq_idx
            ]
            ax.hist(
                vals, bins=30, alpha=0.5,
                label=f"{labels[freq_idx]} xi={xi[freq_idx]:.3g}",
                color=colors[labels[freq_idx]],
            )
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle(f"Sensitivity distributions | {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, freq_idx in zip(axes, freq_indices):
        maes, max_sens = [], []
        for rec in records:
            maes.append(rec["core_mae"])
            for row in rec["freq_rows"]:
                if row["freq_idx"] == freq_idx:
                    max_sens.append(row["max_sens"])
        ax.scatter(maes, max_sens, s=12, alpha=0.5, color=colors[labels[freq_idx]])
        if len(maes) > 1:
            r = np.corrcoef(maes, max_sens)[0, 1]
            ax.set_title(f"{labels[freq_idx]} xi={xi[freq_idx]:.3g}  r={r:.3f}")
        else:
            ax.set_title(f"{labels[freq_idx]} xi={xi[freq_idx]:.3g}")
        ax.set_xlabel("core MAE")
        ax.set_ylabel("max_sens")
    fig.suptitle(f"max_sens vs core MAE | {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sensitivity_vs_mae.png"), dpi=150, bbox_inches="tight")
    plt.close()


def print_sensitivity_correlations(records, freq_indices, xi):
    print("\n--- Sensitivity vs core epsilon MAE ---")
    maes = np.array([rec["core_mae"] for rec in records], dtype=np.float64)
    for freq_idx in freq_indices:
        max_sens = np.array([
            row["max_sens"]
            for rec in records
            for row in rec["freq_rows"]
            if row["freq_idx"] == freq_idx
        ], dtype=np.float64)
        conc = np.array([
            row["concentration"]
            for rec in records
            for row in rec["freq_rows"]
            if row["freq_idx"] == freq_idx
        ], dtype=np.float64)
        if len(maes) > 1:
            r_max = float(np.corrcoef(maes, max_sens)[0, 1])
            r_conc = float(np.corrcoef(maes, conc)[0, 1])
        else:
            r_max = r_conc = float("nan")
        print(f"  freq_idx={freq_idx}  xi={xi[freq_idx]:.4g} eV")
        print(f"    corr(core_mae, max_sens)       = {r_max:.4f}")
        print(f"    corr(core_mae, concentration)  = {r_conc:.4f}")
