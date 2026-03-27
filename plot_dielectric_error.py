"""
Compute absolute error in dielectric function |ε_pred - ε_actual| at each frequency
point for every test material, and plot per material plus a summary.

More stable alternatives to exponential for modeling error(ξ):
  - Power-law (stable, 2–3 params):  E(ξ) = a + b/(ξ + c)^α   with α>0, c>0 small
  - Lorentzian (stable, 3 params):  E(ξ) = a + b/(1 + (ξ/ξ_0)^2)
  - Simple inverse:                 E(ξ) = a + b/(ξ + ξ_0)     [2 params, very stable]
  - Log-linear fit: fit log(E) vs ξ; E = exp(c0 + c1*ξ)       [avoids scale issues]
Tips: use ξ in eV (not index); bound a,b,ξ_0 > 0; or fit log(error) to reduce sensitivity to large values.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model

OUT_DIR = "dielectric_error_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    model_path = "models/mace_kkr_v2.pt"
    lmdb_path = "processed_data/crystal_graphs.lmdb"
    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    use_log = checkpoint.get("use_log", False)

    dataset = LMDBCrystalDataset(lmdb_path, use_kkr=True)
    xi = np.array(dataset.kkr_xi_array)
    n_freq = len(xi)

    if "test_idx" not in checkpoint:
        np.random.seed(42)
        n = len(dataset)
        idx = np.random.permutation(n)
        train_sz = int(n * 0.8)
        val_sz = int(n * 0.1)
        test_indices = idx[train_sz + val_sz :].tolist()
    else:
        test_indices = checkpoint["test_idx"]

    all_errors = []
    n_test = len(test_indices)
    print(f"Computing absolute dielectric error for {n_test} test materials...")

    for i, idx in enumerate(test_indices):
        idx = int(idx)
        info = dataset.get_sample_info(idx)
        inputs, tgt, _ = collate_pool([dataset[idx]])
        atom_fea_t = inputs[0].to(device)
        nbr_fea_t = inputs[1].to(device)
        nbr_fea_idx_t = inputs[2].to(device)
        crystal_atom_idx = [x.to(device) for x in inputs[3]]
        with torch.no_grad():
            pred = model(atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx)
        pred_real = torch.exp(pred) if use_log else pred
        actual = tgt[0, 0].numpy()
        predicted = pred_real[0, 0].cpu().numpy()
        abs_error = np.abs(predicted - actual)
        all_errors.append(abs_error)

        safe_name = info["mpid"].replace("/", "_")[:40]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot(xi, abs_error, "b-", lw=1)
        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel("|ε_pred − ε_actual|")
        ax.set_title(f"{info['formula']} — {info['mpid']}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{i:04d}_{safe_name}.png"), dpi=120)
        plt.close()

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{n_test}] {info['mpid']} {info['formula']}")

    dataset.close()

    all_errors = np.array(all_errors)
    mean_err = np.mean(all_errors, axis=0)
    std_err = np.std(all_errors, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.fill_between(xi, np.maximum(mean_err - std_err, 0), mean_err + std_err, alpha=0.3)
    ax.plot(xi, mean_err, "k-", lw=1.5, label="Mean |ε_pred − ε_actual|")
    ax.set_xlabel("ξ (eV)")
    ax.set_ylabel("Absolute error in ε(iξ)")
    ax.set_title(f"Test set: mean ± std over {n_test} materials")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "summary_mean_abs_error.png"), dpi=150)
    plt.close()

    print(f"Per-material plots saved to {OUT_DIR}/")
    print(f"Summary (mean ± std) saved to {OUT_DIR}/summary_mean_abs_error.png")


if __name__ == "__main__":
    main()