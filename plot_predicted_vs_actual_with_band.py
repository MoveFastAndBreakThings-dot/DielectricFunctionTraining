"""
Plot predicted ε(iξ) ± 2.5·σ(ξ) band vs actual ε(iξ) for test-set crystals.

σ(ξ; b, c, α) = b / (ξ + c)^α  (per-crystal predicted parameters)

Outputs (in plot_band/):
  - One PDF with a grid of individual crystal panels (n_examples crystals)
  - A summary panel showing median prediction, median band, and all actual curves

Usage:
    python3 plot_predicted_vs_actual_with_band.py
    python3 plot_predicted_vs_actual_with_band.py --n_examples 12 --k_sigma 2.5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from mace_model_error_params import FrequencyMACEWithABC

OUT_DIR = "plot_band"
os.makedirs(OUT_DIR, exist_ok=True)


def load_model_from_checkpoint(checkpoint, device):
    model = FrequencyMACEWithABC(
        orig_atom_fea_len=checkpoint["orig_atom_fea_len"],
        nbr_fea_len=checkpoint["nbr_fea_len"],
        atom_fea_len=checkpoint.get("atom_fea_len", 128),
        n_layers=checkpoint.get("n_layers", 4),
        h_fea_len=checkpoint.get("h_fea_len", 256),
        n_h=checkpoint.get("n_h", 1),
        n_frequencies=checkpoint["n_frequencies"],
        n_output_channels=checkpoint["n_output_channels"],
        dropout=checkpoint.get("dropout", 0.1),
        abc_hidden=checkpoint.get("abc_hidden", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def sigma_curve(xi, b, c, alpha):
    """σ(ξ) = b / (ξ + c)^α"""
    return b / (xi + c) ** alpha


def plot_individual_grid(samples, xi, k_sigma, alpha, save_path, log_x=True):
    """Grid of individual crystal panels."""
    n = len(samples)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, sample in enumerate(samples):
        ax = axes[idx // ncols][idx % ncols]
        eps_pred = sample["eps_pred"]
        eps_true = sample["eps_true"]
        b = sample["b"]
        c = sample["c"]
        formula = sample["formula"]

        sigma = sigma_curve(xi, b, c, alpha)
        upper = eps_pred + k_sigma * sigma
        lower = np.maximum(eps_pred - k_sigma * sigma, 1e-6)

        x = xi
        ax.fill_between(x, lower, upper, alpha=0.25, color="C0", label=f"±{k_sigma}σ")
        ax.plot(x, eps_pred, color="C0", lw=1.5, label="Predicted")
        ax.plot(x, eps_true, color="C1", lw=1.5, ls="--", label="Actual")

        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel("ε(iξ)")
        ax.set_title(formula, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0.9)

    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"Predicted ε(iξ) ± {k_sigma}σ vs Actual  (α={alpha:.2f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_summary(samples, xi, k_sigma, alpha, save_path, log_x=True):
    """
    Summary panel:
    - grey lines: all individual actual curves
    - blue line: median predicted ε across crystals
    - blue band: median predicted ε ± median σ*k_sigma across crystals
    - orange dashed: median actual ε
    """
    eps_pred_all = np.stack([s["eps_pred"] for s in samples])   # (N, F)
    eps_true_all = np.stack([s["eps_true"] for s in samples])   # (N, F)
    b_arr = np.array([s["b"] for s in samples])
    c_arr = np.array([s["c"] for s in samples])

    sigma_all = np.stack([sigma_curve(xi, b, c, alpha) for b, c in zip(b_arr, c_arr)])  # (N, F)

    med_pred = np.median(eps_pred_all, axis=0)
    med_true = np.median(eps_true_all, axis=0)
    med_sigma = np.median(sigma_all, axis=0)
    upper = med_pred + k_sigma * med_sigma
    lower = np.maximum(med_pred - k_sigma * med_sigma, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, use_log in zip(axes, [True, False]):
        # individual actual curves
        for eps_t in eps_true_all:
            ax.plot(xi, eps_t, color="grey", alpha=0.08, lw=0.7)

        ax.fill_between(xi, lower, upper, alpha=0.3, color="C0", label=f"Median pred ± {k_sigma}·median σ")
        ax.plot(xi, med_pred, color="C0", lw=2, label="Median predicted")
        ax.plot(xi, med_true, color="C1", lw=2, ls="--", label="Median actual")

        if use_log:
            ax.set_xscale("log")
            ax.set_title("Log ξ axis")
        else:
            ax.set_title("Linear ξ axis")

        ax.set_xlabel("ξ (eV)")
        ax.set_ylabel("ε(iξ)")
        ax.set_ylim(bottom=0.9)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Summary over {len(samples)} test crystals  |  σ(ξ) = b/(ξ+c)^α, α={alpha:.2f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_individual(samples, xi, k_sigma, alpha, out_dir):
    """Save one PNG per crystal to out_dir/{mpid}_{formula}.png"""
    os.makedirs(out_dir, exist_ok=True)
    for i, s in enumerate(samples):
        sigma = sigma_curve(xi, s["b"], s["c"], alpha)
        upper = s["eps_pred"] + k_sigma * sigma
        lower = np.maximum(s["eps_pred"] - k_sigma * sigma, 1e-6)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, use_log in zip(axes, [True, False]):
            ax.fill_between(xi, lower, upper, alpha=0.25, color="C0",
                            label=f"pred ± {k_sigma}σ")
            ax.plot(xi, s["eps_pred"], color="C0", lw=1.5, label="Predicted")
            ax.plot(xi, s["eps_true"], color="C1", lw=1.5, ls="--", label="Actual")
            if use_log:
                ax.set_xscale("log")
                ax.set_title("Log ξ")
            else:
                ax.set_title("Linear ξ")
            ax.set_xlabel("ξ (eV)")
            ax.set_ylabel("ε(iξ)")
            ax.set_ylim(bottom=0.9)
            ax.legend(fontsize=8)

        mae = float(np.mean(np.abs(s["eps_pred"] - s["eps_true"])))
        mpid = s.get("mpid", f"idx{i}")
        formula = s.get("formula", "")
        fig.suptitle(
            f"{mpid}  {formula}  |  MAE={mae:.4f}  b={s['b']:.3f}  c={s['c']:.3f} eV",
            fontsize=10,
        )
        plt.tight_layout()

        # sanitise filename
        safe = f"{mpid}_{formula}".replace("/", "-").replace(" ", "_")
        fname = os.path.join(out_dir, f"{safe}.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)} crystals saved...")

    print(f"  Saved {len(samples)} individual plots to {out_dir}/")


def plot_residual_vs_sigma(samples, xi, alpha, save_path):
    """
    Scatter of actual |ε_pred - ε_true| vs predicted σ(ξ) across all (crystal, ξ) pairs.
    If σ is a good error model, points should scatter along the diagonal.
    """
    residuals, sigmas = [], []
    for s in samples:
        res = np.abs(s["eps_pred"] - s["eps_true"])
        sig = sigma_curve(xi, s["b"], s["c"], alpha)
        residuals.append(res)
        sigmas.append(sig)

    residuals = np.concatenate(residuals)
    sigmas = np.concatenate(sigmas)

    # subsample for speed
    rng = np.random.default_rng(0)
    idx = rng.choice(len(residuals), size=min(10000, len(residuals)), replace=False)
    residuals = residuals[idx]
    sigmas = sigmas[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sigmas, residuals, s=3, alpha=0.2, color="C0")
    lim = max(sigmas.max(), residuals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect)")
    ax.set_xlabel("Predicted σ(ξ)")
    ax.set_ylabel("|ε_pred − ε_true|")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Predicted σ vs actual residual")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/mace_kkr_with_abc_v2.pt")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--n_examples", type=int, default=16,
                        help="Number of individual crystal panels to plot")
    parser.add_argument("--k_sigma", type=float, default=2.5,
                        help="Band width in units of σ")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--all_individual", action="store_true",
                        help="Save one PNG per crystal into plot_band/all_crystals/")
    args = parser.parse_args()

    checkpoint = torch.load(args.model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_checkpoint(checkpoint, device)

    use_log = checkpoint.get("use_log", False)
    alpha = checkpoint.get("alpha", 3.0)
    target_components = checkpoint.get("target_components", ("xx",))

    print(f"Model: {checkpoint.get('model_name', args.model_path)}")
    print(f"  use_log={use_log}, alpha={alpha}, k_sigma={args.k_sigma}")

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True)
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)

    test_idx = checkpoint.get("test_idx")
    if test_idx is None:
        np.random.seed(42)
        n = len(dataset)
        idx = np.random.permutation(n)
        train_sz, val_sz = int(n * 0.8), int(n * 0.1)
        test_idx = idx[train_sz + val_sz:].tolist()

    test_ds = Subset(dataset, test_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate_pool)

    # collect everything in one pass
    samples = []
    ds_idx = 0
    with torch.no_grad():
        for inputs, target, _ in test_loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [x.to(device) for x in crystal_atom_idx]
            target = target.to(device)

            eps_out, bc_out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

            if use_log:
                eps_real = torch.exp(eps_out).cpu().numpy()
            else:
                eps_real = eps_out.cpu().numpy()

            eps_true_np = target.cpu().numpy()
            bc_cpu = bc_out.cpu().numpy()

            for j in range(eps_real.shape[0]):
                global_idx = test_idx[ds_idx]
                info = dataset.get_sample_info(global_idx)
                samples.append({
                    "eps_pred": eps_real[j, 0, :],
                    "eps_true": eps_true_np[j, 0, :],
                    "b": float(bc_cpu[j, 0]),
                    "c": float(bc_cpu[j, 1]),
                    "formula": info["formula"],
                    "mpid": info["mpid"],
                })
                ds_idx += 1

    dataset.close()
    print(f"Collected {len(samples)} test crystals. Generating plots...")

    # pick n_examples: spread across sorted-by-loss to get variety
    # sort by MAE so we pick low/mid/high error examples
    for s in samples:
        s["mae"] = float(np.mean(np.abs(s["eps_pred"] - s["eps_true"])))

    samples_sorted = sorted(samples, key=lambda s: s["mae"])
    n = args.n_examples
    if len(samples_sorted) >= n:
        step = max(1, len(samples_sorted) // n)
        examples = [samples_sorted[i * step] for i in range(n)]
    else:
        examples = samples_sorted

    plot_individual_grid(
        examples, xi, args.k_sigma, alpha,
        save_path=os.path.join(OUT_DIR, "individual_panels.png"),
    )
    plot_summary(
        samples, xi, args.k_sigma, alpha,
        save_path=os.path.join(OUT_DIR, "summary.png"),
    )
    plot_residual_vs_sigma(
        samples, xi, alpha,
        save_path=os.path.join(OUT_DIR, "residual_vs_sigma.png"),
    )

    if args.all_individual:
        print(f"\nSaving one plot per crystal ({len(samples)} total)...")
        plot_all_individual(
            samples, xi, args.k_sigma, alpha,
            out_dir=os.path.join(OUT_DIR, "all_crystals"),
        )

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
