"""
Fit error models to |ε_pred - ε_actual| vs ξ. Run on train, val, and test splits;
report and compare mean ± std of params and RMSE across splits.
"""

import os
import json
import numpy as np
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model

OUT_DIR = "error_model_fits"
os.makedirs(OUT_DIR, exist_ok=True)

EPS = 1e-10


def get_split_indices(checkpoint, n_samples, seed=42):
    """Return train_idx, val_idx, test_idx (same logic as training)."""
    if all(k in checkpoint for k in ("train_idx", "val_idx", "test_idx")):
        return checkpoint["train_idx"], checkpoint["val_idx"], checkpoint["test_idx"]
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size : train_size + val_size].tolist()
    test_idx = indices[train_size + val_size :].tolist()
    return train_idx, val_idx, test_idx


def get_errors_for_indices(dataset, model, checkpoint, device, use_log, indices):
    """Return list of dicts with xi, abs_error, formula, mpid for the given indices."""
    xi = np.array(dataset.kkr_xi_array)
    results = []
    for idx in indices:
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
        results.append({"xi": xi, "abs_error": abs_error, "formula": info["formula"], "mpid": info["mpid"]})
    return results


# --- Simple inverse: E(ξ) = a + b/(ξ + ξ0) ---
def model_simple_inverse(xi, a, b, xi0):
    return a + b / (xi + xi0)


def fit_simple_inverse(xi, abs_error):
    def loss(params):
        a, b, xi0 = params
        if a < 0 or b < 0 or xi0 <= 0:
            return 1e20
        pred = model_simple_inverse(xi, a, b, xi0)
        return np.sum((pred - abs_error) ** 2)

    res = minimize(loss, [0.0, 10.0, 0.1], method="L-BFGS-B", bounds=[(0, None), (0, None), (1e-6, None)])
    a, b, xi0 = res.x
    rmse = np.sqrt(res.fun / len(xi))
    return {"a": a, "b": b, "xi0": xi0}, rmse


# --- Simple inverse with a=0: E(ξ) = b/(ξ + ξ0), error→0 as ξ→∞ ---
def model_simple_inverse_a0(xi, b, xi0):
    return b / (xi + xi0)


def fit_simple_inverse_a0(xi, abs_error):
    def loss(params):
        b, xi0 = params
        if b < 0 or xi0 <= 0:
            return 1e20
        pred = model_simple_inverse_a0(xi, b, xi0)
        return np.sum((pred - abs_error) ** 2)

    res = minimize(loss, [10.0, 0.1], method="L-BFGS-B", bounds=[(0, None), (1e-6, None)])
    b, xi0 = res.x
    rmse = np.sqrt(res.fun / len(xi))
    return {"b": b, "xi0": xi0}, rmse


# --- Lorentzian: E(ξ) = a + b/(1 + (ξ/ξ0)^2) ---
def model_lorentzian(xi, a, b, xi0):
    return a + b / (1.0 + (xi / xi0) ** 2)


def fit_lorentzian(xi, abs_error):
    def loss(params):
        a, b, xi0 = params
        if a < 0 or b < 0 or xi0 <= 0:
            return 1e20
        pred = model_lorentzian(xi, a, b, xi0)
        return np.sum((pred - abs_error) ** 2)

    res = minimize(loss, [0.0, 10.0, 1.0], method="L-BFGS-B", bounds=[(0, None), (0, None), (1e-6, None)])
    a, b, xi0 = res.x
    rmse = np.sqrt(res.fun / len(xi))
    return {"a": a, "b": b, "xi0": xi0}, rmse


# --- Power-law: E(ξ) = a + b/(ξ + c)^α ---
def model_power_law(xi, a, b, c, alpha):
    return a + b / ((xi + c) ** alpha)


def fit_power_law(xi, abs_error):
    def loss(params):
        a, b, c, alpha = params
        if a < 0 or b < 0 or c <= 0 or alpha <= 0:
            return 1e20
        pred = model_power_law(xi, a, b, c, alpha)
        return np.sum((pred - abs_error) ** 2)

    res = minimize(loss, [0.0, 10.0, 0.1, 1.0], method="L-BFGS-B", bounds=[(0, None), (0, None), (1e-4, None), (0.1, 5.0)])
    a, b, c, alpha = res.x
    rmse = np.sqrt(res.fun / len(xi))
    return {"a": a, "b": b, "c": c, "alpha": alpha}, rmse


def fit_power_law_alpha_fixed(alpha_fixed):
    """Power-law with fixed α; fit only (a, b, c). More stable than full power-law."""

    def fit_fn(xi, abs_error):
        def loss(params):
            a, b, c = params
            if a < 0 or b < 0 or c <= 0:
                return 1e20
            pred = model_power_law(xi, a, b, c, alpha_fixed)
            return np.sum((pred - abs_error) ** 2)

        res = minimize(loss, [0.0, 10.0, 0.1], method="L-BFGS-B", bounds=[(0, None), (0, None), (1e-4, None)])
        a, b, c = res.x
        rmse = np.sqrt(res.fun / len(xi))
        return {"a": a, "b": b, "c": c}, rmse

    return fit_fn


# --- Exponential: E(ξ) = a + b·exp(-ξ/ξ0) ---
def model_exponential(xi, a, b, xi0):
    return a + b * np.exp(-xi / xi0)


def fit_exponential(xi, abs_error):
    def loss(params):
        a, b, xi0 = params
        if a < 0 or b < 0 or xi0 <= 0:
            return 1e20
        pred = model_exponential(xi, a, b, xi0)
        return np.sum((pred - abs_error) ** 2)

    res = minimize(loss, [0.0, 10.0, 1.0], method="L-BFGS-B", bounds=[(0, None), (0, None), (1e-6, None)])
    a, b, xi0 = res.x
    rmse = np.sqrt(res.fun / len(xi))
    return {"a": a, "b": b, "xi0": xi0}, rmse


# --- Log-linear: log(E) = c0 + c1*ξ => E = exp(c0 + c1*ξ) ---
def model_log_linear(xi, c0, c1):
    return np.exp(c0 + c1 * xi)


def fit_log_linear(xi, abs_error):
    # Fit in log space for stability: min sum (c0 + c1*xi - log(abs_error))^2
    y = np.log(abs_error + EPS)
    # Linear least squares: y = c0 + c1*xi
    A = np.column_stack([np.ones_like(xi), xi])
    c, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    c0, c1 = c[0], c[1]
    pred = model_log_linear(xi, c0, c1)
    rmse = np.sqrt(np.mean((pred - abs_error) ** 2))
    return {"c0": c0, "c1": c1}, rmse


def run_one_model(name, param_names, fit_fn, results):
    """Fit model to each sample; return list of param dicts, list of RMSEs, and summary."""
    if not results:
        return [], [], {"model": name, "n_materials": 0, "rmse_mean": np.nan, "rmse_std": np.nan}
    param_list = []
    rmse_list = []
    for r in results:
        params, rmse = fit_fn(r["xi"], r["abs_error"])
        param_list.append(params)
        rmse_list.append(rmse)
    rmse_arr = np.array(rmse_list)
    summary = {"model": name, "n_materials": len(results), "rmse_mean": float(np.mean(rmse_arr)), "rmse_std": float(np.std(rmse_arr))}
    for pname in param_names:
        vals = [d[pname] for d in param_list]
        summary[f"{pname}_mean"] = float(np.mean(vals))
        summary[f"{pname}_std"] = float(np.std(vals))
    return param_list, rmse_list, summary


def print_and_save_comparison(name, param_names, summary_train, summary_val, summary_test, out_dir):
    """Print train | val | test comparison and save JSON."""
    splits = [("train", summary_train), ("val", summary_val), ("test", summary_test)]
    comparison = {"model": name, "param_names": param_names}
    print(f"\n   {'':12s}  {'train':>18s}  {'val':>18s}  {'test':>18s}")
    print("   " + "-" * 60)
    for pname in param_names:
        row = {}
        parts = []
        for split_name, s in splits:
            if s.get("n_materials", 0) == 0:
                parts.append("  —  ")
                row[split_name] = None
            else:
                m, std = s.get(f"{pname}_mean"), s.get(f"{pname}_std")
                parts.append(f"{m:.4f} ± {std:.4f}")
                row[split_name] = {"mean": m, "std": std}
        comparison[pname] = row
        print(f"   {pname:12s}  {parts[0]:>18s}  {parts[1]:>18s}  {parts[2]:>18s}")
    rmse_row = {}
    parts = []
    for split_name, s in splits:
        if s.get("n_materials", 0) == 0:
            parts.append("  —  ")
            rmse_row[split_name] = None
        else:
            m, std = s["rmse_mean"], s["rmse_std"]
            parts.append(f"{m:.4f} ± {std:.4f}")
            rmse_row[split_name] = {"mean": m, "std": std}
    comparison["rmse"] = rmse_row
    print(f"   {'RMSE':12s}  {parts[0]:>18s}  {parts[1]:>18s}  {parts[2]:>18s}")
    for split_name, s in splits:
        comparison[f"n_{split_name}"] = s.get("n_materials", 0)
    with open(os.path.join(out_dir, f"compare_{name.replace(' ', '_')}.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def plot_param_distributions(name, param_names, param_list, rmse_list, summary):
    """One row per parameter + RMSE."""
    n_params = len(param_names) + 1
    fig, axes = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
    if n_params == 1:
        axes = [axes]
    for i, pname in enumerate(param_names):
        vals = [d[pname] for d in param_list]
        axes[i].hist(vals, bins=30, alpha=0.7, color="C0", edgecolor="k")
        axes[i].axvline(summary[f"{pname}_mean"], color="k", ls="--", label="Mean")
        axes[i].set_xlabel(pname)
        axes[i].set_ylabel("Count")
        axes[i].set_title(f"{pname}")
        axes[i].legend()
    axes[-1].hist(rmse_list, bins=30, alpha=0.7, color="C1", edgecolor="k")
    axes[-1].axvline(summary["rmse_mean"], color="k", ls="--", label="Mean")
    axes[-1].set_xlabel("RMSE")
    axes[-1].set_ylabel("Count")
    axes[-1].set_title("RMSE")
    axes[-1].legend()
    fig.suptitle(f"Model: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"dist_{name.replace(' ', '_')}.png"), dpi=150)
    plt.close()


def main():
    model_path = "models/mace_kkr_v2.pt"
    if not os.path.exists(model_path) and os.path.exists("models"):
        pts = [f for f in os.listdir("models") if f.endswith(".pt")]
        model_path = os.path.join("models", pts[0]) if pts else None
    if not model_path or not os.path.exists(model_path):
        print("No .pt model found in models/")
        return

    lmdb_path = "processed_data/crystal_graphs.lmdb"
    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(checkpoint, device)
    use_log = checkpoint.get("use_log", False)

    dataset = LMDBCrystalDataset(lmdb_path, use_kkr=True)
    n_samples = len(dataset)
    train_idx, val_idx, test_idx = get_split_indices(checkpoint, n_samples)
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print("Loading |ε_pred - ε_actual| for train, val, test...")
    train_results = get_errors_for_indices(dataset, model, checkpoint, device, use_log, train_idx)
    val_results = get_errors_for_indices(dataset, model, checkpoint, device, use_log, val_idx)
    test_results = get_errors_for_indices(dataset, model, checkpoint, device, use_log, test_idx)
    dataset.close()
    print(f"Fitting 8 models on each split (train / val / test).\n")

    models_to_fit = [
        ("simple_inverse", ["a", "b", "xi0"], fit_simple_inverse),
        ("simple_inverse_a0", ["b", "xi0"], fit_simple_inverse_a0),
        ("lorentzian", ["a", "b", "xi0"], fit_lorentzian),
        ("power_law", ["a", "b", "c", "alpha"], fit_power_law),
        ("power_law_alpha1", ["a", "b", "c"], fit_power_law_alpha_fixed(1.0)),
        ("power_law_alpha2", ["a", "b", "c"], fit_power_law_alpha_fixed(2.0)),
        ("log_linear", ["c0", "c1"], fit_log_linear),
        ("exponential", ["a", "b", "xi0"], fit_exponential),
    ]
    all_comparisons = []

    for i, (name, param_names, fit_fn) in enumerate(models_to_fit):
        print("=" * 60)
        print(f"{i + 1}. {name}")
        print("=" * 60)
        _, _, summary_train = run_one_model(name, param_names, fit_fn, train_results)
        _, _, summary_val = run_one_model(name, param_names, fit_fn, val_results)
        param_list_t, rmse_list_t, summary_test = run_one_model(name, param_names, fit_fn, test_results)
        comparison = print_and_save_comparison(name, param_names, summary_train, summary_val, summary_test, OUT_DIR)
        all_comparisons.append(comparison)
        if test_results:
            plot_param_distributions(name, param_names, param_list_t, rmse_list_t, summary_test)

    with open(os.path.join(OUT_DIR, "all_splits_comparison.json"), "w") as f:
        json.dump(all_comparisons, f, indent=2)

    print("\n" + "=" * 60)
    print("RMSE: train | val | test (mean ± std)")
    print("=" * 60)
    for c in all_comparisons:
        name = c["model"]
        rt = c.get("rmse", {}).get("train")
        rv = c.get("rmse", {}).get("val")
        rte = c.get("rmse", {}).get("test")
        pt = f"{rt['mean']:.4f} ± {rt['std']:.4f}" if rt else "—"
        pv = f"{rv['mean']:.4f} ± {rv['std']:.4f}" if rv else "—"
        pte = f"{rte['mean']:.4f} ± {rte['std']:.4f}" if rte else "—"
        print(f"  {name:20s}  train: {pt:20s}  val: {pv:20s}  test: {pte:20s}")
    print(f"\nResults and comparison JSONs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
