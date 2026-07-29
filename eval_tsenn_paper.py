#!/usr/bin/env python3
"""
Sanity-check TSENN weights on the paper's own test split.

Offline only (safe for sbatch once dataset+checkpoint are local):
  1) Place TSENN_dataset_*.tar.gz + TSENN_models_*.tar.gz in repo
  2) ./setup_tsenn.sh --tar ... --dataset-tar ...
  3) python eval_tsenn_paper.py
     or: sbatch sbatch_eval_tsenn_paper.sh

Compares mean Cartesian MAE on idx_test_250909 to the paper notebook value (~0.127).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric as tg
from e3nn import o3
from e3nn.io import CartesianTensor
from scipy.interpolate import interp1d
from tqdm import tqdm

import predict_tsenn_opt as pts

ROOT = Path(__file__).resolve().parent
TSENN_ROOT = ROOT / "external" / "TSENN"
DEFAULT_DATA = TSENN_ROOT / "dataset" / "symmetrized_dataset_with_bandgap.pkl"
DEFAULT_CKPT = (
    TSENN_ROOT
    / "model"
    / "revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch"
)
DEFAULT_IDX_TEST = TSENN_ROOT / "model" / "idx_test_250909.txt"
DEFAULT_IDX_TRAIN = TSENN_ROOT / "model" / "idx_train_250909.txt"

# Paper notebook (imaginary_part_prediction.ipynb) test-set mean MAE
PAPER_MEAN_MAE_CART = 0.12729490049316006
PAPER_MEDIAN_MAE_CART = 0.09692013405198992


def load_indices(path):
    return [int(x) for x in Path(path).read_text().strip().splitlines() if x.strip()]


def interpolate_imag(df, energy_min=0.0, energy_max=30.0, nstep=300):
    new_x = np.linspace(energy_min, energy_max, nstep)

    def _interp(row):
        return interp1d(
            row["omega"],
            row["rel_permittivity_imag"],
            kind="linear",
            axis=0,
            fill_value=0.0,
            bounds_error=False,
        )(new_x)

    df = df.copy()
    df["rel_permittivity_imag_interp"] = [
        _interp(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="interp")
    ]
    df["energies_interp"] = [new_x] * len(df)
    return df, new_x


def compute_scales(imag_stack):
    """Match train_full_im_tensor / paper notebook denorm scales."""
    from utils.normalize_cart import cart_to_sph

    device = "cpu"
    stack = torch.tensor(imag_stack, dtype=torch.float64, device=device)
    sph = cart_to_sph(stack)
    scalar_0e = sph[:, :, 0]
    tensor_2e = sph[:, :, 1:]
    scale_0e = torch.mean(torch.max(torch.abs(scalar_0e), dim=1).values)
    scale_2e = torch.median(torch.max(torch.norm(tensor_2e, dim=-1), dim=1).values)
    return float(scale_0e), float(scale_2e)


def mean_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        n_atoms = entry.data.pos.shape[0]
        for i in range(n_atoms):
            n.append(int((entry.data.edge_index[0] == i).sum()))
    return float(np.mean(n)) if n else pts.NUM_NEIGHBORS


def symmetric_mae_cart(pred, true):
    """pred/true: (N, F, 3, 3) → per-sample mean MAE over 6 unique components."""
    inds = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    diffs = []
    for i, j in inds:
        diffs.append(np.abs(pred[:, :, i, j] - true[:, :, i, j]))
    stacked = np.stack(diffs, axis=0)  # (6, N, F)
    return stacked.mean(axis=(0, 2))  # (N,)


def predict_eps2(model, graphs, scale_0e, scale_2e, device, batch_size=8):
    cart = CartesianTensor("ij=ji")
    loader = tg.loader.DataLoader(graphs, batch_size=batch_size)
    outs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            batch = batch.to(device)
            output = model(batch)
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
            out_dim = irreps_0e
            output_0e = output[:, :irreps_0e] * scale_0e
            output_2e = (
                output[:, irreps_0e : irreps_0e + irreps_2e]
                .contiguous()
                .view(output.shape[0], out_dim, 5)
                * scale_2e
            )
            sph = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)
            outs.append(cart.to_cartesian(sph).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="TSENN paper-dataset MAE sanity check")
    parser.add_argument("--dataset", default=str(DEFAULT_DATA))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--idx_test", default=str(DEFAULT_IDX_TEST))
    parser.add_argument("--idx_train", default=str(DEFAULT_IDX_TRAIN))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Subsample test for quick smoke")
    parser.add_argument("--expected_mae", type=float, default=PAPER_MEAN_MAE_CART)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.02,
        help="Pass if |mean_mae - expected| <= atol (full test)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--out_json", default="tsenn_opt/paper_eval.json")
    args = parser.parse_args()

    for p, label in [
        (args.dataset, "dataset pkl"),
        (args.checkpoint, "checkpoint"),
        (args.idx_test, "idx_test"),
        (args.idx_train, "idx_train"),
    ]:
        if not Path(p).is_file():
            raise FileNotFoundError(
                f"Missing {label}: {p}\n"
                "On login node: ./setup_tsenn.sh --tar TSENN_models_*.tar.gz "
                "--dataset-tar TSENN_dataset_*.tar.gz"
            )

    pts._ensure_tsenn_path()
    torch.set_default_dtype(pts.DEFAULT_DTYPE)

    from utils.normalize_cart import cart_to_sph
    from utils.utils_data import build_data, load_data, save_or_load_onehot
    from utils.utils_model_full_tensor import Network

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    df, _ = load_data(args.dataset)
    df = df.reset_index(drop=True)
    print(f"loaded {len(df)} materials from {args.dataset}")

    df, omega = interpolate_imag(df)
    imag_stack = np.stack(df["rel_permittivity_imag_interp"].values)
    scale_0e, scale_2e = compute_scales(imag_stack)
    print(f"scale_0e={scale_0e:.4f} scale_2e={scale_2e:.4f}")

    # Normalized spherical targets for build_data.y (unused for MAE vs cart truth)
    sph = cart_to_sph(torch.tensor(imag_stack, dtype=torch.float64))
    sph_np = sph.cpu().numpy().copy()
    sph_np[:, :, 0] /= scale_0e + 1e-12
    sph_np[:, :, 1:] /= scale_2e + 1e-12
    df["sph_coefs"] = list(sph_np)

    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    onehots = save_or_load_onehot(save_path=onehot_dir)
    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = onehots

    print("Building graphs...")
    df["data"] = [
        build_data(
            row,
            "sph_coefs",
            1.0,
            type_onehot,
            mass_onehot,
            dipole_onehot,
            radius_onehot,
            type_encoding,
            pts.R_MAX,
        )
        for _, row in tqdm(df.iterrows(), total=len(df), desc="graphs")
    ]

    idx_train = load_indices(args.idx_train)
    idx_test = load_indices(args.idx_test)
    if args.limit is not None:
        idx_test = idx_test[: args.limit]
        print(f"limited test to {len(idx_test)} samples")

    num_neighbors = mean_neighbors(df, idx_train)
    print(f"num_neighbors (train mean)={num_neighbors:.4f}")

    # Rebuild model with paper num_neighbors
    PeriodicNet = pts.make_periodic_network(Network)
    model = PeriodicNet(
        in_dim=118,
        em_dim=pts.EM_DIM,
        irreps_in=f"{pts.EM_DIM}x0e",
        irreps_out=f"{pts.NSTEP}x0e + {pts.NSTEP}x2e",
        irreps_node_attr=f"{pts.EM_DIM}x0e",
        layers=pts.LAYERS,
        mul=pts.MUL,
        lmax=pts.LMAX,
        max_radius=pts.R_MAX,
        num_neighbors=num_neighbors,
        reduce_output=True,
        dropout_prob=0.0,
        use_batch_norm=False,
    )
    model.to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["state"])
    model.pool = True
    print(f"Loaded {Path(args.checkpoint).name}")

    graphs = list(df.iloc[idx_test]["data"].values)
    pred = predict_eps2(
        model, graphs, scale_0e, scale_2e, device, batch_size=args.batch_size
    )
    true = imag_stack[np.asarray(idx_test)]
    per_sample = symmetric_mae_cart(pred, true)
    mean_mae = float(per_sample.mean())
    median_mae = float(np.median(per_sample))

    print(f"test n={len(idx_test)}")
    print(f"mean MAE cart   = {mean_mae:.6f}  (paper {args.expected_mae:.6f})")
    print(f"median MAE cart = {median_mae:.6f}  (paper {PAPER_MEDIAN_MAE_CART:.6f})")

    # Full-test gate; for --limit only report (don't fail on expected mismatch)
    delta = abs(mean_mae - args.expected_mae)
    if args.limit is None:
        ok = delta <= args.atol
        status = "PASS" if ok else "FAIL"
        print(f"{status}: |Δmean MAE|={delta:.6f} (atol={args.atol})")
    else:
        ok = np.isfinite(mean_mae) and mean_mae < 1.0
        status = "PASS_SMOKE" if ok else "FAIL_SMOKE"
        print(f"{status}: finite MAE={mean_mae:.6f} (<1.0 sanity on subset)")

    out = {
        "status": status,
        "n_test": len(idx_test),
        "mean_mae_cart": mean_mae,
        "median_mae_cart": median_mae,
        "expected_mean_mae_cart": args.expected_mae,
        "delta_mean_mae": delta,
        "atol": args.atol,
        "scale_0e": scale_0e,
        "scale_2e": scale_2e,
        "num_neighbors": num_neighbors,
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "limited": args.limit is not None,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
