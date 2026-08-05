#!/usr/bin/env python3
"""
Plot dielectric functions on the TSENN paper test set:
  GT (paper DFT) vs TSENN vs our CGCNN/MACE.

Each figure has two panels:
  (a) ε₂(ω)     — GT and TSENN only (our model predicts ε(iξ), not ε₂)
  (b) ε(iξ)     — GT and TSENN via KKR on [0, ω_max]; ours direct (ξ≤xi_max)

Outputs PNGs under tsenn_vs_ours_dielectric/ (or --out_dir).

Usage:
  python plot_tsenn_vs_ours_dielectric.py \\
    --our_model_path models/mace_kkr_v1.pt \\
    --lmdb_path processed_data/crystal_graphs.lmdb

  python plot_tsenn_vs_ours_dielectric.py ... --limit 12
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import torch_load_compat  # noqa: F401
import predict_tsenn_opt as pts

from eval_hamaker_tsenn_paper_vs_ours import (
    DEFAULT_DATA,
    DEFAULT_IDX_TEST,
    DEFAULT_IDX_TRAIN,
    DEFAULT_TSENN_CKPT,
    TSENN_ROOT,
    ase_to_structure,
    build_our_inputs,
    compute_scales,
    interpolate_imag,
    kkr_eps,
    load_indices,
    make_shared_xi,
    mean_neighbors,
    predict_ours_batch,
    predict_tsenn_eps2,
    tensor_component,
)
from mace_data import z_table_from_atom_init
from model_loader import load_model
from preprocess_dataset import GaussianDistance, load_atom_init

COLOR_GT = "#111111"
COLOR_TSENN = "#1f77b4"
COLOR_OURS = "#d62728"
COLOR_GAP = "#2ca02c"

# TSENN pickle column names vary slightly across releases / notebooks
BANDGAP_KEYS = (
    "band gap",
    "bandgap",
    "bandgap_mp",
    "bandgap_openmx",
    "band_gap",
)


def row_bandgap(row):
    """Read bandgap (eV) from a TSENN dataset row; None if missing."""
    for key in BANDGAP_KEYS:
        if key not in row.index:
            continue
        val = row[key]
        if val is None:
            continue
        try:
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                continue
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def safe_name(mpid, formula, df_idx):
    raw = f"{mpid or 'noid'}_{formula or 'unk'}_{df_idx}"
    return re.sub(r"[^\w\-.]+", "_", raw)[:120]


def plot_one(
    out_path,
    omega,
    eps2_gt,
    eps2_ts,
    xi_shared,
    eps_gt,
    eps_ts,
    xi_ours,
    eps_ours,
    title,
    component,
    bandgap=None,
):
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8.5, 7.5), constrained_layout=True)

    ax0.plot(omega, eps2_gt, color=COLOR_GT, lw=1.8, label="GT (paper DFT)")
    ax0.plot(omega, eps2_ts, color=COLOR_TSENN, lw=1.6, label="TSENN")
    if bandgap is not None and float(omega[0]) <= bandgap <= float(omega[-1]):
        ax0.axvline(
            bandgap,
            color=COLOR_GAP,
            ls="--",
            lw=1.4,
            label=rf"$E_g$ = {bandgap:.3f} eV",
        )
    ax0.set_xlabel(r"$\omega$ (eV)")
    ax0.set_ylabel(rf"$\varepsilon_2^{{{component}}}(\omega)$")
    ax0.set_xlim(float(omega[0]), float(omega[-1]))
    ax0.legend(frameon=False, fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax0.set_title(title, fontsize=11)

    ax1.plot(xi_shared, eps_gt, color=COLOR_GT, lw=1.8, label="GT (KKR)")
    ax1.plot(xi_shared, eps_ts, color=COLOR_TSENN, lw=1.6, label="TSENN (KKR)")
    if eps_ours is not None:
        ax1.plot(xi_ours, eps_ours, color=COLOR_OURS, lw=1.6, label="Ours")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\xi$ (eV)")
    ax1.set_ylabel(rf"$\varepsilon^{{{component}}}(i\xi)$")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GT / TSENN / ours dielectric curves on paper test set",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATA))
    parser.add_argument("--tsenn_checkpoint", default=str(DEFAULT_TSENN_CKPT))
    parser.add_argument("--idx_test", default=str(DEFAULT_IDX_TEST))
    parser.add_argument("--idx_train", default=str(DEFAULT_IDX_TRAIN))
    parser.add_argument("--our_model_path", required=True)
    parser.add_argument("--lmdb_path", default=None)
    parser.add_argument("--omega_max", type=float, default=30.0)
    parser.add_argument("--n_omega", type=int, default=pts.NSTEP)
    parser.add_argument("--n_xi", type=int, default=500)
    parser.add_argument(
        "--xi_max",
        type=float,
        default=30.0,
        help="Clip our ξ to ≤xi_max for ε(iξ) panel; ≤0 disables",
    )
    parser.add_argument(
        "--component",
        default="xx",
        choices=("xx", "yy", "zz", "xy", "yz", "zx", "iso"),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--atom_init", default="atom_init.json")
    parser.add_argument("--out_dir", default="tsenn_vs_ours_dielectric")
    args = parser.parse_args()

    for p, label in [
        (args.dataset, "paper dataset"),
        (args.tsenn_checkpoint, "TSENN checkpoint"),
        (args.idx_test, "idx_test"),
        (args.idx_train, "idx_train"),
        (args.our_model_path, "our model"),
    ]:
        if not Path(p).is_file():
            raise FileNotFoundError(f"Missing {label}: {p}")

    if args.n_omega != pts.NSTEP:
        raise ValueError(f"--n_omega must be {pts.NSTEP} to match TSENN checkpoint")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_nonneg = args.component in ("xx", "yy", "zz", "iso")
    xi_shared = make_shared_xi(args.omega_max, args.n_xi)

    pts._ensure_tsenn_path()
    torch.set_default_dtype(pts.DEFAULT_DTYPE)

    from utils.normalize_cart import cart_to_sph
    from utils.utils_data import build_data, load_data, save_or_load_onehot
    from utils.utils_model_full_tensor import Network

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  out_dir={out_dir}")

    df, _ = load_data(args.dataset)
    df = df.reset_index(drop=True)
    df, omega = interpolate_imag(df, 0.0, args.omega_max, args.n_omega)
    imag_stack = np.stack(df["rel_permittivity_imag_interp"].values)
    scale_0e, scale_2e = compute_scales(imag_stack)

    sph = cart_to_sph(torch.tensor(imag_stack, dtype=torch.float64))
    sph_np = sph.cpu().numpy().copy()
    sph_np[:, :, 0] /= scale_0e + 1e-12
    sph_np[:, :, 1:] /= scale_2e + 1e-12
    df["sph_coefs"] = list(sph_np)

    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    onehots = save_or_load_onehot(save_path=onehot_dir)
    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = onehots

    print("Building TSENN graphs...")
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
        print(f"limited test to {len(idx_test)}")

    num_neighbors = mean_neighbors(df, idx_train)
    PeriodicNet = pts.make_periodic_network(Network)
    tsenn = PeriodicNet(
        in_dim=118,
        em_dim=pts.EM_DIM,
        irreps_in=f"{pts.EM_DIM}x0e",
        irreps_out=f"{args.n_omega}x0e + {args.n_omega}x2e",
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
    tsenn.to(device)
    state = torch.load(args.tsenn_checkpoint, map_location=device)
    tsenn.load_state_dict(state["state"])
    tsenn.pool = True
    tsenn.eval()

    graphs = list(df.iloc[idx_test]["data"].values)
    eps2_tsenn = predict_tsenn_eps2(
        tsenn, graphs, scale_0e, scale_2e, device, args.batch_size
    )
    eps2_true = imag_stack[np.asarray(idx_test)]

    # ----- our model -----
    torch.set_default_dtype(torch.float32)
    our_ckpt = torch.load(args.our_model_path, map_location="cpu", weights_only=False)
    our_model, our_type = load_model(our_ckpt, torch.device(device))
    use_log = bool(our_ckpt.get("use_log", False))
    n_freq = int(our_ckpt["n_frequencies"])

    def _as_xi(raw):
        if raw is None:
            return None
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().numpy()
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        return arr if arr.size == n_freq else None

    xi_ours = _as_xi(our_ckpt.get("kkr_xi_array", our_ckpt.get("xi_array")))
    if xi_ours is None and args.lmdb_path:
        env = lmdb.open(str(args.lmdb_path), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            meta = pickle.loads(txn.get(b"__metadata__"))
        env.close()
        xi_ours = _as_xi(meta.get("kkr_xi_array"))
        if xi_ours is not None:
            print(f"Loaded kkr_xi_array from {args.lmdb_path}")
    if xi_ours is None:
        xi_ours = np.logspace(np.log10(0.01), np.log10(100.0), n_freq)
        print("WARN: using logspace fallback for our ξ grid")

    if args.xi_max is not None and args.xi_max > 0:
        mask_xi = xi_ours <= float(args.xi_max)
        if int(mask_xi.sum()) < 4:
            raise ValueError(f"--xi_max={args.xi_max} leaves too few ξ points")
        print(f"Clipping our ξ to ≤{args.xi_max} eV ({int(mask_xi.sum())} pts)")
    else:
        mask_xi = np.ones(len(xi_ours), dtype=bool)

    atom_init = load_atom_init(args.atom_init)
    radius = float(our_ckpt.get("radius", our_ckpt.get("mace_r_max", 8.0)))
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)
    z_table = None
    if our_type == "MACE":
        zs = our_ckpt.get("mace_atomic_numbers")
        if zs is not None:
            from mace.tools import AtomicNumberTable

            z_table = AtomicNumberTable(zs=[int(z) for z in zs])
        else:
            z_table = z_table_from_atom_init(args.atom_init)

    our_items, our_ok, meta_rows = [], [], []
    for local_i, idx in enumerate(tqdm(idx_test, desc="build-ours")):
        row = df.iloc[idx]
        atoms = row["symmetrized_structure"]
        mpid = str(row.get("mp_id", row.get("mpid", "")))
        formula = str(row.get("chemical formula", row.get("formula", "")))
        meta_rows.append(
            {
                "df_idx": int(idx),
                "mpid": mpid,
                "formula": formula,
                "bandgap": row_bandgap(row),
            }
        )
        try:
            structure = ase_to_structure(atoms)
            our_items.append(
                build_our_inputs(structure, our_type, our_ckpt, atom_init, gdf, z_table)
            )
            our_ok.append(local_i)
        except Exception as e:
            print(f"  skip ours idx={idx} mpid={mpid}: {e}")

    eps_ours_all = np.full((len(idx_test), len(xi_ours)), np.nan, dtype=np.float64)
    for start in tqdm(range(0, len(our_items), args.batch_size), desc="ours-infer"):
        sl = slice(start, start + args.batch_size)
        pred = predict_ours_batch(
            our_model, our_type, our_items[sl], device, use_log, channel=0
        )
        for j, loc in enumerate(our_ok[sl]):
            eps_ours_all[loc] = pred[j]

    xi_ours_plot = xi_ours[mask_xi]
    n_written = 0
    for local_i in tqdm(range(len(idx_test)), desc="plot"):
        meta = meta_rows[local_i]
        e2_gt = tensor_component(eps2_true[local_i], args.component)
        e2_ts = tensor_component(eps2_tsenn[local_i], args.component)
        eps_gt = kkr_eps(omega, e2_gt, xi_shared, clip_nonneg)
        eps_ts = kkr_eps(omega, e2_ts, xi_shared, clip_nonneg)

        eps_o_full = eps_ours_all[local_i]
        if np.all(np.isfinite(eps_o_full)):
            eps_o = eps_o_full[mask_xi]
        else:
            eps_o = None

        eg = meta["bandgap"]
        eg_str = f"$E_g$={eg:.3f} eV" if eg is not None else "$E_g$=n/a"
        title = (
            f"{meta['mpid']}  {meta['formula']}  {eg_str}  "
            f"(df_idx={meta['df_idx']}, {args.component})"
        )
        fname = safe_name(meta["mpid"], meta["formula"], meta["df_idx"]) + ".png"
        plot_one(
            out_dir / fname,
            omega,
            e2_gt,
            e2_ts,
            xi_shared,
            eps_gt,
            eps_ts,
            xi_ours_plot,
            eps_o,
            title,
            args.component,
            bandgap=eg,
        )
        n_written += 1

    print(f"Wrote {n_written} plots → {out_dir}/")


if __name__ == "__main__":
    main()
