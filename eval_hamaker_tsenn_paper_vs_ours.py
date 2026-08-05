#!/usr/bin/env python3
"""
Hamaker bake-off on the *TSENN paper* test set.

Protocol (all Hamaker ground truth from the paper dataset only):
  1) Truncate paper DFT ε₂(ω) to [0, ω_max] (default 30 eV)
  2) KKR → ε(iξ) on a shared log ξ grid → self-Hamaker A_gt
  3) TSENN on the same structures → ε₂ → same KKR → A_tsenn
  4) Our CGCNN/MACE checkpoint on the same structures → ε(iξ)
     (ξ clipped to ≤ --xi_max, default 30 eV) → A_ours

Does not touch JARVIS OPT LMDBs for GT. Offline once dataset + checkpoints are local.
Pass --lmdb_path only to recover our training ξ grid.

Usage:
  python eval_hamaker_tsenn_paper_vs_ours.py --our_model_path models_latest_opt/cgcnn_kkr_latest_opt.pt
  python eval_hamaker_tsenn_paper_vs_ours.py --our_model_path ... --limit 8
  python eval_hamaker_tsenn_paper_vs_ours.py --our_model_path ... --component iso
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.interpolate import interp1d
from tqdm import tqdm

import torch_load_compat  # noqa: F401  # before e3nn (PyTorch 2.6+)
import predict_tsenn_opt as pts

import torch_geometric as tg
from e3nn import o3
from e3nn.io import CartesianTensor

from compare_hamaker_constants import (
    K_B_EV,
    T_K,
    hamaker_constant_zj,
)
from eval_hamaker_pair_percentiles import PERCENTILES, print_percentile_table
from lmdb_dataset import collate_crystal_graphs
from mace_data import collate_mace_pool, structure_arrays_to_atomic_data, z_table_from_atom_init
from model_forward import model_forward
from model_loader import load_model
from preprocess_dataset import (
    GaussianDistance,
    apply_kkr,
    get_atom_features,
    get_neighbor_features,
    load_atom_init,
)

ROOT = Path(__file__).resolve().parent
TSENN_ROOT = ROOT / "external" / "TSENN"
DEFAULT_DATA = TSENN_ROOT / "dataset" / "symmetrized_dataset_with_bandgap.pkl"
DEFAULT_TSENN_CKPT = (
    TSENN_ROOT
    / "model"
    / "revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch"
)
DEFAULT_IDX_TEST = TSENN_ROOT / "model" / "idx_test_250909.txt"
DEFAULT_IDX_TRAIN = TSENN_ROOT / "model" / "idx_train_250909.txt"


def load_indices(path):
    return [int(x) for x in Path(path).read_text().strip().splitlines() if x.strip()]


def interpolate_imag(df, energy_min, energy_max, nstep):
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
        _interp(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="interp-dft")
    ]
    return df, new_x


def compute_scales(imag_stack):
    from utils.normalize_cart import cart_to_sph

    stack = torch.tensor(imag_stack, dtype=torch.float64)
    sph = cart_to_sph(stack)
    scale_0e = float(torch.mean(torch.max(torch.abs(sph[:, :, 0]), dim=1).values))
    scale_2e = float(
        torch.median(torch.max(torch.norm(sph[:, :, 1:], dim=-1), dim=1).values)
    )
    return scale_0e, scale_2e


def mean_neighbors(df, idx):
    n = []
    for entry in df.iloc[idx].itertuples():
        n_atoms = entry.data.pos.shape[0]
        for i in range(n_atoms):
            n.append(int((entry.data.edge_index[0] == i).sum()))
    return float(np.mean(n)) if n else pts.NUM_NEIGHBORS


def tensor_component(eps2, component):
    """(F, 3, 3) → 1D ε₂(ω)."""
    eps2 = np.asarray(eps2, dtype=np.float64)
    if component == "iso":
        return (eps2[:, 0, 0] + eps2[:, 1, 1] + eps2[:, 2, 2]) / 3.0
    idx = {"xx": (0, 0), "yy": (1, 1), "zz": (2, 2), "xy": (0, 1), "yz": (1, 2), "zx": (2, 0)}
    i, j = idx[component]
    return eps2[:, i, j]


def kkr_eps(omega, eps2, xi, clip_nonneg):
    eps2 = np.asarray(eps2, dtype=np.float64)
    if clip_nonneg:
        eps2 = np.clip(eps2, 0.0, None)
    _, eps_kkr = apply_kkr(omega, eps2, xi_points=xi)
    return np.asarray(eps_kkr, dtype=np.float64)


def make_shared_xi(omega_max, n_xi):
    return np.logspace(np.log10(0.01), np.log10(float(omega_max)), int(n_xi))


def predict_tsenn_eps2(model, graphs, scale_0e, scale_2e, device, batch_size):
    cart = CartesianTensor("ij=ji")
    loader = tg.loader.DataLoader(graphs, batch_size=batch_size)
    outs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="tsenn-infer"):
            batch = batch.to(device)
            output = model(batch)
            n0 = model.irreps_out.count(o3.Irrep("0e"))
            n2 = model.irreps_out.count(o3.Irrep("2e")) * 5
            out0 = output[:, :n0] * scale_0e
            out2 = (
                output[:, n0 : n0 + n2]
                .contiguous()
                .view(output.shape[0], n0, 5)
                * scale_2e
            )
            sph = torch.cat([out0.unsqueeze(2), out2], dim=2)
            outs.append(cart.to_cartesian(sph).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float64)


def ase_to_structure(atoms):
    return AseAtomsAdaptor.get_structure(atoms)


def build_our_inputs(structure, model_type, ckpt, atom_init, gdf, z_table):
    radius = float(ckpt.get("radius", ckpt.get("mace_r_max", 8.0)))
    max_num_nbr = int(ckpt.get("max_num_nbr", 12))
    if model_type == "MACE":
        r_max = float(ckpt.get("mace_r_max", radius))
        positions = np.asarray(structure.cart_coords, dtype=np.float64)
        cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
        atomic_numbers = np.asarray(
            [int(s.specie.Z) for s in structure], dtype=np.int64
        )
        pbc = np.asarray(structure.lattice.pbc, dtype=bool)
        graph = structure_arrays_to_atomic_data(
            positions, cell, atomic_numbers, pbc, z_table, r_max
        )
        return graph

    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(
        structure, radius, max_num_nbr, gdf
    )
    return atom_fea, nbr_fea, nbr_fea_idx


@torch.no_grad()
def predict_ours_batch(model, model_type, items, device, use_log, channel):
    if model_type == "MACE":
        batch, _, _ = collate_mace_pool(
            [(g, np.zeros((1, 1), dtype=np.float32), False) for g in items]
        )
        return model_forward(
            model, model_type, batch, device, use_log=use_log, channel=channel
        ).cpu().numpy()

    graphs = items
    inputs = collate_crystal_graphs(graphs)
    return model_forward(
        model, model_type, inputs, device, use_log=use_log, channel=channel
    ).cpu().numpy()


def rel_err_pct(pred, true):
    return 100.0 * (pred - true) / (true + 1e-30)


def summarize_hamaker(name, a_pred, a_true):
    a_pred = np.asarray(a_pred, dtype=np.float64)
    a_true = np.asarray(a_true, dtype=np.float64)
    rel = rel_err_pct(a_pred, a_true)
    abs_rel = np.abs(rel)
    mae = float(np.mean(np.abs(a_pred - a_true)))
    rmse = float(np.sqrt(np.mean((a_pred - a_true) ** 2)))
    r = float(np.corrcoef(a_pred, a_true)[0, 1]) if len(a_true) > 1 else float("nan")
    print(f"\n--- {name} ---")
    print(f"  MAE(ΔA)   = {mae:.4f} zJ")
    print(f"  RMSE(ΔA)  = {rmse:.4f} zJ")
    print(f"  Pearson r = {r:.4f}")
    print_percentile_table(
        f"{name}: signed rel err (A_pred−A_gt)/A_gt × 100",
        rel,
        PERCENTILES,
        unit="%",
    )
    print_percentile_table(
        f"{name}: abs rel err |A_pred−A_gt|/|A_gt| × 100",
        abs_rel,
        PERCENTILES,
        unit="%",
    )
    return {
        "n": int(len(a_true)),
        "mae_zj": mae,
        "rmse_zj": rmse,
        "pearson_r": r,
        "abs_rel_median_pct": float(np.median(abs_rel)) if len(abs_rel) else None,
        "abs_rel_mean_pct": float(np.mean(abs_rel)) if len(abs_rel) else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hamaker: TSENN vs our model on TSENN paper test set",
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATA))
    parser.add_argument("--tsenn_checkpoint", default=str(DEFAULT_TSENN_CKPT))
    parser.add_argument("--idx_test", default=str(DEFAULT_IDX_TEST))
    parser.add_argument("--idx_train", default=str(DEFAULT_IDX_TRAIN))
    parser.add_argument(
        "--our_model_path",
        required=True,
        help="Our CGCNN/MACE .pt checkpoint (trained elsewhere)",
    )
    parser.add_argument("--omega_max", type=float, default=30.0)
    parser.add_argument(
        "--n_omega",
        type=int,
        default=pts.NSTEP,
        help="Must match TSENN checkpoint spectral length (default 300)",
    )
    parser.add_argument("--n_xi", type=int, default=500)
    parser.add_argument(
        "--component",
        default="xx",
        choices=("xx", "yy", "zz", "xy", "yz", "zx", "iso"),
    )
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--atom_init", default="atom_init.json")
    parser.add_argument(
        "--lmdb_path",
        default=None,
        help="Training LMDB for our model (to recover kkr_xi_array; checkpoints omit it)",
    )
    parser.add_argument(
        "--xi_max",
        type=float,
        default=30.0,
        help="Clip our ε(iξ) grid to ξ≤xi_max before Hamaker (match truncated GT/TSENN). "
        "Set ≤0 to disable.",
    )
    parser.add_argument("--out_json", default="tsenn_opt/hamaker_paper_vs_ours.json")
    args = parser.parse_args()

    for p, label in [
        (args.dataset, "paper dataset pkl"),
        (args.tsenn_checkpoint, "TSENN checkpoint"),
        (args.idx_test, "idx_test"),
        (args.idx_train, "idx_train"),
        (args.our_model_path, "our model"),
    ]:
        if not Path(p).is_file():
            raise FileNotFoundError(
                f"Missing {label}: {p}\n"
                "Prep: ./setup_tsenn.sh --tar TSENN_models_*.tar.gz "
                "--dataset-tar TSENN_dataset_*.tar.gz"
            )

    if args.n_omega != pts.NSTEP:
        raise ValueError(
            f"--n_omega={args.n_omega} must equal TSENN checkpoint length "
            f"({pts.NSTEP}); refuse to load mismatched irreps_out"
        )

    clip_nonneg = args.component in ("xx", "yy", "zz", "iso")
    xi_step_ev = 2.0 * np.pi * K_B_EV * args.temperature_k
    hamaker_kw = dict(
        t_k=args.temperature_k,
        xi_step_ev=xi_step_ev,
        n_max=args.n_matsubara,
    )
    xi_shared = make_shared_xi(args.omega_max, args.n_xi)

    pts._ensure_tsenn_path()
    torch.set_default_dtype(pts.DEFAULT_DTYPE)

    from utils.normalize_cart import cart_to_sph
    from utils.utils_data import build_data, load_data, save_or_load_onehot
    from utils.utils_model_full_tensor import Network

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(
        f"ω ∈ [0, {args.omega_max}] eV  n_omega={args.n_omega}  "
        f"n_xi={args.n_xi}  component={args.component}"
    )

    # ----- paper DFT -----
    df, _ = load_data(args.dataset)
    df = df.reset_index(drop=True)
    print(f"loaded {len(df)} materials from paper dataset")

    df, omega = interpolate_imag(df, 0.0, args.omega_max, args.n_omega)
    imag_stack = np.stack(df["rel_permittivity_imag_interp"].values)  # (N, F, 3, 3)
    scale_0e, scale_2e = compute_scales(imag_stack)
    print(f"TSENN denorm scales: 0e={scale_0e:.4f}  2e={scale_2e:.4f}")

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
    print(f"num_neighbors (train mean)={num_neighbors:.4f}")

    # ----- TSENN model -----
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
    print(f"Loaded TSENN {Path(args.tsenn_checkpoint).name}")

    graphs = list(df.iloc[idx_test]["data"].values)
    eps2_tsenn = predict_tsenn_eps2(
        tsenn, graphs, scale_0e, scale_2e, device, args.batch_size
    )
    eps2_true = imag_stack[np.asarray(idx_test)]

    # ----- our model -----
    # Reload default dtype for our float32 checkpoints
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
            print(f"Loaded kkr_xi_array from {args.lmdb_path}  (n={len(xi_ours)})")
    if xi_ours is None:
        xi_ours = np.logspace(np.log10(0.01), np.log10(100.0), n_freq)
        print(
            "WARN: no kkr_xi_array in checkpoint/LMDB; "
            f"using logspace(0.01, 100, {n_freq}). Pass --lmdb_path for the training LMDB."
        )

    xi_ours_full = xi_ours
    if args.xi_max is not None and args.xi_max > 0:
        mask_xi = xi_ours_full <= float(args.xi_max)
        if int(mask_xi.sum()) < 4:
            raise ValueError(
                f"--xi_max={args.xi_max} leaves <4 ξ points "
                f"(xi range [{xi_ours_full.min():.4g}, {xi_ours_full.max():.4g}])"
            )
        print(
            f"Clipping our ξ to ≤{args.xi_max} eV for Hamaker: "
            f"{len(xi_ours_full)} → {int(mask_xi.sum())} points "
            f"(was max ξ={xi_ours_full.max():.4g} eV)"
        )
    else:
        mask_xi = np.ones(len(xi_ours_full), dtype=bool)
        print("Our ξ clip disabled (--xi_max≤0); using full training grid for Hamaker")

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
    print(f"Loaded our model ({our_type}): {args.our_model_path}  use_log={use_log}")

    channel = 0  # xx (or first target component)
    our_items = []
    our_ok = []
    meta_rows = []
    for local_i, idx in enumerate(tqdm(idx_test, desc="build-ours")):
        row = df.iloc[idx]
        atoms = row["symmetrized_structure"]
        mpid = str(row.get("mp_id", row.get("mpid", "")))
        formula = str(row.get("chemical formula", row.get("formula", "")))
        meta_rows.append({"df_idx": int(idx), "mpid": mpid, "formula": formula})
        try:
            structure = ase_to_structure(atoms)
            item = build_our_inputs(
                structure, our_type, our_ckpt, atom_init, gdf, z_table
            )
            our_items.append(item)
            our_ok.append(local_i)
        except Exception as e:
            print(f"  skip our-model idx={idx} mpid={mpid}: {e}")

    # Store full-length predictions; Hamaker uses ξ≤xi_max slice
    eps_ours_all = np.full((len(idx_test), len(xi_ours_full)), np.nan, dtype=np.float64)
    for start in tqdm(range(0, len(our_items), args.batch_size), desc="ours-infer"):
        sl = slice(start, start + args.batch_size)
        batch_items = our_items[sl]
        batch_locals = our_ok[sl]
        pred = predict_ours_batch(
            our_model, our_type, batch_items, device, use_log, channel
        )
        for j, loc in enumerate(batch_locals):
            eps_ours_all[loc] = pred[j]

    xi_ours_ham = xi_ours_full[mask_xi]

    # ----- KKR + Hamaker -----
    a_gt, a_tsenn, a_ours = [], [], []
    mae_eps_tsenn = []
    mae_eps_ours = []
    n_fail = 0

    for local_i in tqdm(range(len(idx_test)), desc="hamaker"):
        try:
            e2_t = tensor_component(eps2_true[local_i], args.component)
            e2_p = tensor_component(eps2_tsenn[local_i], args.component)
            eps_gt = kkr_eps(omega, e2_t, xi_shared, clip_nonneg)
            eps_ts = kkr_eps(omega, e2_p, xi_shared, clip_nonneg)
            ag = hamaker_constant_zj(xi_shared, eps_gt, **hamaker_kw)
            at = hamaker_constant_zj(xi_shared, eps_ts, **hamaker_kw)
        except Exception as e:
            n_fail += 1
            print(f"  fail KKR/Hamaker local={local_i} {meta_rows[local_i]}: {e}")
            continue

        a_gt.append(ag)
        a_tsenn.append(at)
        mae_eps_tsenn.append(float(np.mean(np.abs(eps_ts - eps_gt))))

        eps_o_full = eps_ours_all[local_i]
        if np.all(np.isfinite(eps_o_full)):
            eps_o = eps_o_full[mask_xi]
            ao = hamaker_constant_zj(xi_ours_ham, eps_o, **hamaker_kw)
            a_ours.append(ao)
            # ε MAE on shared ξ via interp of clipped prediction
            interp = interp1d(
                xi_ours_ham,
                eps_o,
                kind="linear",
                bounds_error=False,
                fill_value=(eps_o[0], 1.0),
            )
            eps_o_on_shared = interp(xi_shared)
            mae_eps_ours.append(float(np.mean(np.abs(eps_o_on_shared - eps_gt))))
        else:
            a_ours.append(np.nan)
            mae_eps_ours.append(np.nan)

    a_gt = np.asarray(a_gt, dtype=np.float64)
    a_tsenn = np.asarray(a_tsenn, dtype=np.float64)
    a_ours = np.asarray(a_ours, dtype=np.float64)
    mae_eps_tsenn = np.asarray(mae_eps_tsenn, dtype=np.float64)
    mae_eps_ours = np.asarray(mae_eps_ours, dtype=np.float64)

    ours_mask = np.isfinite(a_ours)
    print("\n" + "=" * 60)
    print("Hamaker bake-off on TSENN paper test set")
    print(
        f"  ω_max={args.omega_max}  xi_max_ours={args.xi_max}  "
        f"component={args.component}  "
        f"n={len(a_gt)}  kkr_fail={n_fail}  ours_ok={int(ours_mask.sum())}"
    )
    print("=" * 60)

    print("\n--- Core ε(iξ) MAE (truncated paper DFT vs model→KKR / ours) ---")
    print_percentile_table("TSENN ε(iξ) MAE vs GT", mae_eps_tsenn, PERCENTILES)
    print_percentile_table(
        "Ours ε(iξ) MAE vs GT (interp onto shared ξ)",
        mae_eps_ours[ours_mask],
        PERCENTILES,
    )

    stats_tsenn = summarize_hamaker("TSENN vs GT Hamaker", a_tsenn, a_gt)
    stats_ours = None
    if ours_mask.any():
        stats_ours = summarize_hamaker(
            f"Ours ({our_type}) vs GT Hamaker",
            a_ours[ours_mask],
            a_gt[ours_mask],
        )
    else:
        print("\n--- Ours vs GT Hamaker ---")
        print("  (no successful our-model predictions)")

    out = {
        "omega_max": args.omega_max,
        "n_omega": args.n_omega,
        "n_xi": args.n_xi,
        "xi_max_ours": args.xi_max,
        "n_xi_ours_hamaker": int(mask_xi.sum()),
        "component": args.component,
        "n_test": len(idx_test),
        "n_hamaker": len(a_gt),
        "n_fail_kkr": n_fail,
        "n_ours_ok": int(ours_mask.sum()),
        "temperature_k": args.temperature_k,
        "n_matsubara": args.n_matsubara,
        "tsenn_checkpoint": str(args.tsenn_checkpoint),
        "our_model_path": str(args.our_model_path),
        "our_model_type": our_type,
        "core_mae_tsenn_mean": float(np.nanmean(mae_eps_tsenn)),
        "core_mae_ours_mean": float(np.nanmean(mae_eps_ours)) if ours_mask.any() else None,
        "tsenn_hamaker": stats_tsenn,
        "ours_hamaker": stats_ours,
        "a_gt_mean_zj": float(np.mean(a_gt)) if len(a_gt) else None,
        "a_tsenn_mean_zj": float(np.mean(a_tsenn)) if len(a_tsenn) else None,
        "a_ours_mean_zj": float(np.nanmean(a_ours)) if ours_mask.any() else None,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
