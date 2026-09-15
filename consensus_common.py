"""Shared voters, splits, and Hamaker helpers for consensus repulsive screens."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import torch_load_compat  # noqa: F401

from compare_hamaker_constants import get_split_indices
from find_repulsive_lifshitz import (
    classify_sign,
    deltas,
    filter_crystal_indices,
    hamaker_zj_from_delta,
    hamaker_zj_retarded,
    interpolate_eps,
    matsubara_xi,
    predict_model_eps,
    retardation_r,
)
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_loader import load_model


def voter_label(path, prefix):
    return f"{prefix}:{Path(path).stem}"


def add_voter_args(parser):
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument(
        "--model", action="append", default=[], dest="models",
        help="CGCNN/MACE checkpoint (repeatable)",
    )
    parser.add_argument(
        "--tsenn", action="append", default=[], dest="tsenn",
        help="train_tsenn_kkr .torch checkpoint (repeatable)",
    )
    parser.add_argument("--no_dft", action="store_true", help="Do not use LMDB KKR as a voter")
    parser.add_argument(
        "--split", default="test", choices=("train", "val", "test", "all"),
        help="Crystal split. Default test = intersection of all neural voters' held-out sets",
    )
    parser.add_argument(
        "--split_ckpt", default=None,
        help="Extra CGCNN/MACE ckpt whose split to include in the intersection",
    )
    parser.add_argument("--media", default="bromobenzene,diiodomethane")
    parser.add_argument("--min_atoms", type=int, default=2)
    parser.add_argument("--min_density", type=float, default=None)
    parser.add_argument("--min_elements", type=int, default=None)
    parser.add_argument(
        "--keep_elemental",
        action="store_true",
        help="Keep single-element JARVIS cells (default: drop)",
    )
    parser.add_argument(
        "--always_repulsive",
        action="store_true",
        help="Require kind==repulsive (no z sign-flip) for every voter, not just A_nr<0",
    )
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=None)
    parser.add_argument("--z_nm", default="5,100")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--top_print", type=int, default=40)
    return parser


def parse_media(s):
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_z_grid(s):
    return np.array([float(x) for x in s.split(",")], dtype=np.float64)


def open_crystal_dataset(lmdb_path, ckpt, use_mace):
    ds = LMDBCrystalDataset(
        lmdb_path,
        target_components=tuple(ckpt.get("target_components", ("xx",))) if ckpt else ("xx",),
        use_kkr=True,
        metal_gap_threshold=ckpt.get("metal_gap_threshold") if ckpt else None,
        use_mace=use_mace,
        mace_r_max=ckpt.get("mace_r_max") if ckpt else None,
    )
    if use_mace and ckpt is not None and ckpt.get("mace_atomic_numbers") is not None:
        from mace.tools import AtomicNumberTable

        ds.set_mace_graph_config(
            AtomicNumberTable(zs=[int(z) for z in ckpt["mace_atomic_numbers"]]),
            ckpt.get("mace_r_max", ds.mace_r_max),
        )
    return ds


def collect_gt_kkr(dataset, indices, batch_size):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=False),
    )
    gts = []
    for _, target, _ in loader:
        gts.append(target[:, 0, :].numpy())
    return np.clip(np.concatenate(gts, axis=0), 1e-6, None)


def to_matsubara(eps_kkr, xi_kkr, xi_m):
    return np.stack([interpolate_eps(xi_kkr, row, xi_m) for row in eps_kkr])


def predict_ours_eps(model_path, lmdb_path, indices, device, batch_size):
    ckpt = torch.load(model_path, map_location=device)
    model, model_type = load_model(ckpt, device)
    use_mace = model_type == "MACE"
    dataset = open_crystal_dataset(lmdb_path, ckpt, use_mace)
    _, pred_kkr, _ = predict_model_eps(
        model, model_type, dataset, indices, device, bool(ckpt.get("use_log", False)), batch_size,
    )
    xi_kkr = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    dataset.close()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_kkr, xi_kkr, model_type, bool(ckpt.get("use_log", False))


def predict_tsenn_eps(model_path, lmdb_path, indices, device, batch_size, onehot_dir, refine):
    from compare_tsenn_kkr_vs_ours import load_tsenn_kkr, predict_tsenn, read_lmdb_records
    from train_tsenn_kkr import DEFAULT_DTYPE, TSENN_ROOT

    records = read_lmdb_records(lmdb_path, indices)
    torch.set_default_dtype(DEFAULT_DTYPE)
    model, ckpt = load_tsenn_kkr(model_path, device)
    pred = predict_tsenn(
        model, ckpt, records, device, batch_size,
        onehot_dir or str(TSENN_ROOT / "onehot_data"),
        refine=refine,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float32)
    pred = np.clip(np.asarray(pred, dtype=np.float64), 1e-6, None)
    xi_kkr = ckpt.get("kkr_xi_array")
    if xi_kkr is not None:
        xi_kkr = np.asarray(xi_kkr, dtype=np.float64)
    return pred, xi_kkr, bool(ckpt.get("use_log", False))


def hamaker_crystal_vs_fixed(eps_crystal, eps_fixed, eps_medium, temperature_k):
    d_c = deltas(eps_crystal, eps_medium)
    d_f = deltas(eps_fixed, eps_medium)
    return hamaker_zj_from_delta(d_c, np.broadcast_to(d_f, d_c.shape), temperature_k)


def kind_pair(eps_a, eps_m, eps_b, xi, z_grid, temperature_k):
    d_a = deltas(eps_a, eps_m)
    d_b = deltas(eps_b, eps_m)
    a_nr = float(hamaker_zj_from_delta(d_a, d_b, temperature_k))
    a_z = []
    for z in z_grid:
        r_n = retardation_r(z, xi, eps_m)
        r_n[0] = 0.0
        a_z.append(float(hamaker_zj_retarded(d_a, d_b, r_n, temperature_k)))
    return classify_sign(a_nr, a_z[0], a_z[-1]), a_nr


def kind_crystal_vs_fixed(eps_crystal, eps_fixed, eps_medium, xi, z_grid, temperature_k):
    kinds = []
    for i in range(len(eps_crystal)):
        kind, a_nr = kind_pair(eps_crystal[i], eps_medium, eps_fixed, xi, z_grid, temperature_k)
        kinds.append(kind)
    a = hamaker_crystal_vs_fixed(eps_crystal, eps_fixed, eps_medium, temperature_k)
    return np.asarray(kinds), a


def voter_ok(a, kinds, always_repulsive):
    ok = a < 0.0
    if always_repulsive:
        ok = ok & (kinds == "repulsive")
    return ok


def apply_material_filters(metas, min_density, min_atoms, min_elements):
    keep = np.ones(len(metas), dtype=bool)
    dens = np.array([m["density_g_cm3"] for m in metas], dtype=np.float64)
    n_at = np.array([m["n_atoms"] for m in metas], dtype=np.float64)
    n_el = np.array([m["n_elements"] for m in metas], dtype=np.float64)
    if min_density is not None:
        keep &= np.isfinite(dens) & (dens >= min_density)
    if min_atoms is not None:
        keep &= np.isfinite(n_at) & (n_at >= min_atoms)
    if min_elements is not None:
        keep &= np.isfinite(n_el) & (n_el >= min_elements)
    return keep, dens, n_at, n_el


def neural_split_indices(dataset, model_paths, tsenn_paths, split_ckpt_path, split):
    sets = []
    n = len(dataset)
    paths = list(model_paths)
    if split_ckpt_path and split_ckpt_path not in paths:
        paths.append(split_ckpt_path)
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        idx = [int(i) for i in get_split_indices(ckpt, n, split)]
        print(f"  {path}: {split} n={len(idx)}")
        sets.append(set(idx))
    if tsenn_paths:
        from compare_tsenn_kkr_vs_ours import build_jid_to_lmdb, tsenn_split_lmdb_indices

        jid_to_lmdb = build_jid_to_lmdb(dataset)
        for path in tsenn_paths:
            ckpt = torch.load(path, map_location="cpu")
            idx = [int(i) for i in tsenn_split_lmdb_indices(ckpt, jid_to_lmdb, split)]
            print(f"  {path}: {split} n={len(idx)}")
            sets.append(set(idx))
    if not sets:
        raise SystemExit(f"--split {split} needs --model, --tsenn, or --split_ckpt")
    common = set.intersection(*sets)
    print(f"  intersection: {len(common)}")
    if not common:
        raise SystemExit(
            f"Empty {split} intersection across checkpoints. "
            "Retrain TSENN with --align_split_ckpt, or pass --split all."
        )
    return sorted(common)


def load_voters(args, temperature_k):
    from find_repulsive_lifshitz import T_K

    if args.no_dft and not args.models and not args.tsenn:
        raise SystemExit("Need --model / --tsenn, or leave DFT on.")
    t_k = args.temperature_k if args.temperature_k is not None else T_K
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lmdb_path = str(Path(args.lmdb_path))
    xi = matsubara_xi(args.n_matsubara, t_k)

    split_ckpt_path = args.split_ckpt or (args.models[0] if args.models else None)
    split_ckpt = torch.load(split_ckpt_path, map_location="cpu") if split_ckpt_path else {}
    dataset = open_crystal_dataset(lmdb_path, split_ckpt or None, use_mace=False)
    xi_lmdb = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    print(f"split={args.split}")
    if args.split == "all":
        indices = list(range(len(dataset)))
        print(f"  all crystals n={len(indices)}")
    else:
        indices = neural_split_indices(
            dataset, args.models, args.tsenn, args.split_ckpt, args.split,
        )
    n_split = len(indices)
    indices, metas, n_skip_atoms, n_skip_elem = filter_crystal_indices(
        dataset, indices, args.min_atoms, drop_elemental=not args.keep_elemental,
    )
    if args.limit is not None:
        indices = indices[: args.limit]
        metas = metas[: args.limit]
    print(
        f"crystals: split={args.split}  kept={len(indices)}/{n_split}  "
        f"dropped n_atoms={n_skip_atoms}  elemental={n_skip_elem}"
    )
    if not indices:
        raise SystemExit("No crystals left after cell filter.")

    names = [f"{m['jid']}:{m['formula']}".strip(":") for m in metas]
    gt_kkr = collect_gt_kkr(dataset, indices, args.batch_size)
    dataset.close()

    voters = {}
    if not args.no_dft:
        voters["dft"] = to_matsubara(gt_kkr, xi_lmdb, xi)
    for path in args.models:
        print(f"predict {path} on {device}")
        pred_kkr, xi_kkr, model_type, use_log = predict_ours_eps(
            path, lmdb_path, indices, device, args.batch_size,
        )
        label = voter_label(path, model_type.lower())
        print(f"  {label}  use_log={use_log}")
        voters[label] = to_matsubara(pred_kkr, xi_kkr, xi)
    for path in args.tsenn:
        print(f"predict TSENN {path} on {device}")
        pred_kkr, xi_kkr, use_log = predict_tsenn_eps(
            path, lmdb_path, indices, device, args.batch_size,
            args.onehot_dir, refine=not args.no_refine,
        )
        if xi_kkr is None:
            if pred_kkr.shape[1] != len(xi_lmdb):
                raise SystemExit(
                    f"TSENN output F={pred_kkr.shape[1]} != LMDB F={len(xi_lmdb)} and no xi in ckpt"
                )
            xi_kkr = xi_lmdb
        label = voter_label(path, "tsenn")
        print(f"  {label}  use_log={use_log}")
        voters[label] = to_matsubara(pred_kkr, xi_kkr, xi)

    keep_mat, dens, n_at, n_el = apply_material_filters(
        metas, args.min_density, args.min_atoms, args.min_elements,
    )
    print(
        f"material cut: min_density={args.min_density}  min_atoms={args.min_atoms}  "
        f"min_elements={args.min_elements}  → {int(keep_mat.sum())}/{len(metas)}"
    )
    return {
        "voters": voters,
        "names": names,
        "metas": metas,
        "keep_mat": keep_mat,
        "dens": dens,
        "n_at": n_at,
        "n_el": n_el,
        "xi": xi,
        "temperature_k": t_k,
        "device": device,
    }
