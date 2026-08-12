#!/usr/bin/env python3
"""
Compare TSENN-KKR (xx) vs one FrequencyCGCNN/MACE model on the same OPT/MBJ LMDB split.

Reports both:
  1) Dielectric ε_xx(iξ) — MAE and mean |log ε| on the shared KKR grid
  2) Self-Hamaker A — MAE/RMSE/Pearson + relative-error percentiles (zJ)

Logic mirrors compare_hamaker_constants.py + compare_epsilon_base_vs_meta.py,
with a TSENN-KKR loader/predictor (train_tsenn_kkr.py checkpoints).

Usage:
  python compare_tsenn_kkr_vs_ours.py \\
      --tsenn models_latest_opt/tsenn_kkr_xx_latest_opt_best.torch \\
      --ours  models_latest_opt/cgcnn_kkr_latest_opt.pt \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb

  python compare_tsenn_kkr_vs_ours.py --tsenn ... --ours ... --split test --limit 64
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch_load_compat  # noqa: F401

import torch_geometric as tg

from compare_hamaker_constants import (
    K_B_EV,
    T_K,
    get_split_indices,
    hamaker_constant_zj,
    report_hamaker_group,
)
from eval_hamaker_pair_percentiles import PERCENTILES, print_percentile_table
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward
from model_loader import load_model
from train_tsenn_kkr import (
    DEFAULT_DTYPE,
    EM_DIM,
    LAYERS,
    LMAX,
    MUL,
    R_MAX,
    TSENN_ROOT,
    _ensure_tsenn_path,
    make_periodic_network,
    record_to_ase_atoms,
)

OUT_DIR = "tsenn_kkr_vs_ours"


def summarize_eps(name, pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    abs_err = np.abs(pred - true)
    log_err = np.abs(np.log(pred.clip(1e-30)) - np.log(true.clip(1e-30)))
    mae = float(abs_err.mean())
    mean_log = float(log_err.mean())
    per_crystal_mae = abs_err.mean(axis=1)
    per_crystal_log = log_err.mean(axis=1)
    print(f"\n--- {name}: ε_xx(iξ) ---")
    print(f"  MAE(|ε_pred−ε_gt|)     = {mae:.6f}")
    print(f"  mean |log ε err|       = {mean_log:.6f}")
    print_percentile_table(f"{name}: per-crystal MAE", per_crystal_mae, PERCENTILES)
    print_percentile_table(f"{name}: per-crystal |log ε|", per_crystal_log, PERCENTILES)
    return {
        "mae": mae,
        "mean_abs_log": mean_log,
        "per_crystal_mae_median": float(np.median(per_crystal_mae)),
        "per_crystal_abs_log_median": float(np.median(per_crystal_log)),
    }


def summarize_hamaker(name, a_pred, a_true):
    a_pred = np.asarray(a_pred, dtype=np.float64)
    a_true = np.asarray(a_true, dtype=np.float64)
    rel = 100.0 * (a_pred - a_true) / (np.abs(a_true) + 1e-30)
    abs_rel = np.abs(rel)
    mae = float(np.mean(np.abs(a_pred - a_true)))
    rmse = float(np.sqrt(np.mean((a_pred - a_true) ** 2)))
    r = float(np.corrcoef(a_pred, a_true)[0, 1]) if len(a_true) > 1 else float("nan")
    print(f"\n--- {name}: Hamaker A ---")
    print(f"  MAE(ΔA)   = {mae:.4f} zJ")
    print(f"  RMSE(ΔA)  = {rmse:.4f} zJ")
    print(f"  Pearson r = {r:.4f}")
    print_percentile_table(f"{name}: signed rel %", rel, PERCENTILES, unit="%")
    print_percentile_table(f"{name}: abs rel %", abs_rel, PERCENTILES, unit="%")
    return {
        "n": int(len(a_true)),
        "mae_zj": mae,
        "rmse_zj": rmse,
        "pearson_r": r,
        "abs_rel_median_pct": float(np.median(abs_rel)) if len(abs_rel) else None,
        "abs_rel_mean_pct": float(np.mean(abs_rel)) if len(abs_rel) else None,
    }


def load_tsenn_kkr(ckpt_path, device):
    _ensure_tsenn_path()
    from utils.utils_model_full_tensor import Network

    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("model_type") != "TSENN" or ckpt.get("target") != "kkr_eps_xx":
        raise ValueError(
            f"{ckpt_path} does not look like a train_tsenn_kkr.py checkpoint "
            f"(model_type={ckpt.get('model_type')}, target={ckpt.get('target')})"
        )
    nstep = int(ckpt["n_frequencies"])
    em_dim = int(ckpt.get("em_dim", EM_DIM))
    layers = int(ckpt.get("layers", LAYERS))
    mul = int(ckpt.get("mul", MUL))
    lmax = int(ckpt.get("lmax", LMAX))
    r_max = float(ckpt.get("r_max", R_MAX))
    num_neighbors = float(ckpt.get("num_neighbors", 20.0))
    dropout = float(ckpt.get("dropout", 0.0))

    PeriodicNet = make_periodic_network(Network)
    model = PeriodicNet(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=f"{em_dim}x0e",
        irreps_out=f"{nstep}x0e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=layers,
        mul=mul,
        lmax=lmax,
        max_radius=r_max,
        num_neighbors=num_neighbors,
        reduce_output=True,
        dropout_prob=dropout,
        use_batch_norm=False,
    )
    model.load_state_dict(ckpt["state"])
    model.to(device)
    model.pool = True
    model.eval()
    return model, ckpt


def read_lmdb_records(lmdb_path, indices):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    out = []
    with env.begin() as txn:
        for i in indices:
            raw = txn.get(f"{i}".encode("utf-8"))
            if raw is None:
                raise KeyError(f"Missing LMDB key {i}")
            out.append(pickle.loads(raw))
    env.close()
    return out


@torch.no_grad()
def predict_ours(model, model_type, loader, device, use_log):
    preds, trues = [], []
    for inputs, target, _ in loader:
        target = target.to(device)
        pred = model_forward(model, model_type, inputs, device, use_log=use_log)
        preds.append(pred.cpu().numpy())
        trues.append(target[:, 0, :].cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def predict_tsenn(model, ckpt, records, device, batch_size, onehot_dir, refine):
    from utils.utils_data import build_data, save_or_load_onehot

    nstep = int(ckpt["n_frequencies"])
    use_log = bool(ckpt.get("use_log", False))
    scale = float(ckpt.get("scale", 1.0))
    onehots = save_or_load_onehot(save_path=onehot_dir)
    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = onehots
    dummy = np.zeros(nstep, dtype=np.float64)

    graphs = []
    for rec in tqdm(records, desc="tsenn-graphs"):
        atoms, crystal_system = record_to_ase_atoms(rec, refine=refine)
        entry = pd.Series(
            {
                "symmetrized_structure": atoms,
                "crystal_system": crystal_system,
                "eps_xx": dummy,
            }
        )
        graphs.append(
            build_data(
                entry,
                "eps_xx",
                1.0,
                type_onehot,
                mass_onehot,
                dipole_onehot,
                radius_onehot,
                type_encoding,
                float(ckpt.get("r_max", R_MAX)),
            )
        )

    preds = []
    loader = tg.loader.DataLoader(graphs, batch_size=batch_size)
    for batch in tqdm(loader, desc="tsenn-predict"):
        batch = batch.to(device)
        out = model(batch)
        if use_log:
            out = torch.exp(out)
        else:
            out = out * scale
        preds.append(out.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="TSENN-KKR xx vs FrequencyCGCNN/MACE: ε(iξ) + Hamaker on LMDB",
    )
    parser.add_argument("--tsenn", required=True, help="train_tsenn_kkr .torch checkpoint")
    parser.add_argument("--ours", required=True, help="FrequencyCGCNN/MACE .pt checkpoint")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--split_from",
        default="ours",
        choices=("ours", "tsenn"),
        help="Which checkpoint supplies train/val/test indices (default: ours)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--metal_eps_threshold", type=float, default=50.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # CGCNN/MACE + LMDB are float32; only switch to float64 for TSENN below.
    torch.set_default_dtype(torch.float32)

    print(f"device={device}")
    ckpt_t_meta = torch.load(args.tsenn, map_location="cpu")
    if ckpt_t_meta.get("model_type") != "TSENN" or ckpt_t_meta.get("target") != "kkr_eps_xx":
        raise ValueError(
            f"{args.tsenn} does not look like a train_tsenn_kkr.py checkpoint "
            f"(model_type={ckpt_t_meta.get('model_type')}, target={ckpt_t_meta.get('target')})"
        )

    ckpt_o = torch.load(args.ours, map_location=device)
    model_o, model_type_o = load_model(ckpt_o, device)
    use_log_o = bool(ckpt_o.get("use_log", False))
    use_mace = model_type_o == "MACE"
    target_components = tuple(ckpt_o.get("target_components", ("xx",)))

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=ckpt_o.get("metal_gap_threshold"),
        use_mace=use_mace,
        mace_r_max=ckpt_o.get("mace_r_max"),
    )
    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    if int(ckpt_t_meta["n_frequencies"]) != len(xi):
        raise ValueError(
            f"TSENN n_frequencies={ckpt_t_meta['n_frequencies']} != LMDB ξ length={len(xi)}"
        )

    if args.split_from == "ours":
        split_idx = list(get_split_indices(ckpt_o, len(dataset), args.split))
    else:
        # TSENN stores indices into its filtered jid list — map jids → LMDB indices
        jids = ckpt_t_meta.get("jids")
        key = {"train": "idx_train", "val": "idx_valid", "test": "idx_test"}[args.split]
        local_idx = ckpt_t_meta.get(key)
        if not jids or local_idx is None:
            raise ValueError("TSENN checkpoint missing jids / split indices")
        jid_to_lmdb = {}
        for i in range(len(dataset)):
            jid_to_lmdb[dataset.get_sample_info(i)["jid"]] = i
        split_idx = []
        missing = 0
        for li in local_idx:
            jid = jids[li]
            if jid not in jid_to_lmdb:
                missing += 1
                continue
            split_idx.append(jid_to_lmdb[jid])
        if missing:
            print(f"warning: {missing} TSENN split jids not in LMDB")
        if not split_idx:
            raise RuntimeError("No overlapping jids for TSENN split")

    if args.limit is not None:
        split_idx = split_idx[: args.limit]

    print("=" * 64)
    print(f"TSENN : {args.tsenn}")
    print(
        f"  use_log={ckpt_t_meta.get('use_log')}  scale={ckpt_t_meta.get('scale')}  "
        f"layers={ckpt_t_meta.get('layers')}"
    )
    print(f"Ours  : {args.ours}  [{model_type_o}]  use_log={use_log_o}")
    print(f"Split : {args.split} from {args.split_from}  n={len(split_idx)}")
    print(f"LMDB  : {args.lmdb_path}  F={len(xi)}")
    print("=" * 64)

    loader = DataLoader(
        Subset(dataset, split_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=use_mace),
    )
    pred_o, true = predict_ours(model_o, model_type_o, loader, device, use_log_o)
    del model_o
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records = read_lmdb_records(args.lmdb_path, split_idx)
    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    torch.set_default_dtype(DEFAULT_DTYPE)
    model_t, ckpt_t = load_tsenn_kkr(args.tsenn, device)
    pred_t = predict_tsenn(
        model_t,
        ckpt_t,
        records,
        device,
        args.batch_size,
        onehot_dir,
        refine=not args.no_refine,
    )
    torch.set_default_dtype(torch.float32)

    infos = [dataset.get_sample_info(i) for i in split_idx]
    dataset.close()

    stats_eps_t = summarize_eps("TSENN-KKR", pred_t, true)
    stats_eps_o = summarize_eps("Ours", pred_o, true)

    xi_step = 2.0 * np.pi * K_B_EV * args.temperature_k
    hamaker_kw = dict(
        t_k=args.temperature_k,
        xi_step_ev=xi_step,
        n_max=args.n_matsubara,
    )
    a_true, a_t, a_o, eps_low = [], [], [], []
    for i in range(len(true)):
        eps_gt = true[i]
        eps_low.append(float(eps_gt[0]))
        a_true.append(hamaker_constant_zj(xi, eps_gt, **hamaker_kw))
        a_t.append(hamaker_constant_zj(xi, pred_t[i], **hamaker_kw))
        a_o.append(hamaker_constant_zj(xi, pred_o[i], **hamaker_kw))
    a_true = np.asarray(a_true)
    a_t = np.asarray(a_t)
    a_o = np.asarray(a_o)
    eps_low = np.asarray(eps_low)
    insulator = eps_low <= args.metal_eps_threshold

    print("\n" + "=" * 64)
    print("Hamaker (all)")
    print("=" * 64)
    stats_h_t = summarize_hamaker("TSENN-KKR", a_t, a_true)
    stats_h_o = summarize_hamaker("Ours", a_o, a_true)

    report_hamaker_group(
        "Insulators only — TSENN-KKR",
        a_true[insulator],
        a_t[insulator],
        args.metal_eps_threshold,
    )
    report_hamaker_group(
        "Insulators only — Ours",
        a_true[insulator],
        a_o[insulator],
        args.metal_eps_threshold,
    )

    # Side-by-side winner counts
    eps_log_t = np.abs(np.log(pred_t.clip(1e-30)) - np.log(true.clip(1e-30))).mean(axis=1)
    eps_log_o = np.abs(np.log(pred_o.clip(1e-30)) - np.log(true.clip(1e-30))).mean(axis=1)
    ham_abs_t = np.abs(a_t - a_true)
    ham_abs_o = np.abs(a_o - a_true)
    print("\n" + "=" * 64)
    print("Head-to-head (lower error wins)")
    print("=" * 64)
    print(
        f"  ε |log| : TSENN better {(eps_log_t < eps_log_o).sum()}/"
        f"{len(eps_log_t)}  |  Ours better {(eps_log_o < eps_log_t).sum()}"
    )
    print(
        f"  Hamaker : TSENN better {(ham_abs_t < ham_abs_o).sum()}/"
        f"{len(ham_abs_t)}  |  Ours better {(ham_abs_o < ham_abs_t).sum()}"
    )

    stem = f"{Path(args.tsenn).stem}_vs_{Path(args.ours).stem}_{args.split}"
    csv_path = out_dir / f"{stem}_per_crystal.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "jid",
                "mpid",
                "formula",
                "eps_low",
                "log_err_tsenn",
                "log_err_ours",
                "a_true_zj",
                "a_tsenn_zj",
                "a_ours_zj",
                "abs_da_tsenn",
                "abs_da_ours",
            ]
        )
        for i, info in enumerate(infos):
            w.writerow(
                [
                    info.get("jid", ""),
                    info.get("mpid", ""),
                    info.get("formula", ""),
                    eps_low[i],
                    eps_log_t[i],
                    eps_log_o[i],
                    a_true[i],
                    a_t[i],
                    a_o[i],
                    ham_abs_t[i],
                    ham_abs_o[i],
                ]
            )

    payload = {
        "tsenn": args.tsenn,
        "ours": args.ours,
        "lmdb_path": args.lmdb_path,
        "split": args.split,
        "split_from": args.split_from,
        "n": len(split_idx),
        "n_frequencies": len(xi),
        "eps_tsenn": stats_eps_t,
        "eps_ours": stats_eps_o,
        "hamaker_tsenn": stats_h_t,
        "hamaker_ours": stats_h_o,
        "head_to_head": {
            "eps_tsenn_better": int((eps_log_t < eps_log_o).sum()),
            "eps_ours_better": int((eps_log_o < eps_log_t).sum()),
            "ham_tsenn_better": int((ham_abs_t < ham_abs_o).sum()),
            "ham_ours_better": int((ham_abs_o < ham_abs_t).sum()),
        },
    }
    json_path = Path(args.out_json) if args.out_json else out_dir / f"{stem}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
