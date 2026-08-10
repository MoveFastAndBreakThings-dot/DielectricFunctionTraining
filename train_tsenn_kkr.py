#!/usr/bin/env python3
"""
Train TSENN architecture from scratch to predict ε_xx(iξ) on our KKR grid (~500 pts).

Same equivariant Network + paper hypers (layers=4, em=128, lmax=2, …), fresh weights,
scalar head irreps_out = "{N}x0e" (xx only — matches FrequencyCGCNN/MACE targets).

Requires: external/TSENN setup, torch-geometric, e3nn, ase, mendeleev, scipy.

Example:
  python train_tsenn_kkr.py --limit 64 --epochs 2
  python train_tsenn_kkr.py \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb \\
      --models_dir models_latest_opt --model_name tsenn_kkr_xx_latest_opt
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from tqdm import tqdm

import torch_load_compat  # noqa: F401

import torch_geometric as tg
import torch_scatter
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

ROOT = Path(__file__).resolve().parent
TSENN_ROOT = ROOT / "external" / "TSENN"

# Paper checkpoint naming hypers (revision_250909_*_layers_4_*)
R_MAX = 6.0
EM_DIM = 128
LMAX = 2
LAYERS = 4
MUL = 32
DEFAULT_DTYPE = torch.float64


def _ensure_tsenn_path():
    if not TSENN_ROOT.is_dir():
        raise FileNotFoundError(f"TSENN not found at {TSENN_ROOT}. Run ./setup_tsenn.sh first.")
    sys.path.insert(0, str(TSENN_ROOT))


def record_to_ase_atoms(data, refine=True, symprec=1e-2):
    structure = Structure(
        Lattice(data["cell"]),
        [int(z) for z in data["atomic_numbers"]],
        data["positions"],
        coords_are_cartesian=True,
    )
    if refine:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        structure = sga.get_refined_structure()
    crystal_system = SpacegroupAnalyzer(structure, symprec=symprec).get_crystal_system()
    return AseAtomsAdaptor.get_atoms(structure), crystal_system


def kkr_xx_from_record(rec, xi_dst):
    """Return ε_xx(iξ) on shared xi_dst, shape (F,)."""
    xi_src = np.asarray(rec["kkr_xi"], dtype=np.float64)
    eps = np.asarray(rec["kkr_targets"]["kkr_xx"], dtype=np.float64)

    if len(xi_src) == len(xi_dst) and np.allclose(xi_src, xi_dst):
        return eps

    interp = interp1d(
        xi_src,
        eps,
        kind="linear",
        bounds_error=False,
        fill_value=(float(eps[0]), 1.0),
    )
    return interp(xi_dst).astype(np.float64)


def make_periodic_network(Network):
    class _PeriodicNetwork(Network):
        def __init__(self, in_dim, em_dim, **kwargs):
            self.pool = False
            if kwargs.get("reduce_output", True):
                kwargs = dict(kwargs)
                kwargs["reduce_output"] = False
                self.pool = True
            super().__init__(**kwargs)
            self.em_z = nn.Linear(in_dim, em_dim)
            self.em_x = nn.Linear(in_dim, em_dim)

        def forward(self, data):
            data.z = F.relu(self.em_z(data.z))
            data.x = F.relu(self.em_x(data.x))
            output = super().forward(data)
            if self.pool:
                output = torch_scatter.scatter_mean(output, data.batch, dim=0)
            return output

    return _PeriodicNetwork


def train_val_test_split(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        idx[:n_train].tolist(),
        idx[n_train : n_train + n_val].tolist(),
        idx[n_train + n_val :].tolist(),
    )


def get_neighbors(graphs, indices):
    n = []
    for i in indices:
        pos = graphs[i].pos
        edge_index = graphs[i].edge_index
        for a in range(pos.shape[0]):
            n.append(int((edge_index[0] == a).sum()))
    return np.asarray(n, dtype=np.float64)


def batch_target(y, nstep):
    """build_data stores y as (1, F); batched → (B, 1, F) or (B, F)."""
    if y.dim() == 3:
        return y.view(y.shape[0], nstep)
    return y.view(y.shape[0], -1)[:, :nstep]


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, nstep):
    model.eval()
    total = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        target = batch_target(batch.y, nstep)
        total += loss_fn(pred, target).item()
        n_batches += 1
    return total / max(n_batches, 1)


def load_lmdb_rows(lmdb_path, xi_dst, refine, limit, gap_threshold, gap_table):
    gaps = {}
    if gap_table is not None:
        with open(gap_table) as f:
            for line in f:
                row = json.loads(line)
                gaps[row["jid"]] = row

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    rows = []
    skipped = 0
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        n = int(meta["total_samples"])
        for i in tqdm(range(n), desc="scan-lmdb"):
            rec = pickle.loads(txn.get(f"{i}".encode("utf-8")))
            if "kkr_targets" not in rec or "kkr_xi" not in rec:
                skipped += 1
                continue
            if "kkr_xx" not in rec["kkr_targets"]:
                skipped += 1
                continue
            if gap_threshold is not None:
                g = gaps.get(rec["jid"])
                gap = None if g is None else g.get("opt_gap")
                if gap is None:
                    gap = rec.get("opt_gap")
                if gap is None or float(gap) <= gap_threshold:
                    skipped += 1
                    continue
            try:
                atoms, crystal_system = record_to_ase_atoms(rec, refine=refine)
                eps_xx = kkr_xx_from_record(rec, xi_dst)
            except Exception:
                skipped += 1
                continue
            rows.append(
                {
                    "jid": rec["jid"],
                    "mpid": rec.get("mpid") or "",
                    "formula": rec.get("formula") or "",
                    "symmetrized_structure": atoms,
                    "crystal_system": crystal_system,
                    "eps_xx": eps_xx,
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    env.close()
    print(f"Loaded {len(rows)} materials (skipped {skipped})")
    return rows, meta


def main():
    parser = argparse.ArgumentParser(description="Train TSENN from scratch on KKR ε_xx(iξ)")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--models_dir", default="models_latest_opt")
    parser.add_argument("--model_name", default="tsenn_kkr_xx_latest_opt")
    parser.add_argument("--onehot_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layers", type=int, default=LAYERS)
    parser.add_argument("--em_dim", type=int, default=EM_DIM)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument(
        "--use_log",
        action="store_true",
        help="Train on log(ε_xx); matches train_mace_frequency --use_log",
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=None,
        help="If set, keep only opt_gap > threshold (needs --gap_table or LMDB opt_gap)",
    )
    parser.add_argument("--gap_table", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    _ensure_tsenn_path()
    torch.set_default_dtype(DEFAULT_DTYPE)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from utils.utils_data import build_data, save_or_load_onehot
    from utils.utils_model_full_tensor import Network

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        meta0 = pickle.loads(txn.get(b"__metadata__"))
    env.close()
    if meta0.get("kkr_xi_array") is None:
        raise ValueError("LMDB missing kkr_xi_array; re-run process_optics_lmdb.py")
    xi_dst = np.asarray(meta0["kkr_xi_array"], dtype=np.float64)
    nstep = len(xi_dst)
    print(f"KKR grid: {nstep} ξ points, ξ∈[{xi_dst.min():.4g}, {xi_dst.max():.4g}] eV")

    rows, _ = load_lmdb_rows(
        args.lmdb_path,
        xi_dst,
        refine=not args.no_refine,
        limit=args.limit,
        gap_threshold=args.gap_threshold,
        gap_table=args.gap_table,
    )
    if len(rows) < 10:
        raise RuntimeError(f"Too few samples ({len(rows)}); check LMDB / filters")

    stack = np.stack([r["eps_xx"] for r in rows], axis=0)
    if args.use_log:
        stack = np.log(np.clip(stack, 1e-12, None))
        scale = 1.0
        print("target=log(ε_xx), scale=1")
    else:
        scale = float(np.median(np.max(np.abs(stack), axis=1)))
        scale = max(scale, 1e-12)
        stack = stack / scale
        print(f"target=ε_xx / scale, scale={scale:.6g}")

    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    onehots = save_or_load_onehot(save_path=onehot_dir)
    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = onehots

    graphs = []
    for i, r in enumerate(tqdm(rows, desc="build-graphs")):
        entry = pd.Series(
            {
                "symmetrized_structure": r["symmetrized_structure"],
                "crystal_system": r["crystal_system"],
                "eps_xx": stack[i],
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
                R_MAX,
            )
        )

    idx_train, idx_valid, idx_test = train_val_test_split(len(graphs), seed=args.seed)
    print(f"split train/val/test: {len(idx_train)}/{len(idx_valid)}/{len(idx_test)}")

    loader_train = tg.loader.DataLoader(
        [graphs[i] for i in idx_train],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    loader_valid = tg.loader.DataLoader(
        [graphs[i] for i in idx_valid],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    n_train = get_neighbors(graphs, idx_train)
    num_neighbors = float(n_train.mean()) if len(n_train) else 20.0
    print(f"mean num_neighbors (train)={num_neighbors:.3f}")

    PeriodicNet = make_periodic_network(Network)
    model = PeriodicNet(
        in_dim=118,
        em_dim=args.em_dim,
        irreps_in=f"{args.em_dim}x0e",
        irreps_out=f"{nstep}x0e",
        irreps_node_attr=f"{args.em_dim}x0e",
        layers=args.layers,
        mul=MUL,
        lmax=LMAX,
        max_radius=R_MAX,
        num_neighbors=num_neighbors,
        reduce_output=True,
        dropout_prob=args.dropout,
        use_batch_norm=False,
    )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=10, T_mult=1, eta_min=0.0
    )
    loss_fn = nn.L1Loss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}  irreps_out={nstep}x0e (xx only)")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f"{args.model_name}_best.torch"
    latest_path = models_dir / f"{args.model_name}.torch"
    best_valid = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_batches = 0
        for batch in tqdm(loader_train, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            batch = batch.to(device)
            pred = model(batch)
            target = batch_target(batch.y, nstep)
            loss = loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1

        train_loss = evaluate(model, loader_train, device, loss_fn, nstep)
        valid_loss = evaluate(model, loader_valid, device, loss_fn, nstep)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_batch_loss": running / max(n_batches, 1),
                "lr": opt.param_groups[0]["lr"],
                "wall": time.time() - t0,
            }
        )
        print(
            f"epoch {epoch+1:4d}  train={train_loss:.5f}  valid={valid_loss:.5f}  "
            f"lr={opt.param_groups[0]['lr']:.3e}"
        )

        payload = {
            "state": model.state_dict(),
            "history": history,
            "scale": scale,
            "n_frequencies": nstep,
            "kkr_xi_array": xi_dst,
            "target": "kkr_eps_xx",
            "target_components": ["xx"],
            "use_log": args.use_log,
            "model_type": "TSENN",
            "r_max": R_MAX,
            "em_dim": args.em_dim,
            "layers": args.layers,
            "mul": MUL,
            "lmax": LMAX,
            "num_neighbors": num_neighbors,
            "dropout": args.dropout,
            "idx_train": idx_train,
            "idx_valid": idx_valid,
            "idx_test": idx_test,
            "jids": [r["jid"] for r in rows],
            "lmdb_path": args.lmdb_path,
        }
        torch.save(payload, latest_path)
        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(payload, best_path)
            print(f"  saved best → {best_path} (valid={best_valid:.5f})")

    print(f"Done. best_valid={best_valid:.5f}  latest={latest_path}")


if __name__ == "__main__":
    main()
