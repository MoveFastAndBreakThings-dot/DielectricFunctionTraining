#!/usr/bin/env python3
"""
Run pretrained TSENN on OPT LMDB semiconductors → ε₂(ω) then KK → ε₁(ω).

Fully offline: reads local LMDB + gap table + checkpoint only (safe for sbatch).

Semiconductor filter: join gaps by jid (opt_gap > threshold) from
tsenn_opt/jid_gaps.jsonl — build that on the login node first:
  python build_jid_gap_table.py --from_manifest raw_opt/manifest.jsonl

Example:
  python predict_tsenn_opt.py --limit 5
  python predict_tsenn_opt.py --lmdb_path latest_data_opt/crystal_graphs.lmdb
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Union

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter
from e3nn import o3
from e3nn.io import CartesianTensor
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
TSENN_ROOT = ROOT / "external" / "TSENN"
DEFAULT_CKPT = (
    TSENN_ROOT
    / "model"
    / "revision_250909_Lmax_2_Lr_0.01_bs_8_em_128_layers_4_best.torch"
)

# Paper notebook hyperparameters (material_prediction.ipynb)
ENERGY_MIN = 0.0
ENERGY_MAX = 30.0
NSTEP = 300
R_MAX = 6.0
EM_DIM = 128
LMAX = 2
LAYERS = 4
MUL = 32
SCALE_0E = 11.6698
SCALE_2E = 2.2851
NUM_NEIGHBORS = 43.404443289593345
DEFAULT_DTYPE = torch.float64


def _ensure_tsenn_path():
    if not TSENN_ROOT.is_dir():
        raise FileNotFoundError(
            f"TSENN not found at {TSENN_ROOT}. Run ./setup_tsenn.sh first."
        )
    sys.path.insert(0, str(TSENN_ROOT))


def load_gap_table(path):
    table = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            table[row["jid"]] = row
    return table


def load_existing_pred_jids(pred_lmdb):
    done = set()
    if not Path(pred_lmdb).exists():
        return done
    env = lmdb.open(str(pred_lmdb), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        for key, _ in txn.cursor():
            if key == b"__metadata__":
                continue
            done.add(key.decode("utf-8"))
    env.close()
    return done


def iter_opt_records(lmdb_path):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        n = meta["total_samples"]
        for i in range(n):
            raw = txn.get(f"{i}".encode("utf-8"))
            if raw is None:
                break
            yield i, pickle.loads(raw)
    env.close()


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
    atoms = AseAtomsAdaptor.get_atoms(structure)
    return atoms, crystal_system


def kk_real_from_imag_tensor(omega, imag_tensor):
    """
    Vectorized Kramers–Kronig (matches TSENN notebook formula, no +1):
      ε₁(ω) = (2/π) P∫ ω' ε₂(ω') / (ω'² − ω²) dω'
    omega: (F,), imag_tensor: (F, 3, 3) → (F, 3, 3)
    """
    omega = np.asarray(omega, dtype=np.float64)
    imag = np.asarray(imag_tensor, dtype=np.float64)
    omega = np.where(omega == 0.0, np.finfo(float).eps, omega)
    w = omega[None, :]
    w0 = omega[:, None]
    denom = w * w - w0 * w0
    np.fill_diagonal(denom, np.nan)

    real = np.zeros_like(imag)
    for i in range(3):
        for j in range(i, 3):
            integrand = (w * imag[None, :, i, j]) / denom
            integrand = np.nan_to_num(integrand, nan=0.0)
            chi = (2.0 / np.pi) * np.trapezoid(integrand, omega, axis=1)
            real[:, i, j] = chi
            if i != j:
                real[:, j, i] = chi
    return real.astype(np.float32)


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

        def forward(self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
            data.z = F.relu(self.em_z(data.z))
            data.x = F.relu(self.em_x(data.x))
            output = super().forward(data)
            if self.pool:
                output = torch_scatter.scatter_mean(output, data.batch, dim=0)
            return output

    return _PeriodicNetwork


def build_model(Network, device):
    PeriodicNet = make_periodic_network(Network)
    model = PeriodicNet(
        in_dim=118,
        em_dim=EM_DIM,
        irreps_in=f"{EM_DIM}x0e",
        irreps_out=f"{NSTEP}x0e + {NSTEP}x2e",
        irreps_node_attr=f"{EM_DIM}x0e",
        layers=LAYERS,
        mul=MUL,
        lmax=LMAX,
        max_radius=R_MAX,
        num_neighbors=NUM_NEIGHBORS,
        reduce_output=True,
        dropout_prob=0.0,
        use_batch_norm=False,
    )
    model.to(device)
    return model


def build_graph_rows(records, onehots, refine):
    type_onehot, mass_onehot, dipole_onehot, radius_onehot, type_encoding = onehots
    from utils.utils_data import build_data

    omega = np.linspace(ENERGY_MIN, ENERGY_MAX, NSTEP)
    dummy_sph = np.zeros((NSTEP, 6), dtype=np.float64)
    rows = []
    meta = []
    for rec in records:
        atoms, crystal_system = record_to_ase_atoms(rec, refine=refine)
        entry = pd.Series(
            {
                "symmetrized_structure": atoms,
                "crystal_system": crystal_system,
                "sph_coefs": dummy_sph,
            }
        )
        data = build_data(
            entry,
            "sph_coefs",
            1.0,
            type_onehot,
            mass_onehot,
            dipole_onehot,
            radius_onehot,
            type_encoding,
            R_MAX,
        )
        rows.append(data)
        meta.append(
            {
                "jid": rec["jid"],
                "mpid": rec.get("mpid") or "",
                "formula": rec.get("formula") or "",
                "opt_gap": rec.get("_opt_gap"),
                "mbj_gap": rec.get("_mbj_gap"),
                "crystal_system": crystal_system,
            }
        )
    return rows, meta, omega


def predict_batch(model, graphs, device):
    loader = tg.loader.DataLoader(graphs, batch_size=len(graphs))
    cart = CartesianTensor("ij=ji")
    outs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            irreps_0e = model.irreps_out.count(o3.Irrep("0e"))
            irreps_2e = model.irreps_out.count(o3.Irrep("2e")) * 5
            out_dim = irreps_0e
            output_0e = output[:, :irreps_0e] * SCALE_0E
            output_2e = (
                output[:, irreps_0e : irreps_0e + irreps_2e]
                .contiguous()
                .view(output.shape[0], out_dim, 5)
                * SCALE_2E
            )
            sph = torch.cat([output_0e.unsqueeze(2), output_2e], dim=2)
            eps2 = cart.to_cartesian(sph).cpu().numpy().astype(np.float32)
            outs.append(eps2)
    return np.concatenate(outs, axis=0)


def write_predictions(pred_lmdb, records_out, omega):
    Path(pred_lmdb).parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(pred_lmdb),
        map_size=1024**4,
        subdir=False,
        lock=True,
    )
    with env.begin(write=True) as txn:
        for rec in records_out:
            txn.put(rec["jid"].encode("utf-8"), pickle.dumps(rec))
        meta_raw = txn.get(b"__metadata__")
        meta = pickle.loads(meta_raw) if meta_raw else {"n_written": 0}
        meta["n_written"] = meta.get("n_written", 0) + len(records_out)
        meta["omega"] = np.asarray(omega, dtype=np.float32)
        meta["energy_min"] = ENERGY_MIN
        meta["energy_max"] = ENERGY_MAX
        meta["nstep"] = NSTEP
        txn.put(b"__metadata__", pickle.dumps(meta))
    env.close()


def main():
    parser = argparse.ArgumentParser(description="TSENN inference on OPT semiconductors")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--gap_table", default="tsenn_opt/jid_gaps.jsonl")
    parser.add_argument("--out_lmdb", default="tsenn_opt/predictions.lmdb")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--gap_threshold", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Max new semiconductors")
    parser.add_argument("--no_refine", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--onehot_dir", default=None)
    args = parser.parse_args()

    _ensure_tsenn_path()
    torch.set_default_dtype(DEFAULT_DTYPE)

    from utils.utils_data import save_or_load_onehot
    from utils.utils_model_full_tensor import Network

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Place the Figshare models tar in the repo and run ./setup_tsenn.sh"
        )

    gap_path = Path(args.gap_table)
    if not gap_path.is_file():
        raise FileNotFoundError(
            f"Gap table missing: {gap_path}\n"
            "On login node (not sbatch): "
            "python build_jid_gap_table.py --from_manifest raw_opt/manifest.jsonl"
        )
    gaps = load_gap_table(gap_path)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    onehot_dir = args.onehot_dir or str(TSENN_ROOT / "onehot_data")
    # save_or_load_onehot uses paths relative to CWD in their code; pass absolute
    onehots = save_or_load_onehot(save_path=onehot_dir)

    model = build_model(Network, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state"])
    model.pool = True
    model.eval()
    print(f"Loaded {ckpt_path.name}")

    done = load_existing_pred_jids(args.out_lmdb)
    print(f"Already predicted: {len(done)}")

    candidates = []
    n_miss_gap = 0
    n_metal = 0
    for _, data in tqdm(iter_opt_records(args.lmdb_path), desc="scan-lmdb"):
        jid = data["jid"]
        if jid in done:
            continue
        ginfo = gaps.get(jid)
        if ginfo is None or ginfo.get("opt_gap") is None:
            n_miss_gap += 1
            continue
        opt_gap = float(ginfo["opt_gap"])
        if opt_gap <= args.gap_threshold:
            n_metal += 1
            continue
        data = dict(data)
        data["_opt_gap"] = opt_gap
        data["_mbj_gap"] = ginfo.get("mbj_gap")
        candidates.append(data)
        if args.limit is not None and len(candidates) >= args.limit:
            break

    print(
        f"To predict: {len(candidates)} "
        f"(skipped missing_gap={n_miss_gap}, metal/low_gap={n_metal})"
    )
    if not candidates:
        return

    refine = not args.no_refine
    omega = None
    for start in tqdm(range(0, len(candidates), args.batch_size), desc="predict"):
        batch_recs = candidates[start : start + args.batch_size]
        graphs, meta, omega = build_graph_rows(batch_recs, onehots, refine=refine)
        eps2 = predict_batch(model, graphs, device)
        out_recs = []
        for i, m in enumerate(meta):
            eps1 = kk_real_from_imag_tensor(omega, eps2[i])
            out_recs.append(
                {
                    "jid": m["jid"],
                    "mpid": m["mpid"],
                    "formula": m["formula"],
                    "opt_gap": m["opt_gap"],
                    "mbj_gap": m["mbj_gap"],
                    "crystal_system": m["crystal_system"],
                    "omega": np.asarray(omega, dtype=np.float32),
                    "eps2": eps2[i],
                    "eps1": eps1,
                }
            )
        write_predictions(args.out_lmdb, out_recs, omega)

    print(f"Wrote predictions → {args.out_lmdb}")


if __name__ == "__main__":
    main()
