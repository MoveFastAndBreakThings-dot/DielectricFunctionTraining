#!/usr/bin/env python3
"""
Find A | medium | B stacks with repulsive Lifshitz–van der Waals forces.

Mismatch condition (each Matsubara mode n):
    (ε_A(iξ_n) − ε_m(iξ_n)) (ε_B(iξ_n) − ε_m(iξ_n)) < 0
i.e. ε_m sits between ε_A and ε_B. Then Δ_Am Δ_Bm < 0 and that mode is repulsive.

Distance:
  * Magnitude always depends on z. Parallel-plate force density F ∝ A / z^3
    (non-retarded) or steeper once retardation cuts high-ξ modes.
  * Sign can depend on z. If mismatch holds only at low ξ (typical liquids),
    large-z Casimir repulsion can flip to small-z vdW attraction. Full-spectrum
    mismatch (e.g. Vac/Gr/Au in Vagli & Tian 2025) stays repulsive at all z.
  * Vacuum as medium: ε_m = 1 and ε(iξ) ≥ 1 for all passive dielectrics, so
    no planar A|Vac|B repulsion exists.

Built-in ε(iξ): Ninham–Parsegian oscillators from van Zwol & Palasantzas,
Phys. Rev. A 81, 062502 (2010), plus a Drude gold.

Primary use is a trained FrequencyCGCNN / MACE checkpoint: predict ε_xx(iξ)
for crystals, then screen A|medium|B. Media default to bromobenzene and
diiodomethane (high-n Casimir liquids). Bodies default to predicted crystals
+ gold, after dropping 1-atom and single-element JARVIS cells.

Usage:
  python find_repulsive_lifshitz.py \\
      --model_path models_latest_opt/cgcnn_kkr_latest_opt.pt \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb
  python find_repulsive_lifshitz.py --model_path ... --lmdb_path ... \\
      --media bromobenzene,diiodomethane,benzene --bodies crystals,gold --compare_gt
  python find_repulsive_lifshitz.py --media vacuum --bodies gold,silica,PTFE

All A_nr<0 pairs are written to --out_pairs (npz) and per-material density /
ε(0) to --out_materials (csv). The printed table is only --top_k.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import torch_load_compat  # noqa: F401

from compare_hamaker_constants import get_split_indices
from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from model_forward import model_forward
from model_loader import load_model

K_B_J = 1.380649e-23
K_B_EV = 8.617333e-5
T_K = 300.0
ZJ_PER_J = 1e21
HBARC_EV_NM = 197.3269788  # ħc in eV·nm

# van Zwol & Palasantzas PRA 81, 062502 (2010) Table I: C_i then ω_i (eV).
# Polar liquids get a Debye term so ε(0) matches the tabulated ε0 (oscillators
# are fitted on 10^{-2}–10^2 eV and miss the microwave).
_OSC = {
    "PTFE": {
        "C": [9.30e-3, 1.83e-2, 1.39e-1, 1.12e-1, 1.95e-1, 4.38e-1, 1.06e-1, 3.86e-2],
        "w": [3.00e-4, 7.60e-3, 5.57e-2, 1.26e-1, 6.71, 18.6, 42.1, 77.6],
        "eps0": 2.1,
    },
    "silica": {
        "C": [0.784, 0.203, 0.417, 0.393, 5.01e-2, 0.820, 0.217, 5.50e-2],
        "w": [4.11e-2, 0.112, 0.112, 0.111, 14.5, 17.0, 8.14, 91.6],
        "eps0": 4.0,
    },
    "polystyrene": {
        "C": [1.21e-2, 2.19e-2, 1.79e-2, 3.06e-2, 0.303, 0.623, 0.325, 3.31e-2],
        "w": [1.00e-3, 1.32e-2, 3.88, 0.131, 5.99, 10.2, 18.8, 51.5],
        "eps0": 2.45,
    },
    "water": {
        "C": [1.43, 9.74, 2.16, 0.532, 0.389, 0.265, 0.136],
        "w": [2.29e-2, 8.77e-4, 4.93e-3, 0.103, 9.50, 20.9, 26.4],
        "eps0": 78.7,
    },
    "bromobenzene": {
        "C": [5.44e-2, 1.84e-2, 4.75e-2, 0.532, 0.645, 0.240, 9.27e-3],
        "w": [5.02e-3, 3.09e-2, 0.111, 6.75, 13.3, 24.0, 99.9],
        "eps0": 5.37,
    },
    "ethanol": {
        "C": [0.957, 1.62, 0.140, 0.126, 0.416, 0.244, 7.10e-2],
        "w": [1.62e-3, 4.32e-3, 0.112, 6.87, 15.2, 15.6, 43.8],
        "eps0": 24.8,
    },
    "cyclohexane": {
        "C": [1.51e-2, 0.134, 0.517, 0.305, 4.65e-2],
        "w": [0.216, 8.03, 10.9, 18.4, 41.3],
        "eps0": 2.01,
    },
    "benzene": {
        "C": [2.32e-2, 6.99e-3, 8.51e-2, 0.257, 0.659, 0.226, 1.43e-2],
        "w": [0.876, 6.71, 4.48, 17.0, 8.48, 23.3, 70.1],
        "eps0": 2.273,
    },
    "diiodomethane": {
        "C": [1.94e-2, 0.925, 0.922, 8.59e-2, 9.61e-2],
        "w": [9.38e-2, 7.46, 13.3, 58.5, 13.7],
        "eps0": 5.318,
    },
}

AU_WP_EV = 9.0
AU_GAMMA_EV = 0.035
DEBYE_W_EV = 1.0e-4


def eps_oscillator(xi, C, w, eps0=None):
    xi = np.asarray(xi, dtype=np.float64)
    e = np.ones_like(xi, dtype=np.float64)
    for c, omega in zip(C, w):
        e = e + c / (1.0 + (xi / omega) ** 2)
    if eps0 is None:
        return e
    extra = float(eps0) - (1.0 + float(np.sum(C)))
    if extra > 0.0:
        e = e + extra / (1.0 + xi / DEBYE_W_EV)
    return e


def eps_gold(xi):
    xi = np.asarray(xi, dtype=np.float64)
    x = np.maximum(xi, 1e-12)
    return 1.0 + AU_WP_EV**2 / (x * (x + AU_GAMMA_EV))


def eps_vacuum(xi):
    return np.ones_like(np.asarray(xi, dtype=np.float64))


def builtin_eps_fn(name):
    if name == "vacuum":
        return eps_vacuum
    if name == "gold":
        return eps_gold
    spec = _OSC[name]
    return lambda xi, spec=spec: eps_oscillator(xi, spec["C"], spec["w"], spec["eps0"])


BUILTIN_NAMES = ("vacuum", "gold") + tuple(_OSC.keys())
DEFAULT_MEDIA = "bromobenzene,diiodomethane"
_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
U_TO_G_CM3 = 1.66053906660  # u/Å³ → g/cm³
# Standard atomic weights (u), Z = 1..94. Only used for a density column.
_ATOMIC_MASS_U = np.array(
    [
        1.008, 4.003, 6.94, 9.012, 10.81, 12.011, 14.007, 15.999, 18.998, 20.18,
        22.99, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078,
        44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38,
        69.723, 72.63, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906, 91.224,
        92.906, 95.95, 97.91, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71,
        121.76, 127.60, 126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24,
        145.0, 150.36, 151.96, 157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05,
        174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59,
        204.38, 207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04,
        231.04, 238.03, 237.0, 244.0,
    ],
    dtype=np.float64,
)


def formula_n_elements(formula):
    return len(set(_ELEMENT_RE.findall(formula or "")))


def keep_crystal(formula, n_atoms, min_atoms, drop_elemental):
    if int(n_atoms) < min_atoms:
        return False
    if drop_elemental and formula_n_elements(formula) < 2:
        return False
    return True


def _mass_u(atomic_numbers):
    z = np.asarray(atomic_numbers, dtype=int)
    z = z[(z >= 1) & (z <= len(_ATOMIC_MASS_U))]
    if z.size == 0:
        return float("nan")
    return float(_ATOMIC_MASS_U[z - 1].sum())


def record_meta(data, idx):
    formula = data.get("formula") or ""
    n_atoms = data.get("n_atoms")
    if n_atoms is None:
        n_atoms = len(data.get("atomic_numbers") or data.get("atom_fea") or [])
    n_atoms = int(n_atoms)
    cell = data.get("cell")
    if cell is None:
        volume = float("nan")
    else:
        volume = float(abs(np.linalg.det(np.asarray(cell, dtype=np.float64).reshape(3, 3))))
    dens_n = n_atoms / volume if volume > 0 else float("nan")
    zs = data.get("atomic_numbers")
    dens_g = float("nan")
    if zs is not None and volume > 0:
        dens_g = _mass_u(zs) * U_TO_G_CM3 / volume
    gap = data.get("opt_gap")
    mbj = data.get("mbj_gap")
    return {
        "jid": data.get("jid") or f"idx{idx}",
        "mpid": data.get("mpid") or "",
        "formula": formula,
        "n_atoms": n_atoms,
        "n_elements": formula_n_elements(formula),
        "volume_A3": volume,
        "density_n_per_A3": dens_n,
        "density_g_cm3": dens_g,
        "opt_gap": float(gap) if gap is not None else float("nan"),
        "mbj_gap": float(mbj) if mbj is not None else float("nan"),
    }


def builtin_meta(name):
    return {
        "jid": name,
        "mpid": "",
        "formula": name,
        "n_atoms": float("nan"),
        "n_elements": float("nan"),
        "volume_A3": float("nan"),
        "density_n_per_A3": float("nan"),
        "density_g_cm3": float("nan"),
        "opt_gap": float("nan"),
        "mbj_gap": float("nan"),
    }


def matsubara_xi(n_max, temperature_k=T_K):
    return np.arange(n_max, dtype=np.float64) * (2.0 * np.pi * K_B_EV * temperature_k)


def interpolate_eps(xi_src, eps_src, xi_query):
    xi_src = np.asarray(xi_src, dtype=np.float64)
    eps_src = np.asarray(eps_src, dtype=np.float64)
    order = np.argsort(xi_src)
    xi_src = xi_src[order]
    eps_src = eps_src[order]
    out = np.interp(xi_query, xi_src, eps_src, left=eps_src[0], right=1.0)
    out[xi_query == 0.0] = eps_src[0]
    return out


def load_lmdb_eps(lmdb_path, xi_query, limit=None, min_atoms=2, drop_elemental=True):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    names, stack, metas = [], [], []
    n_skip = 0
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        n = int(meta["total_samples"])
        for i in range(n):
            rec = pickle.loads(txn.get(f"{i}".encode("utf-8")))
            kkr = rec.get("kkr_targets") or {}
            if "kkr_xx" not in kkr or rec.get("kkr_xi") is None:
                continue
            formula = rec.get("formula") or ""
            n_atoms = rec.get("n_atoms")
            if n_atoms is None:
                n_atoms = len(rec.get("atomic_numbers") or rec.get("atom_fea") or [])
            if not keep_crystal(formula, n_atoms, min_atoms, drop_elemental):
                n_skip += 1
                continue
            eps = interpolate_eps(rec["kkr_xi"], kkr["kkr_xx"], xi_query)
            jid = rec.get("jid") or f"idx{i}"
            names.append(f"{jid}:{formula}".strip(":"))
            stack.append(eps)
            metas.append(record_meta(rec, i))
            if limit is not None and len(stack) >= int(limit):
                break
    env.close()
    if n_skip:
        print(f"  dropped {n_skip} 1-atom/elemental cells")
    if not stack:
        return names, np.zeros((0, len(xi_query))), metas
    return names, np.stack(stack, axis=0), metas


def filter_crystal_indices(dataset, indices, min_atoms, drop_elemental):
    kept, metas = [], []
    n_skip_atoms = 0
    n_skip_elem = 0
    for i in indices:
        data = pickle.loads(dataset._txn.get(f"{i}".encode("utf-8")))
        meta = record_meta(data, i)
        if meta["n_atoms"] < min_atoms:
            n_skip_atoms += 1
            continue
        if drop_elemental and meta["n_elements"] < 2:
            n_skip_elem += 1
            continue
        kept.append(i)
        metas.append(meta)
    return kept, metas, n_skip_atoms, n_skip_elem


def crystal_label(dataset, idx):
    info = dataset.get_sample_info(idx)
    formula = info.get("formula") or ""
    jid = info.get("jid") or f"idx{idx}"
    return f"{jid}:{formula}".strip(":")


@torch.no_grad()
def predict_model_eps(model, model_type, dataset, indices, device, use_log, batch_size):
    """Predict ε_xx(iξ) on the LMDB KKR grid. Returns names, pred (N,F), gt (N,F)."""
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=get_collate_fn(use_mace=model_type == "MACE"),
    )
    preds, gts = [], []
    for inputs, target, _ in tqdm(loader, desc="model-eps"):
        pred = model_forward(model, model_type, inputs, device, use_log=use_log, channel=0)
        preds.append(pred.detach().cpu().numpy())
        gts.append(target[:, 0, :].numpy())
    pred = np.clip(np.concatenate(preds, axis=0), 1e-6, None)
    gt = np.clip(np.concatenate(gts, axis=0), 1e-6, None)
    names = [crystal_label(dataset, i) for i in indices]
    return names, pred, gt


def deltas(eps_body, eps_med):
    return (eps_body - eps_med) / (eps_body + eps_med + 1e-30)


def hamaker_zj_from_delta(d_a, d_b, temperature_k=T_K):
    """Non-retarded A (zJ). A < 0 is repulsive. d_a, d_b: (..., F) with n=0 first."""
    pref = 1.5 * K_B_J * temperature_k * ZJ_PER_J
    prod = d_a * d_b
    a = 0.5 * pref * prod[..., 0]
    if prod.shape[-1] > 1:
        a = a + pref * prod[..., 1:].sum(axis=-1)
    return a


def retardation_r(z_nm, xi_ev, eps_m):
    return 2.0 * z_nm * xi_ev * np.sqrt(np.maximum(eps_m, 1e-12)) / HBARC_EV_NM


def hamaker_zj_retarded(d_a, d_b, r_n, temperature_k=T_K):
    """
    Linearized retarded Hamaker (zJ): mode n≥1 weighted by (1+r) e^{-r}.
    r_n[0] should be 0 (static term is never retarded).
    """
    pref = 1.5 * K_B_J * temperature_k * ZJ_PER_J
    w = (1.0 + r_n) * np.exp(-np.clip(r_n, 0.0, 50.0))
    w = np.array(w, dtype=np.float64, copy=True)
    w[0] = 0.5
    return pref * np.sum(d_a * d_b * w, axis=-1)


def force_density_kn_m2(a_zj, z_nm):
    """Parallel-plate F (kN/m²). Positive = repulsive. F = −A / (6π z³)."""
    a_j = a_zj / ZJ_PER_J
    z_m = np.asarray(z_nm, dtype=np.float64) * 1e-9
    f_pa = -a_j / (6.0 * np.pi * z_m**3)
    return f_pa / 1e3


def classify_sign(a_nr, a_near, a_far):
    if a_nr >= 0 and a_near >= 0 and a_far >= 0:
        return "attractive"
    if a_nr < 0 and a_near < 0 and a_far < 0:
        return "repulsive"
    return "sign_flip"


def eval_triple(eps_a, eps_m, eps_b, xi, z_grid, temperature_k=T_K):
    d_a = deltas(eps_a, eps_m)
    d_b = deltas(eps_b, eps_m)
    a_nr = float(hamaker_zj_from_delta(d_a, d_b, temperature_k))
    frac = float(np.mean((eps_a - eps_m) * (eps_b - eps_m) < 0))
    a_z = []
    for z in z_grid:
        r_n = retardation_r(z, xi, eps_m)
        r_n[0] = 0.0
        a_z.append(float(hamaker_zj_retarded(d_a, d_b, r_n, temperature_k)))
    a_z = np.asarray(a_z)
    kind = classify_sign(a_nr, a_z[0], a_z[-1])
    return {
        "A_nonret_zj": a_nr,
        "mismatch_frac": frac,
        "kind": kind,
        "A_ret_zj": a_z,
        "F_kn_m2": force_density_kn_m2(a_z, z_grid),
    }


def hamaker_matrix(eps_bodies, eps_m, temperature_k):
    d_all = deltas(eps_bodies, eps_m)
    pref = 1.5 * K_B_J * temperature_k * ZJ_PER_J
    w = np.ones(d_all.shape[1])
    w[0] = 0.5
    dw = d_all * np.sqrt(w)
    return pref * (dw @ dw.T)


def search_pairs(names, eps, media_idx, body_idx, xi, z_grid, temperature_k, top_k, gt_eps=None):
    body_idx = np.asarray(body_idx, dtype=np.int32)
    chunks_ia, chunks_ib, chunks_im, chunks_a = [], [], [], []
    chunks_gt = []
    vacuum_n_rep = 0
    n_checked = 0
    for im in media_idx:
        a_mat = hamaker_matrix(eps[body_idx], eps[im], temperature_k)
        n_checked += int(a_mat.size)
        ia, ib = np.where(np.triu(a_mat < 0.0, k=1))
        if names[im] == "vacuum":
            vacuum_n_rep += int(len(ia))
        ja = body_idx[ia]
        jb = body_idx[ib]
        keep = (ja != im) & (jb != im)
        ia, ib, ja, jb = ia[keep], ib[keep], ja[keep], jb[keep]
        aval = a_mat[ia, ib].astype(np.float32)
        chunks_ia.append(ja)
        chunks_ib.append(jb)
        chunks_im.append(np.full(len(ja), im, dtype=np.int32))
        chunks_a.append(aval)
        if gt_eps is not None:
            gt_mat = hamaker_matrix(gt_eps[body_idx], gt_eps[im], temperature_k)
            chunks_gt.append(gt_mat[ia, ib].astype(np.float32))
            del gt_mat
        del a_mat

    if chunks_a:
        ia_all = np.concatenate(chunks_ia)
        ib_all = np.concatenate(chunks_ib)
        im_all = np.concatenate(chunks_im)
        a_all = np.concatenate(chunks_a)
        gt_all = np.concatenate(chunks_gt) if chunks_gt else None
    else:
        ia_all = np.zeros(0, dtype=np.int32)
        ib_all = np.zeros(0, dtype=np.int32)
        im_all = np.zeros(0, dtype=np.int32)
        a_all = np.zeros(0, dtype=np.float32)
        gt_all = None if gt_eps is None else np.zeros(0, dtype=np.float32)

    n_repulsive = int(a_all.size)
    hits = []
    if n_repulsive and top_k > 0:
        k = min(int(top_k), n_repulsive)
        order = np.argpartition(a_all, k - 1)[:k]
        order = order[np.argsort(a_all[order])]
        for t in order:
            ja, jb, im = int(ia_all[t]), int(ib_all[t]), int(im_all[t])
            stats = eval_triple(eps[ja], eps[im], eps[jb], xi, z_grid, temperature_k)
            row = {"A": names[ja], "medium": names[im], "B": names[jb], **stats}
            if gt_eps is not None:
                gst = eval_triple(gt_eps[ja], gt_eps[im], gt_eps[jb], xi, z_grid, temperature_k)
                row["A_gt_zj"] = gst["A_nonret_zj"]
                row["kind_gt"] = gst["kind"]
                row["gt_agrees"] = (stats["A_nonret_zj"] < 0) and (gst["A_nonret_zj"] < 0)
            hits.append(row)
    pairs = {"ia": ia_all, "ib": ib_all, "im": im_all, "A_nr_zj": a_all}
    if gt_all is not None:
        pairs["A_gt_zj"] = gt_all
    return hits, n_checked, vacuum_n_rep, n_repulsive, pairs


def print_table(hits, z_grid):
    z_near, z_far = z_grid[0], z_grid[-1]
    show_gt = bool(hits) and "A_gt_zj" in hits[0]
    hdr = (
        f"\n{'A':<22} {'medium':<16} {'B':<22} {'A_nr (zJ)':>12} "
        f"{'mismatch':>8} {'kind':<12} {f'F({z_near:g}nm)':>12} {f'F({z_far:g}nm)':>12}"
    )
    if show_gt:
        hdr += f" {'A_gt (zJ)':>12} {'kind_gt':<12} {'agree':>5}"
    print(hdr)
    print("-" * (len(hdr) + 4))
    for h in hits:
        line = (
            f"{h['A'][:22]:<22} {h['medium'][:16]:<16} {h['B'][:22]:<22} "
            f"{h['A_nonret_zj']:12.4f} {h['mismatch_frac']:8.2f} {h['kind']:<12} "
            f"{h['F_kn_m2'][0]:12.4f} {h['F_kn_m2'][-1]:12.4f}"
        )
        if show_gt:
            line += (
                f" {h['A_gt_zj']:12.4f} {h['kind_gt']:<12} "
                f"{'yes' if h['gt_agrees'] else 'no':>5}"
            )
        print(line)
    print("F in kN/m² (parallel plates); positive = repulsive. A_nr < 0 = repulsive.")


def plot_force_vs_z(named, library, xi, temperature_k, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.logspace(np.log10(2.0), np.log10(200.0), 80)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for a, m, b in named:
        if a not in library or m not in library or b not in library:
            continue
        st = eval_triple(library[a], library[m], library[b], xi, z, temperature_k)
        ax.semilogx(z, st["F_kn_m2"], label=f"{a}|{m}|{b} ({st['kind']})")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("separation z (nm)")
    ax.set_ylabel("force density (kN/m²), + repulsive")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_materials_csv(path, names, metas, eps, builtin_set):
    cols = [
        "idx", "name", "jid", "mpid", "formula", "n_atoms", "n_elements",
        "volume_A3", "density_n_per_A3", "density_g_cm3", "opt_gap", "mbj_gap",
        "eps_xi0", "builtin",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, (name, meta) in enumerate(zip(names, metas)):
            w.writerow(
                [
                    i,
                    name,
                    meta["jid"],
                    meta["mpid"],
                    meta["formula"],
                    meta["n_atoms"],
                    meta["n_elements"],
                    meta["volume_A3"],
                    meta["density_n_per_A3"],
                    meta["density_g_cm3"],
                    meta["opt_gap"],
                    meta["mbj_gap"],
                    float(eps[i, 0]),
                    name in builtin_set,
                ]
            )


def write_pairs_npz(path, names, metas, xi, eps, gt_eps, pairs):
    payload = {
        "names": np.asarray(names),
        "xi": np.asarray(xi, dtype=np.float64),
        "eps": np.asarray(eps, dtype=np.float32),
        "ia": pairs["ia"],
        "ib": pairs["ib"],
        "im": pairs["im"],
        "A_nr_zj": pairs["A_nr_zj"],
        "jid": np.asarray([m["jid"] for m in metas]),
        "mpid": np.asarray([m["mpid"] for m in metas]),
        "formula": np.asarray([m["formula"] for m in metas]),
        "n_atoms": np.asarray([m["n_atoms"] for m in metas], dtype=np.float32),
        "n_elements": np.asarray([m["n_elements"] for m in metas], dtype=np.float32),
        "volume_A3": np.asarray([m["volume_A3"] for m in metas], dtype=np.float32),
        "density_n_per_A3": np.asarray([m["density_n_per_A3"] for m in metas], dtype=np.float32),
        "density_g_cm3": np.asarray([m["density_g_cm3"] for m in metas], dtype=np.float32),
        "opt_gap": np.asarray([m["opt_gap"] for m in metas], dtype=np.float32),
        "mbj_gap": np.asarray([m["mbj_gap"] for m in metas], dtype=np.float32),
    }
    if "A_gt_zj" in pairs:
        payload["A_gt_zj"] = pairs["A_gt_zj"]
    if gt_eps is not None:
        payload["eps_gt"] = np.asarray(gt_eps, dtype=np.float32)
    np.savez(path, **payload)


def parse_name_list(s, available, builtins=None, crystals=None):
    if s is None or s.strip().lower() in ("all", "*"):
        return list(available)
    builtins = list(builtins or [])
    crystals = list(crystals or [])
    out = []
    for part in s.split(","):
        name = part.strip()
        if not name:
            continue
        key = name.lower()
        if key == "builtins":
            out.extend(n for n in builtins if n in available)
            continue
        if key == "crystals":
            out.extend(n for n in crystals if n in available)
            continue
        if name not in available:
            raise SystemExit(f"Unknown material {name!r}. Aliases: all, builtins, crystals.")
        out.append(name)
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def main():
    parser = argparse.ArgumentParser(
        description="Screen A|medium|B stacks for repulsive Lifshitz–vdW forces",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="FrequencyCGCNN/MACE checkpoint; predict ε(iξ) for crystals",
    )
    parser.add_argument(
        "--lmdb_path",
        default=None,
        help="crystal_graphs.lmdb (required with --model_path; else GT-only crystals)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of crystals")
    parser.add_argument("--split", default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--compare_gt",
        action="store_true",
        help="For each model hit, also compute A from LMDB KKR ε (DFT ground truth)",
    )
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--z_nm", default="5,10,20,50,100", help="Distances for retarded A and F")
    parser.add_argument(
        "--media",
        default=None,
        help="Comma list, or all/builtins/crystals. Default: bromobenzene,diiodomethane",
    )
    parser.add_argument(
        "--bodies",
        default=None,
        help="Comma list, or all/builtins/crystals. Default with model: crystals,gold",
    )
    parser.add_argument(
        "--min_atoms",
        type=int,
        default=2,
        help="Drop crystals with fewer atoms in the cell (default: 2, i.e. no 1-atom cells)",
    )
    parser.add_argument(
        "--keep_elemental",
        action="store_true",
        help="Keep single-element crystals (He, Na, Si, ...). Default is to drop them",
    )
    parser.add_argument(
        "--crystals_as_media",
        action="store_true",
        help="Allow crystals as the intervening medium (bulk 3-layer stack)",
    )
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--out_json", default="repulsive_lifshitz.json")
    parser.add_argument("--out_csv", default="repulsive_lifshitz.csv")
    parser.add_argument(
        "--out_pairs",
        default="repulsive_lifshitz_pairs.npz",
        help="All A_nr<0 pairs (ia,ib,im,A_nr). Empty string to skip",
    )
    parser.add_argument(
        "--out_materials",
        default="repulsive_lifshitz_materials.csv",
        help="One row per material (density, n_atoms, eps(0), ...). Empty string to skip",
    )
    parser.add_argument("--plot", default="repulsive_lifshitz_F_vs_z.png")
    args = parser.parse_args()

    if args.model_path and not args.lmdb_path:
        raise SystemExit("--lmdb_path is required with --model_path")
    if args.media is None:
        args.media = DEFAULT_MEDIA
    if args.bodies is None:
        args.bodies = "crystals,gold" if args.model_path else "all"

    z_grid = np.array([float(x) for x in args.z_nm.split(",")], dtype=np.float64)
    xi = matsubara_xi(args.n_matsubara, args.temperature_k)

    names = list(BUILTIN_NAMES)
    eps_list = [builtin_eps_fn(n)(xi) for n in names]
    gt_list = [e.copy() for e in eps_list]
    metas = [builtin_meta(n) for n in names]
    builtin_set = set(BUILTIN_NAMES)
    model_meta = None

    if args.model_path:
        lmdb_path = Path(args.lmdb_path)
        if not lmdb_path.is_file():
            raise SystemExit(f"LMDB not found: {lmdb_path}")
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading {args.model_path} on {device}")
        ckpt = torch.load(args.model_path, map_location=device)
        use_log = bool(ckpt.get("use_log", False))
        model, model_type = load_model(ckpt, device)
        use_mace = model_type == "MACE"
        target_components = tuple(ckpt.get("target_components", ("xx",)))
        dataset = LMDBCrystalDataset(
            str(lmdb_path),
            target_components=target_components,
            use_kkr=True,
            metal_gap_threshold=ckpt.get("metal_gap_threshold"),
            use_mace=use_mace,
            mace_r_max=ckpt.get("mace_r_max"),
        )
        if use_mace and ckpt.get("mace_atomic_numbers") is not None:
            from mace.tools import AtomicNumberTable

            dataset.set_mace_graph_config(
                AtomicNumberTable(zs=[int(z) for z in ckpt["mace_atomic_numbers"]]),
                ckpt.get("mace_r_max", dataset.mace_r_max),
            )
        if args.split == "all":
            indices = list(range(len(dataset)))
        else:
            indices = list(get_split_indices(ckpt, len(dataset), args.split))
        n_split = len(indices)
        indices, crystal_metas, n_skip_atoms, n_skip_elem = filter_crystal_indices(
            dataset, indices, args.min_atoms, drop_elemental=not args.keep_elemental,
        )
        if args.limit is not None:
            indices = indices[: args.limit]
            crystal_metas = crystal_metas[: args.limit]
        print(
            f"  type={model_type}  use_log={use_log}  split={args.split}  "
            f"n={len(indices)}/{n_split}  ξ_KKR={dataset.n_frequencies}"
        )
        if n_skip_atoms or n_skip_elem:
            print(
                f"  dropped {n_skip_atoms} with n_atoms<{args.min_atoms}, "
                f"{n_skip_elem} elemental"
            )
        if not indices:
            raise SystemExit("No crystals left after 1-atom/elemental filter.")
        cnames, pred_kkr, gt_kkr = predict_model_eps(
            model, model_type, dataset, indices, device, use_log, args.batch_size,
        )
        xi_kkr = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
        dataset.close()
        pred_m = np.stack([interpolate_eps(xi_kkr, row, xi) for row in pred_kkr])
        gt_m = np.stack([interpolate_eps(xi_kkr, row, xi) for row in gt_kkr])
        names.extend(cnames)
        eps_list.extend(list(pred_m))
        gt_list.extend(list(gt_m))
        metas.extend(crystal_metas)
        model_meta = {
            "model_path": args.model_path,
            "model_type": model_type,
            "use_log": use_log,
            "split": args.split,
            "n_crystals": len(cnames),
            "n_split": n_split,
            "n_dropped_atoms": n_skip_atoms,
            "n_dropped_elemental": n_skip_elem,
            "min_atoms": args.min_atoms,
            "drop_elemental": not args.keep_elemental,
            "lmdb_path": str(lmdb_path),
        }
    elif args.lmdb_path:
        path = Path(args.lmdb_path)
        if not path.is_file():
            raise SystemExit(f"LMDB not found: {path}")
        print(f"Loading DFT-KKR crystals from {path} ...")
        cnames, ceps, crystal_metas = load_lmdb_eps(
            path, xi, limit=args.limit,
            min_atoms=args.min_atoms, drop_elemental=not args.keep_elemental,
        )
        print(f"  {len(cnames)} crystals with KKR ε_xx")
        names.extend(cnames)
        eps_list.extend(list(ceps))
        gt_list.extend(list(ceps))
        metas.extend(crystal_metas)

    eps = np.stack(eps_list, axis=0)
    gt_eps = np.stack(gt_list, axis=0)
    name_to_i = {n: i for i, n in enumerate(names)}
    crystal_names = [n for n in names if n not in builtin_set]

    media_avail = names if args.crystals_as_media else [n for n in names if n in builtin_set]
    media_names = parse_name_list(
        args.media, media_avail, builtins=BUILTIN_NAMES, crystals=crystal_names,
    )
    body_names = parse_name_list(
        args.bodies, names, builtins=BUILTIN_NAMES, crystals=crystal_names,
    )
    if not body_names:
        raise SystemExit("No bodies selected. With --model_path, default is crystals,gold.")
    media_idx = [name_to_i[n] for n in media_names]
    body_idx = [name_to_i[n] for n in body_names]

    n_body_builtin = sum(n in builtin_set for n in body_names)
    n_body_crystal = len(body_names) - n_body_builtin
    print("=" * 72)
    print("Repulsive Lifshitz–vdW search  (A | medium | B)")
    if model_meta:
        print(f"  model={model_meta['model_path']}  [{model_meta['model_type']}]")
    print(f"  T={args.temperature_k} K   n_matsubara={args.n_matsubara}   z={list(z_grid)} nm")
    print(f"  media ({len(media_idx)}): {', '.join(media_names)}")
    print(f"  bodies ({len(body_idx)}): {n_body_builtin} builtin + {n_body_crystal} crystals")
    print(
        f"  crystal filter: min_atoms={args.min_atoms}  "
        f"drop_elemental={not args.keep_elemental}"
    )
    print("=" * 72)
    print(
        "\nDistance: |F| always falls with z (∼z⁻³ non-retarded, faster when retarded).\n"
        "The SIGN is z-independent only if mismatch holds across the Matsubara sum.\n"
        "Vacuum gap: ε_m=1 ≤ ε_A, ε_B for all passive materials → no planar repulsion."
    )

    hits, n_checked, vacuum_n_rep, n_repulsive, pairs = search_pairs(
        names, eps, media_idx, body_idx, xi, z_grid, args.temperature_k, args.top_k,
        gt_eps=gt_eps,
    )
    n_flip = sum(1 for h in hits if h["kind"] == "sign_flip")
    n_rep = sum(1 for h in hits if h["kind"] == "repulsive")
    print(
        f"\nChecked {n_checked} unordered pairs, {n_repulsive} with A_nr<0. "
        f"Showing top {len(hits)}: {n_rep} always-repulsive, {n_flip} sign-flip."
    )
    if "vacuum" in media_names:
        print(f"Vacuum as medium: {vacuum_n_rep} repulsive pairs (expected 0).")
    if hits:
        n_agree = sum(1 for h in hits if h.get("gt_agrees"))
        print(f"GT agreement among shown hits: {n_agree}/{len(hits)} also A_gt<0")

    if hits:
        print_table(hits, z_grid)
    else:
        print("\nNo repulsive stacks. Try a high-n liquid medium (bromobenzene, diiodomethane)")
        print("between a low-ε solid (PTFE, silica, or a predicted crystal) and gold.")

    demo = [
        ("silica", "vacuum", "gold"),
        ("PTFE", "vacuum", "gold"),
        ("silica", "bromobenzene", "gold"),
        ("PTFE", "bromobenzene", "gold"),
        ("PTFE", "diiodomethane", "gold"),
        ("silica", "water", "gold"),
        ("PTFE", "water", "gold"),
        ("silica", "ethanol", "gold"),
    ]
    lib = {n: eps[name_to_i[n]] for n in BUILTIN_NAMES}
    print("\n--- Classic stacks (force vs z) ---")
    print(f"{'stack':<36} {'kind':<12} {'A_nr (zJ)':>12} {'mismatch':>8}")
    for a, m, b in demo:
        st = eval_triple(lib[a], lib[m], lib[b], xi, z_grid, args.temperature_k)
        print(
            f"{a}|{m}|{b:<16} {st['kind']:<12} {st['A_nonret_zj']:12.4f} {st['mismatch_frac']:8.2f}"
            f"   F({z_grid[0]:g}nm)={st['F_kn_m2'][0]:+.4f}  F({z_grid[-1]:g}nm)={st['F_kn_m2'][-1]:+.4f} kN/m²"
        )

    payload = {
        "temperature_k": args.temperature_k,
        "n_matsubara": args.n_matsubara,
        "z_nm": z_grid.tolist(),
        "n_hits": len(hits),
        "n_repulsive_total": n_repulsive,
        "vacuum_repulsive_pairs": int(vacuum_n_rep),
        "model": model_meta,
        "hits": [
            {
                "A": h["A"],
                "medium": h["medium"],
                "B": h["B"],
                "A_nonret_zj": h["A_nonret_zj"],
                "mismatch_frac": h["mismatch_frac"],
                "kind": h["kind"],
                "A_ret_zj": h["A_ret_zj"].tolist(),
                "F_kn_m2": h["F_kn_m2"].tolist(),
                **(
                    {
                        "A_gt_zj": h["A_gt_zj"],
                        "kind_gt": h["kind_gt"],
                        "gt_agrees": h["gt_agrees"],
                    }
                    if "A_gt_zj" in h
                    else {}
                ),
            }
            for h in hits
        ],
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["A", "medium", "B", "A_nonret_zj", "mismatch_frac", "kind"]
        if hits and "A_gt_zj" in hits[0]:
            cols += ["A_gt_zj", "kind_gt", "gt_agrees"]
        w.writerow(
            cols
            + [f"A_ret_z{z:g}_zj" for z in z_grid]
            + [f"F_z{z:g}_kNpm2" for z in z_grid]
        )
        for h in hits:
            extra = []
            if "A_gt_zj" in h:
                extra = [h["A_gt_zj"], h["kind_gt"], h["gt_agrees"]]
            w.writerow(
                [h["A"], h["medium"], h["B"], h["A_nonret_zj"], h["mismatch_frac"], h["kind"]]
                + extra
                + list(h["A_ret_zj"])
                + list(h["F_kn_m2"])
            )
    print(f"\nWrote {args.out_json} and {args.out_csv}")

    if len(metas) != len(names):
        raise SystemExit(f"internal error: {len(metas)} metas vs {len(names)} names")
    if args.out_materials:
        write_materials_csv(args.out_materials, names, metas, eps, builtin_set)
        print(f"Wrote {args.out_materials}  ({len(names)} materials)")
    if args.out_pairs:
        print(f"Writing all {n_repulsive} A_nr<0 pairs to {args.out_pairs} ...")
        write_pairs_npz(args.out_pairs, names, metas, xi, eps, gt_eps, pairs)
        print(f"Wrote {args.out_pairs}")

    if args.plot:
        plot_force_vs_z(demo, lib, xi, args.temperature_k, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
