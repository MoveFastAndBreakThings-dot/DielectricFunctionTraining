#!/usr/bin/env python3
"""
Consensus repulsive crystal | liquid | crystal.

Same voters and test-set intersection as consensus_repulsive.py, but both
bodies are JARVIS crystals. Liquids are van Zwol oscillators. No gold.

  python consensus_crystal_pairs.py \\
      --lmdb_path latest_data_opt/crystal_graphs.lmdb \\
      --model models_latest_opt/cgcnn_kkr_latest_opt.pt \\
      --model models_latest_opt/mace_kkr_latest_opt.pt \\
      --tsenn models_latest_opt/tsenn_kkr_xx_latest_opt_best.torch \\
      --min_density 2 --min_atoms 4
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from consensus_common import (
    add_voter_args,
    kind_pair,
    load_voters,
    parse_media,
    parse_z_grid,
)
from find_repulsive_lifshitz import BUILTIN_NAMES, builtin_eps_fn, hamaker_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Consensus repulsive crystal|liquid|crystal (all voters must agree)",
    )
    add_voter_args(parser)
    parser.add_argument("--out_csv", default="consensus_crystal_pairs.csv")
    parser.add_argument("--out_npz", default="consensus_crystal_pairs.npz")
    parser.add_argument(
        "--csv_max", type=int, default=10000,
        help="Write at most this many most-negative pairs to CSV (npz has all)",
    )
    args = parser.parse_args()

    media_names = parse_media(args.media)
    for name in media_names:
        if name not in BUILTIN_NAMES:
            raise SystemExit(f"Unknown builtin {name!r}. Available: {', '.join(BUILTIN_NAMES)}")

    z_grid = parse_z_grid(args.z_nm)
    packed = load_voters(args, args.temperature_k)
    voters = packed["voters"]
    names = packed["names"]
    metas = packed["metas"]
    keep_mat = packed["keep_mat"]
    dens = packed["dens"]
    n_at = packed["n_at"]
    n_el = packed["n_el"]
    xi = packed["xi"]
    t_k = packed["temperature_k"]
    voter_names = list(voters.keys())

    keep_idx = np.flatnonzero(keep_mat)
    n_keep = len(keep_idx)
    if n_keep < 2:
        raise SystemExit("Need at least 2 crystals after filters for crystal|liquid|crystal.")
    names_k = [names[i] for i in keep_idx]
    voters_k = {v: eps[keep_idx] for v, eps in voters.items()}
    n_pairs_space = n_keep * (n_keep - 1) // 2
    print("=" * 72)
    print("Consensus repulsive filter  (crystal | medium | crystal)")
    print(f"  voters ({len(voter_names)}): {', '.join(voter_names)}")
    print(f"  media: {', '.join(media_names)}")
    print(f"  bodies: {n_keep} crystals   unordered pairs/medium: {n_pairs_space}")
    print(f"  T={t_k} K   n_matsubara={args.n_matsubara}")
    print("=" * 72)

    ia_all, ib_all, im_all = [], [], []
    a_store = {v: [] for v in voter_names}

    for m_i, medium in enumerate(media_names):
        eps_m = builtin_eps_fn(medium)(xi)
        print(f"\n{medium}")
        a_by_voter = {}
        consensus = None
        for vname, eps_c in voters_k.items():
            a_mat = hamaker_matrix(eps_c, eps_m, t_k)
            a_by_voter[vname] = a_mat
            ok = a_mat < 0.0
            n_alone = int(np.triu(ok, k=1).sum())
            print(f"  {vname:<32}  repulsive pairs: {n_alone}")
            consensus = ok if consensus is None else (consensus & ok)

        ia, ib = np.where(np.triu(consensus, k=1))
        print(f"  consensus: {len(ia)} / {n_pairs_space}")
        if args.always_repulsive and len(ia):
            print(f"  --always_repulsive on {len(ia)} pairs (slow if large)")
            keep_pair = np.ones(len(ia), dtype=bool)
            for t in range(len(ia)):
                for vname, eps_c in voters_k.items():
                    kind, _ = kind_pair(
                        eps_c[ia[t]], eps_m, eps_c[ib[t]], xi, z_grid, t_k,
                    )
                    if kind != "repulsive":
                        keep_pair[t] = False
                        break
            ia, ib = ia[keep_pair], ib[keep_pair]
            print(f"  after --always_repulsive: {len(ia)}")

        ia_all.append(ia.astype(np.int32))
        ib_all.append(ib.astype(np.int32))
        im_all.append(np.full(len(ia), m_i, dtype=np.int32))
        for vname in voter_names:
            a_store[vname].append(a_by_voter[vname][ia, ib].astype(np.float32))
        del a_by_voter

    if ia_all:
        ia = np.concatenate(ia_all)
        ib = np.concatenate(ib_all)
        im = np.concatenate(im_all)
        a_cat = {v: np.concatenate(a_store[v]) for v in voter_names}
    else:
        ia = np.zeros(0, dtype=np.int32)
        ib = np.zeros(0, dtype=np.int32)
        im = np.zeros(0, dtype=np.int32)
        a_cat = {v: np.zeros(0, dtype=np.float32) for v in voter_names}

    a_mean = np.mean(np.stack([a_cat[v] for v in voter_names], axis=0), axis=0) if len(ia) else np.zeros(0)
    order = np.argsort(a_mean) if len(ia) else np.zeros(0, dtype=np.int64)

    payload = {
        "names": np.asarray(names_k),
        "jid": np.asarray([metas[i]["jid"] for i in keep_idx]),
        "formula": np.asarray([metas[i]["formula"] for i in keep_idx]),
        "density_g_cm3": dens[keep_idx],
        "n_atoms": n_at[keep_idx],
        "n_elements": n_el[keep_idx],
        "media": np.asarray(media_names),
        "ia": ia,
        "ib": ib,
        "im": im,
        "A_mean_zj": a_mean.astype(np.float32),
        "voter_names": np.asarray(voter_names),
    }
    for v in voter_names:
        payload[f"A_{v}_zj"] = a_cat[v]
    np.savez(args.out_npz, **payload)
    print(f"\nWrote {args.out_npz}  ({len(ia)} consensus pairs)")

    n_csv = min(int(args.csv_max), len(order))
    a_cols = [f"A_{v}_zj" for v in voter_names]
    cols = [
        "A", "jid_A", "formula_A", "B", "jid_B", "formula_B", "medium",
        *a_cols, "A_mean_zj",
        "density_A", "density_B", "n_atoms_A", "n_atoms_B",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in order[:n_csv]:
            ja, jb, jm = int(ia[t]), int(ib[t]), int(im[t])
            ia0, ib0 = int(keep_idx[ja]), int(keep_idx[jb])
            row = [
                names_k[ja], metas[ia0]["jid"], metas[ia0]["formula"],
                names_k[jb], metas[ib0]["jid"], metas[ib0]["formula"],
                media_names[jm],
            ]
            row.extend(float(a_cat[v][t]) for v in voter_names)
            row.extend(
                [
                    float(a_mean[t]),
                    float(dens[ia0]), float(dens[ib0]),
                    float(n_at[ia0]), float(n_at[ib0]),
                ]
            )
            w.writerow(row)
    print(f"Wrote {args.out_csv}  (top {n_csv} by mean A)")

    nshow = min(int(args.top_print), len(order))
    if nshow:
        extra = "  ".join(f"{v[:12]:>12}" for v in voter_names)
        print(f"\n{'A':<22} {'medium':<16} {'B':<22} {extra} {'mean':>10}")
        print("-" * (70 + 14 * len(voter_names)))
        for t in order[:nshow]:
            ja, jb, jm = int(ia[t]), int(ib[t]), int(im[t])
            nums = "  ".join(f"{float(a_cat[v][t]):12.3f}" for v in voter_names)
            print(
                f"{names_k[ja][:22]:<22} {media_names[jm][:16]:<16} {names_k[jb][:22]:<22} "
                f"{nums} {float(a_mean[t]):10.3f}"
            )


if __name__ == "__main__":
    main()
