#!/usr/bin/env python3
"""
Consensus repulsive crystal | liquid | gold.

Voters: DFT-KKR plus any CGCNN / MACE / TSENN checkpoints. Gold is still the
simple Drude builtin (not Palik); liquids are van Zwol oscillators.

Default split is the intersection of every neural voter's test set.

  python consensus_repulsive.py \\
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
    hamaker_crystal_vs_fixed,
    kind_crystal_vs_fixed,
    load_voters,
    parse_media,
    parse_z_grid,
    voter_ok,
)
from find_repulsive_lifshitz import BUILTIN_NAMES, builtin_eps_fn, eval_triple

FIXED_BODY = "gold"


def main():
    parser = argparse.ArgumentParser(
        description="Consensus repulsive crystal|liquid|gold (all voters must agree)",
    )
    add_voter_args(parser)
    parser.add_argument("--fixed_body", default=FIXED_BODY)
    parser.add_argument(
        "--require_both_media",
        action="store_true",
        help="Crystal must be consensus-repulsive in every listed medium",
    )
    parser.add_argument("--out_csv", default="consensus_gold_triples.csv")
    parser.add_argument("--out_crystals", default="consensus_gold_crystals.csv")
    args = parser.parse_args()

    media_names = parse_media(args.media)
    for name in media_names + [args.fixed_body]:
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
    n_mat = int(keep_mat.sum())

    eps_fixed = {n: builtin_eps_fn(n)(xi) for n in media_names + [args.fixed_body]}
    eps_gold = eps_fixed[args.fixed_body]

    print("=" * 72)
    print("Consensus repulsive filter  (crystal | medium | gold)")
    print(f"  voters ({len(voter_names)}): {', '.join(voter_names)}")
    print(f"  media: {', '.join(media_names)}   fixed body: {args.fixed_body}  [Drude, not Palik]")
    print(f"  T={t_k} K   n_matsubara={args.n_matsubara}")
    print("=" * 72)

    rows = []
    crystal_media = {i: set() for i in np.flatnonzero(keep_mat)}
    name_to_i = {n: i for i, n in enumerate(names)}

    for medium in media_names:
        eps_m = eps_fixed[medium]
        a_by_voter = {}
        kind_by_voter = {}
        consensus = keep_mat.copy()
        print(f"\n{medium}")
        for vname, eps_c in voters.items():
            a = hamaker_crystal_vs_fixed(eps_c, eps_gold, eps_m, t_k)
            a_by_voter[vname] = a
            kinds = None
            if args.always_repulsive:
                kinds, _ = kind_crystal_vs_fixed(eps_c, eps_gold, eps_m, xi, z_grid, t_k)
                kind_by_voter[vname] = kinds
            ok = voter_ok(a, kinds, args.always_repulsive)
            consensus &= ok
            print(f"  {vname:<32}  repulsive alone: {int((ok & keep_mat).sum())}")

        idx = np.flatnonzero(consensus)
        print(f"  consensus: {len(idx)} / {n_mat}")
        for i in idx:
            a_vals = {v: float(a_by_voter[v][i]) for v in voter_names}
            row = {
                "crystal": names[i],
                "jid": metas[i]["jid"],
                "formula": metas[i]["formula"],
                "medium": medium,
                "density_g_cm3": float(dens[i]),
                "n_atoms": float(n_at[i]),
                "n_elements": float(n_el[i]),
                "volume_A3": float(metas[i]["volume_A3"]),
                "opt_gap": float(metas[i]["opt_gap"]),
                "mbj_gap": float(metas[i]["mbj_gap"]),
                "A_mean_zj": float(np.mean(list(a_vals.values()))),
                "A_min_zj": float(np.min(list(a_vals.values()))),
            }
            for v in voter_names:
                row[f"A_{v}_zj"] = a_vals[v]
                row[f"eps0_{v}"] = float(voters[v][i, 0])
                if args.always_repulsive:
                    row[f"kind_{v}"] = str(kind_by_voter[v][i])
            rows.append(row)
            crystal_media[i].add(medium)

    if args.require_both_media:
        need = set(media_names)
        rows = [r for r in rows if crystal_media[name_to_i[r["crystal"]]] >= need]

    rows.sort(key=lambda r: r["A_mean_zj"])
    a_cols = [f"A_{v}_zj" for v in voter_names]
    kind_cols = [f"kind_{v}" for v in voter_names] if args.always_repulsive else []
    eps0_cols = [f"eps0_{v}" for v in voter_names]
    cols = (
        ["crystal", "jid", "formula", "medium"]
        + a_cols
        + ["A_mean_zj", "A_min_zj"]
        + kind_cols
        + ["density_g_cm3", "n_atoms", "n_elements", "volume_A3", "opt_gap", "mbj_gap"]
        + eps0_cols
    )
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])

    kept_idx = sorted({name_to_i[r["crystal"]] for r in rows})
    with open(args.out_crystals, "w", newline="") as f:
        w = csv.writer(f)
        xtal_cols = [
            "crystal", "jid", "formula", "n_media", "media",
            "A_mean_zj", "density_g_cm3", "n_atoms", "n_elements", "opt_gap",
        ] + eps0_cols
        w.writerow(xtal_cols)
        for i in kept_idx:
            sub = [r for r in rows if r["crystal"] == names[i]]
            rec = {
                "crystal": names[i],
                "jid": metas[i]["jid"],
                "formula": metas[i]["formula"],
                "n_media": len(sub),
                "media": ",".join(sorted(r["medium"] for r in sub)),
                "A_mean_zj": float(np.mean([r["A_mean_zj"] for r in sub])),
                "density_g_cm3": float(dens[i]),
                "n_atoms": float(n_at[i]),
                "n_elements": float(n_el[i]),
                "opt_gap": float(metas[i]["opt_gap"]),
            }
            for v in voter_names:
                rec[f"eps0_{v}"] = float(voters[v][i, 0])
            w.writerow([rec[c] for c in xtal_cols])

    print(f"\nConsensus triples: {len(rows)}  unique crystals: {len(kept_idx)}")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_crystals}")

    print("\n--- Classic stacks (sanity) ---")
    for a_name, m_name in (("silica", "bromobenzene"), ("PTFE", "bromobenzene"),
                           ("PTFE", "diiodomethane"), ("silica", "vacuum")):
        if m_name not in eps_fixed and m_name != "vacuum":
            continue
        st = eval_triple(
            builtin_eps_fn(a_name)(xi), builtin_eps_fn(m_name)(xi), eps_gold,
            xi, z_grid, t_k,
        )
        print(f"  {a_name}|{m_name}|{args.fixed_body}  {st['kind']:<12} A_nr={st['A_nonret_zj']:8.3f} zJ")

    nshow = min(int(args.top_print), len(rows))
    if nshow:
        extra = "  ".join(f"{v[:12]:>12}" for v in voter_names)
        print(f"\n{'crystal':<28} {'medium':<16} {extra} {'mean':>10} {'ρ':>7}")
        print("-" * (70 + 14 * len(voter_names)))
        for r in rows[:nshow]:
            nums = "  ".join(f"{r[f'A_{v}_zj']:12.3f}" for v in voter_names)
            print(
                f"{r['crystal'][:28]:<28} {r['medium'][:16]:<16} {nums} "
                f"{r['A_mean_zj']:10.3f} {r['density_g_cm3']:7.2f}"
            )


if __name__ == "__main__":
    main()
