#!/usr/bin/env python3
"""
Keep Munday-geometry triples: crystal | bromobenzene or diiodomethane | gold.

  python filter_gold_pairs.py
  python filter_gold_pairs.py --min_density 2 --min_atoms 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

GOLD_LIQUIDS = ("bromobenzene", "diiodomethane")


def find_name(names, target):
    for i, n in enumerate(names):
        if str(n) == target:
            return i
    return None


def split_label(name):
    s = str(name)
    if ":" in s:
        jid, formula = s.split(":", 1)
        return jid, formula
    return s, s


def main():
    parser = argparse.ArgumentParser(
        description="Filter repulsive pairs to crystal | BB or CH2I2 | gold",
    )
    parser.add_argument("--pairs", default="repulsive_lifshitz_pairs.npz")
    parser.add_argument("--out_csv", default="gold_liquid_pairs.csv")
    parser.add_argument(
        "--media",
        default="bromobenzene,diiodomethane",
        help="Liquids to keep (default: BB and CH2I2)",
    )
    parser.add_argument("--min_density", type=float, default=None, help="g/cm³ of the crystal")
    parser.add_argument("--min_atoms", type=int, default=None)
    parser.add_argument("--min_elements", type=int, default=None)
    parser.add_argument(
        "--gt_repulsive",
        action="store_true",
        help="Keep only rows with A_gt < 0",
    )
    parser.add_argument(
        "--include_builtins",
        action="store_true",
        help="Also keep PTFE|liquid|gold etc. Default: JARVIS crystals only",
    )
    parser.add_argument("--top_print", type=int, default=40)
    args = parser.parse_args()

    z = np.load(args.pairs, allow_pickle=True)
    names = z["names"]
    ia, ib, im = np.asarray(z["ia"]), np.asarray(z["ib"]), np.asarray(z["im"])
    a_nr = np.asarray(z["A_nr_zj"])
    a_gt = np.asarray(z["A_gt_zj"]) if "A_gt_zj" in z.files else None
    dens = np.asarray(z["density_g_cm3"])
    n_at = np.asarray(z["n_atoms"])
    n_el = np.asarray(z["n_elements"])
    vol = np.asarray(z["volume_A3"])
    gap = np.asarray(z["opt_gap"])
    mbj = np.asarray(z["mbj_gap"])
    eps0 = np.asarray(z["eps"][:, 0])
    jids = z["jid"] if "jid" in z.files else names
    formulas = z["formula"] if "formula" in z.files else names

    g = find_name(names, "gold")
    if g is None:
        raise SystemExit("No 'gold' material in the npz.")

    media_want = [p.strip() for p in args.media.split(",") if p.strip()]
    media_idx = []
    for m in media_want:
        i = find_name(names, m)
        if i is None:
            raise SystemExit(f"Medium {m!r} not in the npz.")
        media_idx.append(i)
    media_idx = np.asarray(media_idx, dtype=np.int32)

    mask = ((ia == g) | (ib == g)) & np.isin(im, media_idx)
    crystal = np.where(ia == g, ib, ia)
    mask &= crystal != g
    if not args.include_builtins:
        mask &= np.isfinite(n_at[crystal])

    if args.min_density is not None:
        mask &= np.isfinite(dens[crystal]) & (dens[crystal] >= args.min_density)
    if args.min_atoms is not None:
        mask &= np.isfinite(n_at[crystal]) & (n_at[crystal] >= args.min_atoms)
    if args.min_elements is not None:
        mask &= np.isfinite(n_el[crystal]) & (n_el[crystal] >= args.min_elements)
    if args.gt_repulsive:
        if a_gt is None:
            raise SystemExit("--gt_repulsive needs A_gt_zj in the npz")
        mask &= a_gt < 0.0

    idx = np.flatnonzero(mask)
    order = idx[np.argsort(a_nr[idx])]
    crystal = crystal[order]
    med = im[order]
    a_s = a_nr[order]
    gt_s = a_gt[order] if a_gt is not None else None

    print(f"Loaded {args.pairs}  {len(a_nr)} pairs")
    print(f"crystal | {{{', '.join(media_want)}}} | gold  →  {len(order)} triples")
    n_xtal = len(np.unique(crystal))
    print(f"  unique crystals: {n_xtal}")
    for mi in media_idx:
        n_m = int((med == mi).sum())
        if n_m == 0:
            continue
        sl = a_s[med == mi]
        print(
            f"  {str(names[int(mi)]):<16}  {n_m:5d}  "
            f"A_nr [{sl.min():.2f}, {sl.max():.2f}] zJ"
        )
    if gt_s is not None:
        n_agree = int((gt_s < 0).sum())
        print(f"  A_gt<0: {n_agree}/{len(order)}")

    cols = [
        "crystal", "jid", "formula", "medium", "A_nr_zj", "A_gt_zj", "gt_agrees",
        "eps0_crystal", "density_g_cm3", "n_atoms", "n_elements", "volume_A3",
        "opt_gap", "mbj_gap",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, c, m in zip(range(len(order)), crystal, med):
            c, m = int(c), int(m)
            jid, form = split_label(names[c])
            if "jid" in z.files:
                jid = str(jids[c])
            if "formula" in z.files:
                form = str(formulas[c])
            gt = float(gt_s[i]) if gt_s is not None else ""
            agree = (float(a_s[i]) < 0) and (gt_s is not None and float(gt_s[i]) < 0)
            w.writerow(
                [
                    str(names[c]),
                    jid,
                    form,
                    str(names[m]),
                    float(a_s[i]),
                    gt,
                    agree if gt_s is not None else "",
                    float(eps0[c]),
                    float(dens[c]),
                    float(n_at[c]),
                    float(n_el[c]),
                    float(vol[c]),
                    float(gap[c]),
                    float(mbj[c]),
                ]
            )
    print(f"Wrote {args.out_csv}  (most negative A_nr first)")

    nshow = min(int(args.top_print), len(order))
    if nshow:
        print(f"\n{'crystal':<28} {'medium':<16} {'A_nr':>10} {'A_gt':>10} {'ρ':>7} {'N':>5} {'ε(0)':>8}")
        print("-" * 92)
        for i in range(nshow):
            c, m = int(crystal[i]), int(med[i])
            gt = f"{float(gt_s[i]):10.3f}" if gt_s is not None else f"{'':>10}"
            print(
                f"{str(names[c])[:28]:<28} {str(names[m])[:16]:<16} "
                f"{float(a_s[i]):10.3f} {gt} {float(dens[c]):7.2f} "
                f"{float(n_at[c]):5.0f} {float(eps0[c]):8.3f}"
            )


if __name__ == "__main__":
    main()
