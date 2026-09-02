#!/usr/bin/env python3
"""
Overview plots for repulsive_lifshitz_pairs.npz.

  python plot_repulsive_pairs.py
  python plot_repulsive_pairs.py --pairs repulsive_lifshitz_pairs.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def short(name, n=24):
    s = str(name)
    return s if len(s) <= n else s[: n - 1] + "…"


def top_counts(ids, names, k=12):
    u, c = np.unique(ids, return_counts=True)
    order = np.argsort(-c)[:k]
    return [short(names[int(u[i])]) for i in order], c[order]


def main():
    parser = argparse.ArgumentParser(description="Visualize A|medium|B repulsive pairs")
    parser.add_argument("--pairs", default="repulsive_lifshitz_pairs.npz")
    parser.add_argument("--out_png", default="repulsive_pairs_overview.png")
    parser.add_argument("--out_csv", default="repulsive_pairs_browse.csv")
    parser.add_argument("--top", type=int, default=3000, help="Most negative A rows in CSV")
    parser.add_argument("--sample", type=int, default=3000, help="Random extra rows in CSV")
    parser.add_argument("--hex_sample", type=int, default=400000)
    args = parser.parse_args()

    path = Path(args.pairs)
    z = np.load(path, allow_pickle=True)
    names = z["names"]
    ia, ib, im = z["ia"], z["ib"], z["im"]
    a_nr = z["A_nr_zj"]
    a_gt = z["A_gt_zj"] if "A_gt_zj" in z.files else None
    dens = z["density_g_cm3"]
    n_at = z["n_atoms"]
    eps0 = z["eps"][:, 0]
    n = len(a_nr)
    print(f"Loaded {path}  {n} pairs  {len(names)} materials")

    lo = np.where(eps0[ia] <= eps0[ib], ia, ib)
    hi = np.where(eps0[ia] <= eps0[ib], ib, ia)

    print("\n--- media ---")
    for mid in np.unique(im):
        mask = im == mid
        print(
            f"  {names[int(mid)]:<16}  {int(mask.sum()):>12}  "
            f"A_nr [{a_nr[mask].min():.1f}, {a_nr[mask].max():.1f}] zJ"
        )

    gold = [i for i, n_ in enumerate(names) if str(n_) == "gold"]
    if gold:
        g = gold[0]
        gmask = (ia == g) | (ib == g)
        print(f"\n--- gold as a body: {int(gmask.sum())} pairs ---")
        if gmask.any():
            print(f"  A_nr [{a_nr[gmask].min():.1f}, {a_nr[gmask].max():.1f}] zJ")

    print("\n--- most common LOW-ε body (the PTFE role) ---")
    labs, cnt = top_counts(lo, names, 15)
    for lab, c in zip(labs, cnt):
        print(f"  {c:10d}  {lab}")
    print("\n--- most common HIGH-ε body (the metal role) ---")
    labs, cnt = top_counts(hi, names, 15)
    for lab, c in zip(labs, cnt):
        print(f"  {c:10d}  {lab}")

    rng = np.random.default_rng(0)
    n_hex = min(int(args.hex_sample), n)
    hs = rng.choice(n, size=n_hex, replace=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2))

    ax = axes[0, 0]
    for mid in np.unique(im):
        mask = im == mid
        ax.hist(
            a_nr[mask], bins=80, range=(float(a_nr.min()), 0.0),
            histtype="step", label=str(names[int(mid)]), density=True,
        )
    ax.set_xlabel("A_nr (zJ), negative = repulsive")
    ax.set_ylabel("density")
    ax.set_title("Hamaker distribution by medium")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labs, cnt = top_counts(lo, names, 12)
    ax.barh(range(len(labs))[::-1], cnt[::-1])
    ax.set_yticks(range(len(labs))[::-1], labs[::-1], fontsize=8)
    ax.set_xlabel("pair count")
    ax.set_title("Top low-ε bodies")

    ax = axes[0, 2]
    labs, cnt = top_counts(hi, names, 12)
    ax.barh(range(len(labs))[::-1], cnt[::-1])
    ax.set_yticks(range(len(labs))[::-1], labs[::-1], fontsize=8)
    ax.set_xlabel("pair count")
    ax.set_title("Top high-ε bodies")

    ax = axes[1, 0]
    dlo, dhi = dens[lo[hs]], dens[hi[hs]]
    ok = np.isfinite(dlo) & np.isfinite(dhi)
    ax.hexbin(dlo[ok], dhi[ok], gridsize=50, bins="log", mincnt=1)
    ax.set_xlabel("low-ε density (g/cm³)")
    ax.set_ylabel("high-ε density (g/cm³)")
    ax.set_title(f"Density of both walls ({n_hex} sample)")
    ax.axvline(2.0, color="k", lw=0.6, ls="--")
    ax.axhline(2.0, color="k", lw=0.6, ls="--")

    ax = axes[1, 1]
    ax.hexbin(eps0[lo[hs]], eps0[hi[hs]], gridsize=50, bins="log", mincnt=1)
    ax.set_xlabel("low-ε  ε(ξ=0)")
    ax.set_ylabel("high-ε  ε(ξ=0)")
    ax.set_title("Static ε of both walls")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax = axes[1, 2]
    if gold:
        gmask = (ia == gold[0]) | (ib == gold[0])
        other = np.where(ia == gold[0], ib, ia)
        od = dens[other[gmask]]
        oa = a_nr[gmask]
        ok = np.isfinite(od)
        if ok.sum() > 20:
            ax.hexbin(od[ok], oa[ok], gridsize=40, bins="log", mincnt=1)
            ax.set_xlabel("crystal density (g/cm³)")
            ax.set_ylabel("A_nr vs gold (zJ)")
            ax.set_title(f"crystal | liquid | gold  ({int(gmask.sum())} pairs)")
        else:
            ax.text(0.5, 0.5, "too few gold pairs", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        nlo, nhi = n_at[lo[hs]], n_at[hi[hs]]
        ok = np.isfinite(nlo) & np.isfinite(nhi)
        ax.hexbin(nlo[ok], nhi[ok], gridsize=40, bins="log", mincnt=1)
        ax.set_xlabel("low-ε n_atoms")
        ax.set_ylabel("high-ε n_atoms")
        ax.set_title("Cell size of both walls")

    fig.suptitle(f"{n} repulsive A|medium|B pairs  ({path.name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"\nWrote {args.out_png}")

    top_n = min(int(args.top), n)
    top_idx = np.argpartition(a_nr, top_n - 1)[:top_n]
    top_idx = top_idx[np.argsort(a_nr[top_idx])]
    n_rand = min(int(args.sample), n)
    rand_idx = rng.choice(n, size=n_rand, replace=False)
    shown = np.unique(np.concatenate([top_idx, rand_idx]))

    cols = [
        "rank_kind", "A", "medium", "B", "A_nr_zj", "A_gt_zj",
        "eps0_A", "eps0_B", "dens_A", "dens_B", "n_atoms_A", "n_atoms_B",
        "low_eps_body", "high_eps_body",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        top_set = set(int(i) for i in top_idx)
        for k in shown:
            k = int(k)
            kind = "top" if k in top_set else "random"
            gt = float(a_gt[k]) if a_gt is not None else ""
            w.writerow(
                [
                    kind,
                    names[int(ia[k])],
                    names[int(im[k])],
                    names[int(ib[k])],
                    float(a_nr[k]),
                    gt,
                    float(eps0[int(ia[k])]),
                    float(eps0[int(ib[k])]),
                    float(dens[int(ia[k])]),
                    float(dens[int(ib[k])]),
                    float(n_at[int(ia[k])]),
                    float(n_at[int(ib[k])]),
                    names[int(lo[k])],
                    names[int(hi[k])],
                ]
            )
    print(f"Wrote {args.out_csv}  ({len(shown)} rows: {top_n} most negative + sample)")
    print("Open the CSV for a spreadsheet view; dashed lines on density plot are 2 g/cm³.")


if __name__ == "__main__":
    main()
