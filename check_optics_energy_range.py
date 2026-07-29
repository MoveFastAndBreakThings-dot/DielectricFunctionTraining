"""
Check imaginary-axis / optical energy grids in an optics LMDB.

Reports min/max of opt_en (real-frequency eV grid from VASP LOPTICS)
across the dataset, and how many materials reach above a threshold (default 100 eV).

Usage:
  python check_optics_energy_range.py --lmdb_path latest_data_opt/crystal_graphs.lmdb
  python check_optics_energy_range.py --lmdb_path latest_data_opt/crystal_graphs.lmdb --sample 500
"""

import argparse
import pickle
import random

import lmdb
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Check opt_en energy range in LMDB")
    parser.add_argument("--lmdb_path", default="latest_data_opt/crystal_graphs.lmdb")
    parser.add_argument("--threshold_ev", type=float, default=100.0)
    parser.add_argument(
        "--sample", type=int, default=None,
        help="If set, only check this many random materials (faster). Default: all.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        n = int(meta["total_samples"])
        indices = list(range(n))
        if args.sample is not None and args.sample < n:
            random.seed(args.seed)
            indices = random.sample(indices, args.sample)

        maxima = []
        minima = []
        n_pts = []
        above = 0
        missing = 0
        for i in indices:
            data = pickle.loads(txn.get(f"{i}".encode("utf-8")))
            en = data.get("opt_en")
            if en is None:
                missing += 1
                continue
            en = np.asarray(en, dtype=np.float64)
            e_min, e_max = float(en.min()), float(en.max())
            minima.append(e_min)
            maxima.append(e_max)
            n_pts.append(len(en))
            if e_max >= args.threshold_ev:
                above += 1

    env.close()
    maxima = np.asarray(maxima)
    minima = np.asarray(minima)
    n_pts = np.asarray(n_pts)
    checked = len(maxima)

    print(f"LMDB: {args.lmdb_path}")
    print(f"Total samples in LMDB: {n}")
    print(f"Checked: {checked} (missing opt_en: {missing})")
    if checked == 0:
        return
    print(f"opt_en points per material: min={n_pts.min()} median={int(np.median(n_pts))} max={n_pts.max()}")
    print(f"ω_min (eV): global_min={minima.min():.6f} median={np.median(minima):.6f} max_of_mins={minima.max():.6f}")
    print(f"ω_max (eV): global_min={maxima.min():.6f} median={np.median(maxima):.6f} global_max={maxima.max():.6f}")
    print(
        f"Materials with ω_max >= {args.threshold_ev:g} eV: "
        f"{above}/{checked} ({100.0 * above / checked:.1f}%)"
    )
    # percentiles of max energy
    for p in (1, 5, 50, 95, 99):
        print(f"  ω_max p{p}: {np.percentile(maxima, p):.4f} eV")


if __name__ == "__main__":
    main()
