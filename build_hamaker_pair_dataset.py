"""
Build a dataset of random material pairs with ground-truth pairwise Hamaker constants.

Pairwise Lifshitz formula (vacuum gap, finite T):

    A_12 = (3/2) k_B T [ (1/2) Δ_1(0)Δ_2(0) + Σ_{n≥1} Δ_1(n)Δ_2(n) ]   (zJ saved)

Material count and default pair count are read from the LMDB (no hardcoded sizes).

Outputs in --out_dir (default: meta_hamaker_pairs/):
  pairs.npz     — idx_a, idx_b, a_true_zj, mpid_a, mpid_b (object arrays)
  metadata.json — generation settings
  splits.json   — train/val/test indices into the pair list

Usage:
    python3 build_hamaker_pair_dataset.py
    python3 build_hamaker_pair_dataset.py --n_pairs 100000 --out_dir meta_hamaker_pairs
"""

import argparse
import json
import os

import numpy as np

from compare_hamaker_constants import T_K, hamaker_pair_constant_zj, train_val_test_split
from lmdb_dataset import LMDBCrystalDataset

DEFAULT_PAIR_CAP = 50_000


def max_pair_count(n_materials, allow_self_pairs=False):
    if allow_self_pairs:
        return n_materials * n_materials
    return n_materials * (n_materials - 1) // 2


def default_n_pairs(n_materials, allow_self_pairs=False, cap=DEFAULT_PAIR_CAP):
    return min(cap, max_pair_count(n_materials, allow_self_pairs))


def sample_pairs(n_materials, n_pairs, rng, allow_self_pairs=False):
    if allow_self_pairs:
        idx_a = rng.randint(0, n_materials, size=n_pairs)
        idx_b = rng.randint(0, n_materials, size=n_pairs)
        return idx_a, idx_b

    max_unique = max_pair_count(n_materials, allow_self_pairs=False)
    if n_pairs <= max_unique:
        all_i, all_j = np.triu_indices(n_materials, k=1)
        perm = rng.permutation(len(all_i))[:n_pairs]
        return all_i[perm], all_j[perm]

    idx_a = rng.randint(0, n_materials, size=n_pairs)
    idx_b = rng.randint(0, n_materials, size=n_pairs)
    same = idx_a == idx_b
    idx_b[same] = (idx_b[same] + 1) % n_materials
    return idx_a, idx_b


def main():
    parser = argparse.ArgumentParser(description="Build random material-pair Hamaker dataset")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--out_dir", type=str, default="meta_hamaker_pairs")
    parser.add_argument("--n_pairs", type=int, default=None,
                        help="Number of pairs to sample (default: min(50000, all unique unordered pairs))")
    parser.add_argument("--pair_cap", type=int, default=DEFAULT_PAIR_CAP,
                        help="Upper bound used when --n_pairs is omitted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_matsubara", type=int, default=500)
    parser.add_argument("--temperature_k", type=float, default=T_K)
    parser.add_argument("--target_components", type=str, default="xx")
    parser.add_argument("--allow_self_pairs", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    target_components = tuple(args.target_components.split(","))

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True)
    n_materials = len(dataset)
    max_pairs = max_pair_count(n_materials, args.allow_self_pairs)
    n_pairs = args.n_pairs if args.n_pairs is not None else default_n_pairs(
        n_materials, args.allow_self_pairs, cap=args.pair_cap,
    )
    if n_pairs > max_pairs:
        print(f"  capping n_pairs {n_pairs} -> {max_pairs} (all unique pairs in LMDB)")
        n_pairs = max_pairs

    xi = np.asarray(dataset.kkr_xi_array, dtype=np.float64)
    hamaker_kw = dict(t_k=args.temperature_k, n_max=args.n_matsubara)

    print(f"LMDB: {args.lmdb_path}")
    print(f"  materials (auto-detected): {n_materials}")
    print(f"  max unique pairs: {max_pairs:,}")
    print(f"  sampling: {n_pairs:,} pairs  |  ξ points: {len(xi)}")

    idx_a, idx_b = sample_pairs(n_materials, n_pairs, rng, args.allow_self_pairs)
    a_true = np.empty(n_pairs, dtype=np.float64)
    mpid_a = np.empty(n_pairs, dtype=object)
    mpid_b = np.empty(n_pairs, dtype=object)

    for i in range(n_pairs):
        ia, ib = int(idx_a[i]), int(idx_b[i])
        _, tgt_a, _ = dataset[ia]
        _, tgt_b, _ = dataset[ib]
        eps_a = tgt_a[0]
        eps_b = tgt_b[0]
        a_true[i] = hamaker_pair_constant_zj(xi, eps_a, eps_b, **hamaker_kw)
        info_a = dataset.get_sample_info(ia)
        info_b = dataset.get_sample_info(ib)
        mpid_a[i] = info_a["mpid"]
        mpid_b[i] = info_b["mpid"]
        if (i + 1) % 1000 == 0 or i + 1 == n_pairs:
            print(f"  {i + 1}/{n_pairs}")

    dataset.close()

    np.savez(
        os.path.join(args.out_dir, "pairs.npz"),
        idx_a=idx_a.astype(np.int32),
        idx_b=idx_b.astype(np.int32),
        a_true_zj=a_true,
        mpid_a=mpid_a,
        mpid_b=mpid_b,
    )

    train_idx, val_idx, test_idx = train_val_test_split(
        n_pairs, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed,
    )
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump(splits, f)

    metadata = {
        "lmdb_path": args.lmdb_path,
        "n_pairs": n_pairs,
        "n_materials": n_materials,
        "max_unique_pairs": max_pairs,
        "pair_cap": args.pair_cap,
        "seed": args.seed,
        "n_matsubara": args.n_matsubara,
        "temperature_k": args.temperature_k,
        "target_components": list(target_components),
        "allow_self_pairs": args.allow_self_pairs,
        "a_true_zj_mean": float(a_true.mean()),
        "a_true_zj_std": float(a_true.std()),
        "a_true_zj_min": float(a_true.min()),
        "a_true_zj_max": float(a_true.max()),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved {args.out_dir}/pairs.npz")
    print(f"  A_12 (zJ): mean={a_true.mean():.2f}  std={a_true.std():.2f}  "
        f"min={a_true.min():.2f}  max={a_true.max():.2f}")
    print(f"  splits: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")


if __name__ == "__main__":
    main()
