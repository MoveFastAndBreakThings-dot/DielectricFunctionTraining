"""
Step 2: parse raw JVASP zips → crystal graphs + KKR → LMDB (CPU-heavy, no net).

  python process_optics_lmdb.py --raw_dir raw_opt --out_dir latest_data_opt --workers 32
  python process_optics_lmdb.py --raw_dir raw_mbj --out_dir latest_data_mbj --workers 32

Use with sbatch (see sbatch_process_optics.sh).
"""

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lmdb
from tqdm import tqdm

from jarvis_optics_common import (
    GaussianDistance,
    gap_float,
    load_atom_init,
    make_lmdb_record,
    parse_zip_path,
    write_lmdb_metadata,
)

# Set in pool initializer (per process).
_STATE = {}


def _init_worker(atom_init_path, radius, max_num_nbr, n_xi_points, source, meta_path):
    meta = {}
    if meta_path and Path(meta_path).exists():
        with open(meta_path) as f:
            for line in f:
                row = json.loads(line)
                meta[row["jid"]] = row
    _STATE["atom_init"] = load_atom_init(atom_init_path)
    _STATE["gdf"] = GaussianDistance(dmin=0, dmax=radius, step=0.2)
    _STATE["radius"] = radius
    _STATE["max_num_nbr"] = max_num_nbr
    _STATE["n_xi_points"] = n_xi_points
    _STATE["source"] = source
    _STATE["meta"] = meta


def _process_one(zip_path):
    zip_path = Path(zip_path)
    jid = zip_path.name.split(".zip")[0]
    try:
        energy, real_d, imag_d, structure = parse_zip_path(zip_path)
        info = _STATE["meta"].get(jid, {})
        opt_gap = gap_float(info.get("opt_gap"))
        mbj_gap = gap_float(info.get("mbj_gap"))
        if _STATE["source"] == "mbj":
            gap_for_metal = mbj_gap if mbj_gap is not None else opt_gap
        else:
            gap_for_metal = opt_gap if opt_gap is not None else mbj_gap
        record = make_lmdb_record(
            jid,
            info.get("mpid") or "",
            gap_for_metal,
            mbj_gap,
            structure,
            energy,
            real_d,
            imag_d,
            _STATE["atom_init"],
            _STATE["gdf"],
            _STATE["radius"],
            _STATE["max_num_nbr"],
            _STATE["n_xi_points"],
        )
        return jid, record, None
    except Exception as exc:
        return jid, None, str(exc)


def load_existing_jids(lmdb_path):
    done = set()
    n = 0
    if not Path(lmdb_path).exists():
        return done, n
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        meta_raw = txn.get(b"__metadata__")
        if meta_raw is not None:
            n = pickle.loads(meta_raw).get("total_samples", 0)
        for i in range(n):
            raw = txn.get(f"{i}".encode("utf-8"))
            if raw is None:
                break
            done.add(pickle.loads(raw).get("jid"))
    env.close()
    return done, n


def run_process(
    raw_dir,
    out_dir,
    source,
    atom_init_path="atom_init.json",
    workers=32,
    radius=8.0,
    max_num_nbr=12,
    n_xi_points=500,
    map_size=200 * 1024**3,
    resume=True,
    limit=None,
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "crystal_graphs.lmdb"
    meta_path = raw_dir / "manifest.jsonl"

    zips = sorted(raw_dir.glob("JVASP-*.zip"))
    done, start_idx = load_existing_jids(lmdb_path) if resume else (set(), 0)
    todo = [z for z in zips if z.name.split(".zip")[0] not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"Raw zips: {len(zips)}, already in LMDB: {len(done)}, to process: {len(todo)}")
    print(f"Workers: {workers} → {lmdb_path}")

    atom_init = load_atom_init(atom_init_path)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)

    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=False)
    idx = start_idx
    ok = fail = 0
    energy_array = kkr_xi_array = None
    if start_idx > 0:
        with env.begin() as txn:
            prev = pickle.loads(txn.get(b"__metadata__"))
            energy_array = prev.get("energy_array")
            kkr_xi_array = prev.get("kkr_xi_array")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(atom_init_path, radius, max_num_nbr, n_xi_points, source, str(meta_path)),
    ) as pool:
        futures = {pool.submit(_process_one, str(z)): z for z in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"proc-{source}"):
            jid, record, err = fut.result()
            if err is not None:
                fail += 1
                print(f"  FAIL {jid}: {err}")
                continue
            if energy_array is None:
                energy_array = record["opt_en"]
            if kkr_xi_array is None:
                kkr_xi_array = record["kkr_xi"]
            with env.begin(write=True) as txn:
                txn.put(f"{idx}".encode("utf-8"), pickle.dumps(record))
                write_lmdb_metadata(
                    txn, idx + 1, energy_array, kkr_xi_array,
                    atom_init, gdf, radius, max_num_nbr,
                )
            idx += 1
            ok += 1

    env.close()
    summary = {
        "source": source,
        "raw_dir": str(raw_dir),
        "lmdb_path": str(lmdb_path),
        "ok": ok,
        "fail": fail,
        "total_samples": idx,
    }
    with open(out_dir / "process_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Process raw optics zips → LMDB")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--source", choices=("opt", "mbj"), required=True)
    parser.add_argument("--atom_init", default="atom_init.json")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max_num_nbr", type=int, default=12)
    parser.add_argument("--n_xi_points", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    run_process(
        args.raw_dir,
        args.out_dir,
        args.source,
        atom_init_path=args.atom_init,
        workers=args.workers,
        radius=args.radius,
        max_num_nbr=args.max_num_nbr,
        n_xi_points=args.n_xi_points,
        resume=not args.no_resume,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
