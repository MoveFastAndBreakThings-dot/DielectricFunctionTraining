"""
Step 1: download JARVIS optics VASP zips to disk (login node / machine with net).

  python download_optics_raw.py --source opt --out_dir raw_opt --workers 32
  python download_optics_raw.py --source mbj --out_dir raw_mbj --workers 32

Writes:
  raw_opt/JVASP-XXXXX.zip
  raw_opt/manifest.jsonl   # jid, mpid, gaps for the process step
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from jarvis_optics_common import (
    download_bytes,
    gap_float,
    get_download_entries,
    load_dft3d_index,
    load_raw_catalog,
)


def save_one(entry, out_dir, meta, source, timeout):
    name = entry["name"]
    if not name.endswith(".zip"):
        name = name + ".zip"
    jid = name.split(".zip")[0]
    dest = Path(out_dir) / name
    if dest.exists() and dest.stat().st_size > 4:
        with open(dest, "rb") as f:
            magic = f.read(2)
        if magic == b"PK":
            info = meta.get(jid, {})
            return jid, "skip", _manifest_row(jid, info, source)

    zip_bytes = download_bytes(entry["download_url"], timeout=timeout)
    tmp = dest.with_suffix(".zip.partial")
    with open(tmp, "wb") as f:
        f.write(zip_bytes)
    tmp.replace(dest)

    info = meta.get(jid, {})
    return jid, "ok", _manifest_row(jid, info, source)


def _manifest_row(jid, info, source):
    opt_gap = gap_float(info.get("opt_gap"))
    mbj_gap = gap_float(info.get("mbj_gap"))
    if source == "mbj":
        gap_for_metal = mbj_gap if mbj_gap is not None else opt_gap
    else:
        gap_for_metal = opt_gap if opt_gap is not None else mbj_gap
    return {
        "jid": jid,
        "mpid": info.get("mpid") or "",
        "opt_gap": gap_for_metal,
        "mbj_gap": mbj_gap,
    }


def run_download(source, out_dir, workers=32, limit=None, timeout=120):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    key, entries = get_download_entries(source)
    if limit is not None:
        entries = entries[:limit]
    print(f"Downloading up to {len(entries)} zips → {out_dir} ({workers} threads)")

    meta = load_dft3d_index()
    # existing manifest rows (resume metadata)
    known = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                row = json.loads(line)
                known[row["jid"]] = row

    ok = skip = fail = 0
    with open(manifest_path, "a") as man_f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(save_one, e, out_dir, meta, source, timeout): e
                for e in entries
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"dl-{source}"):
                entry = futures[fut]
                try:
                    jid, status, row = fut.result()
                    if status == "ok":
                        ok += 1
                    else:
                        skip += 1
                    if jid not in known:
                        man_f.write(json.dumps(row) + "\n")
                        man_f.flush()
                        known[jid] = row
                except Exception as exc:
                    fail += 1
                    jid = entry["name"].split(".zip")[0]
                    print(f"  FAIL {jid}: {exc}")

    summary = {
        "source": source,
        "catalog_key": key,
        "out_dir": str(out_dir),
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "n_zips": len(list(out_dir.glob("JVASP-*.zip"))),
    }
    with open(out_dir / "download_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Download JARVIS optics zips only")
    parser.add_argument("--source", choices=("opt", "mbj"), required=True)
    parser.add_argument("--out_dir", default=None, help="default: raw_opt or raw_mbj")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--list_keys", action="store_true")
    args = parser.parse_args()

    if args.list_keys:
        fls = load_raw_catalog()
        for k, v in fls.items():
            print(f"{k}: {len(v) if isinstance(v, list) else '?'}")
        return

    out_dir = args.out_dir or (f"raw_{args.source}")
    run_download(args.source, out_dir, args.workers, args.limit, args.timeout)


if __name__ == "__main__":
    main()
