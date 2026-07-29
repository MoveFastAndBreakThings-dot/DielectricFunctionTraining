#!/usr/bin/env python3
"""
Export jid → band gaps for semiconductor filtering.

LOGIN NODE / laptop only if using JARVIS dft_3d (needs network).
Do NOT run this under sbatch.

Offline options (no network):
  python build_jid_gap_table.py --from_manifest raw_opt/manifest.jsonl
  python build_jid_gap_table.py --from_jsonl existing_gaps.jsonl --out tsenn_opt/jid_gaps.jsonl

Online (login node with jarvis-tools):
  python build_jid_gap_table.py --from_dft3d
"""

import argparse
import json
from pathlib import Path

from jarvis_optics_common import gap_float


def rows_from_manifest(path):
    rows = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            jid = row["jid"]
            rows[jid] = {
                "jid": jid,
                "mpid": row.get("mpid") or "",
                "opt_gap": gap_float(row.get("opt_gap")),
                "mbj_gap": gap_float(row.get("mbj_gap")),
            }
    return rows


def rows_from_jsonl(path):
    return rows_from_manifest(path)


def rows_from_dft3d():
    from jarvis_optics_common import load_dft3d_index

    index = load_dft3d_index()
    rows = {}
    for jid, info in index.items():
        rows[jid] = {
            "jid": jid,
            "mpid": info.get("mpid") or "",
            "opt_gap": gap_float(info.get("opt_gap")),
            "mbj_gap": gap_float(info.get("mbj_gap")),
        }
    return rows


def write_table(rows, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_with_opt = 0
    n_semi = 0
    with open(out, "w") as f:
        for jid in sorted(rows):
            row = rows[jid]
            opt_gap = row.get("opt_gap")
            if opt_gap is not None:
                n_with_opt += 1
                if opt_gap > 0.1:
                    n_semi += 1
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows → {out}")
    print(f"  with opt_gap: {n_with_opt}")
    print(f"  opt_gap > 0.1 eV: {n_semi}")


def main():
    parser = argparse.ArgumentParser(
        description="Build jid gap table (login/offline; never sbatch)"
    )
    parser.add_argument("--out", default="tsenn_opt/jid_gaps.jsonl")
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from_dft3d",
        action="store_true",
        help="JARVIS dft_3d (needs network + jarvis-tools; login node only)",
    )
    src.add_argument(
        "--from_manifest",
        default=None,
        help="Offline: raw_opt/manifest.jsonl (or similar)",
    )
    src.add_argument(
        "--from_jsonl",
        default=None,
        help="Offline: copy/normalize an existing gap jsonl",
    )
    args = parser.parse_args()

    if args.from_manifest:
        rows = rows_from_manifest(args.from_manifest)
    elif args.from_jsonl:
        rows = rows_from_jsonl(args.from_jsonl)
    elif args.from_dft3d:
        rows = rows_from_dft3d()
    else:
        # Prefer offline manifest if present, else dft_3d
        default_manifest = Path("raw_opt/manifest.jsonl")
        if default_manifest.is_file():
            print(f"Using offline manifest {default_manifest}")
            rows = rows_from_manifest(default_manifest)
        else:
            print("No raw_opt/manifest.jsonl; falling back to JARVIS dft_3d (needs net)")
            rows = rows_from_dft3d()

    write_table(rows, args.out)


if __name__ == "__main__":
    main()
