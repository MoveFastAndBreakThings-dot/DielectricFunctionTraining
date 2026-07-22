"""
Download latest JARVIS-DFT frequency-dependent optical spectra and write
ENdiele-style JSON for preprocess_dataset.py.

Unlike the 2018 ENdiele20.json dump, jarvis-tools' dft_3d metadata does not
include full ε(E) arrays. Those live on per-material NIST XML pages (and as
raw VASP files on Figshare). This script:

  1) downloads latest dft_3d metadata
  2) keeps materials with OPT optics (epsx != 'na')
  3) fetches ε(E) from JARVIS webpages
  4) writes data/jarvis_optics_latest.json

Usage:
    python3 download_jarvis_optics.py
    python3 download_jarvis_optics.py --limit 20   # smoke test
    python3 preprocess_dataset.py --input_dir data --output_dir processed_data
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".pydeps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from jarvis.db.webpages import Webpage
from tqdm import tqdm


COMPONENTS = [
    ("xx", "real_xx", "imag_xx"),
    ("yy", "real_yy", "imag_yy"),
    ("zz", "real_zz", "imag_zz"),
    ("xy", "real_xy", "imag_xy"),
    ("yz", "real_yz", "imag_yz"),
    ("zx", "real_zx", "imag_zx"),
]


def atoms_to_cif(atoms_dict):
    atoms = Atoms.from_dict(atoms_dict)
    a, b, c, alpha, beta, gamma = atoms.lattice.parameters
    lines = [
        "data_image0",
        "_symmetry_space_group_name_H-M    'P 1'",
        "_symmetry_Int_Tables_number       1",
        f"_cell_length_a                   {a:.10f}",
        f"_cell_length_b                   {b:.10f}",
        f"_cell_length_c                   {c:.10f}",
        f"_cell_angle_alpha                {alpha:.10f}",
        f"_cell_angle_beta                 {beta:.10f}",
        f"_cell_angle_gamma                {gamma:.10f}",
        "loop_",
        " _symmetry_equiv_pos_as_xyz",
        "  'x, y, z'",
        "loop_",
        " _atom_site_type_symbol",
        " _atom_site_label",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
    ]
    for i, (el, frac) in enumerate(zip(atoms.elements, atoms.frac_coords), start=1):
        lines.append(
            f"  {el}  {el}{i}  {frac[0]:.10f}  {frac[1]:.10f}  {frac[2]:.10f}"
        )
    return "\n".join(lines) + "\n"


def has_value(x):
    return x is not None and x != "na" and x != "NA"


def spectrum_to_sample(entry, opt_spec, mbj_spec=None):
    sample = {
        "jid": entry.get("jid", ""),
        "mpid": entry.get("reference", entry.get("mpid", "")),
        "cif": atoms_to_cif(entry["atoms"]),
        "opt_en": opt_spec["energies"].tolist(),
        "opt_gap": entry.get("optb88vdw_bandgap", None),
        "mbj_gap": entry.get("mbj_bandgap", None),
    }
    for suffix, real_key, imag_key in COMPONENTS:
        sample[f"opt_real{suffix}"] = opt_spec[real_key].tolist()
        sample[f"opt_imag{suffix}"] = opt_spec[imag_key].tolist()
    if mbj_spec is not None:
        sample["mbj_en"] = mbj_spec["energies"].tolist()
        for suffix, real_key, imag_key in COMPONENTS:
            sample[f"mbj_real{suffix}"] = mbj_spec[real_key].tolist()
            sample[f"mbj_imag{suffix}"] = mbj_spec[imag_key].tolist()
    return sample


def load_checkpoint(path):
    if not path.exists():
        return [], set()
    with open(path) as f:
        rows = json.load(f)
    return rows, {r["jid"] for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--cache_dir", default="data/jarvis_cache")
    parser.add_argument("--limit", type=int, default=0, help="0 = all materials with OPT optics")
    parser.add_argument("--include_mbj", action="store_true", help="Also fetch MBJ spectra when available")
    parser.add_argument("--save_every", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "jarvis_optics_latest.json"

    print("Downloading latest dft_3d metadata (cached under data/jarvis_cache)...")
    dft_3d = data("dft_3d", store_dir=args.cache_dir)

    candidates = [e for e in dft_3d if has_value(e.get("epsx"))]
    print(f"Materials with OPT optics (epsx present): {len(candidates)}")
    if args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"Limiting to first {len(candidates)} materials")

    rows, done = load_checkpoint(out_json)
    print(f"Resuming with {len(done)} already downloaded")

    failed = 0
    for entry in tqdm(candidates, desc="Fetching optics XML"):
        jid = entry["jid"]
        if jid in done:
            continue
        try:
            page = Webpage(jid=jid)
            opt_spec = page.get_dft_semilocal_dielectric_function()
            mbj_spec = None
            if args.include_mbj and has_value(entry.get("mepsx")):
                mbj_spec = page.get_dft_mbj_dielectric_function()
            rows.append(spectrum_to_sample(entry, opt_spec, mbj_spec))
            done.add(jid)
        except Exception as e:
            failed += 1
            print(f"\nSkip {jid}: {e}")
            continue

        if len(rows) % args.save_every == 0:
            with open(out_json, "w") as f:
                json.dump(rows, f)

    with open(out_json, "w") as f:
        json.dump(rows, f)

    print(f"Wrote {len(rows)} materials to {out_json} ({failed} failed)")
    print("Next:")
    print(f"  python3 preprocess_dataset.py --input_dir {out_dir} --output_dir processed_data")


if __name__ == "__main__":
    main()
