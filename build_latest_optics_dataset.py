"""
Download JARVIS raw LOPTICS / TBmBJ VASP zips and write preprocessed LMDB
(crystal graphs + KKR ε(iξ)) — same format as preprocess_dataset.py.

Outputs:
  latest_data_opt/crystal_graphs.lmdb
  latest_data_mbj/crystal_graphs.lmdb

Server usage:
  pip install jarvis-tools requests tqdm pymatgen lmdb scipy
  python build_latest_optics_dataset.py --list_keys
  python build_latest_optics_dataset.py --source opt --limit 20
  python build_latest_optics_dataset.py --source mbj
"""

import argparse
import io
import json
import os
import pickle
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import lmdb
import numpy as np
import requests
from tqdm import tqdm

from preprocess_dataset import (
    GaussianDistance,
    apply_kkr,
    get_atom_features,
    get_neighbor_features,
    load_atom_init,
)

COMPONENTS = ("xx", "yy", "zz", "xy", "yz", "zx")
COL_INDEX = {"xx": 1, "yy": 2, "zz": 3, "xy": 4, "yz": 5, "zx": 6}

OPT_KEYS = ("OPT-LOPTICS",)
MBJ_KEYS = ("TBMBJ", "OPT-TBMBJ", "MBJ", "TB-MBJ", "TBmBJ")
FIGSHARE_PROJECT = {
    "opt": 86378,
    "mbj": 84020,
}


def _fix_url(url):
    return url.replace("ndownloader.figshare.com", "figshare.com/ndownloader")


def load_raw_catalog():
    from jarvis.db.figshare import data

    fls = data("raw_files")
    if not isinstance(fls, dict):
        raise RuntimeError(f"raw_files expected dict, got {type(fls)}")
    return fls


def resolve_catalog_key(fls, source):
    candidates = OPT_KEYS if source == "opt" else MBJ_KEYS
    for cand in candidates:
        if cand in fls:
            return cand
    needle = "LOPTICS" if source == "opt" else "MBJ"
    for k in fls.keys():
        if needle.upper() in str(k).upper():
            return k
    return None


def list_figshare_project_entries(project_id):
    entries = []
    page = 1
    while True:
        url = (
            f"https://api.figshare.com/v2/projects/{project_id}/articles"
            f"?page_size=100&page={page}"
        )
        arts = requests.get(url, timeout=60).json()
        if not arts:
            break
        for art in arts:
            detail = requests.get(art["url_public_api"], timeout=60).json()
            for f in detail.get("files", []):
                name = f.get("name", "")
                if not name.endswith(".zip") or not name.startswith("JVASP-"):
                    continue
                entries.append({"name": name, "download_url": f["download_url"]})
        if len(arts) < 100:
            break
        page += 1
    return entries


def get_download_entries(source):
    fls = load_raw_catalog()
    key = resolve_catalog_key(fls, source)
    if key is not None:
        entries = [e for e in fls[key] if isinstance(e, dict) and "download_url" in e]
        print(f"Using raw_files[{key!r}]: {len(entries)} zips")
        return key, entries
    project_id = FIGSHARE_PROJECT[source]
    print(f"No raw_files key for {source}; falling back to Figshare project {project_id}")
    entries = list_figshare_project_entries(project_id)
    print(f"Figshare project {project_id}: {len(entries)} zips")
    return f"figshare_project_{project_id}", entries


def load_dft3d_index():
    from jarvis.db.figshare import data

    print("Loading dft_3d metadata index...")
    index = {}
    for row in data("dft_3d"):
        jid = row.get("jid")
        if not jid:
            continue
        index[jid] = {
            "mpid": row.get("mpid") or row.get("reference") or "",
            "opt_gap": row.get("optb88vdw_bandgap"),
            "mbj_gap": row.get("mbj_bandgap"),
            "atoms": row.get("atoms"),
        }
    print(f"  indexed {len(index)} dft_3d entries")
    return index


def atoms_to_structure(atoms_dict_or_obj):
    from jarvis.core.atoms import Atoms
    from pymatgen.core import Lattice, Structure

    if isinstance(atoms_dict_or_obj, dict):
        atoms = Atoms.from_dict(atoms_dict_or_obj)
    else:
        atoms = atoms_dict_or_obj
    return Structure(
        Lattice(atoms.lattice_mat),
        atoms.elements,
        atoms.cart_coords,
        coords_are_cartesian=True,
    )


def parse_zip(zip_bytes):
    """Return (energy, real_dict, imag_dict, structure)."""
    from jarvis.io.vasp.inputs import Poscar
    from jarvis.io.vasp.outputs import Vasprun

    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    names = z.namelist()
    vrun_name = next((n for n in names if n.endswith("vasprun.xml")), None)
    if vrun_name is None:
        raise FileNotFoundError(f"No vasprun.xml in zip; files={names[:10]}")

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False) as tmp:
        tmp.write(z.read(vrun_name))
        vpath = tmp.name
    try:
        vrun = Vasprun(vpath)
        reals, imags = vrun.dielectric_loptics
    finally:
        os.remove(vpath)

    reals = np.asarray(reals, dtype=np.float64)
    imags = np.asarray(imags, dtype=np.float64)
    if reals.ndim != 2 or reals.shape[1] < 7:
        raise ValueError(f"Unexpected dielectric shape: {reals.shape}")

    energy = reals[:, 0].astype(np.float32)
    real_d, imag_d = {}, {}
    for comp in COMPONENTS:
        j = COL_INDEX[comp]
        real_d[comp] = reals[:, j].astype(np.float32)
        imag_d[comp] = imags[:, j].astype(np.float32)

    structure = None
    poscar_name = next((n for n in names if n.endswith("POSCAR") or n == "POSCAR"), None)
    if poscar_name is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vasp", delete=False) as tmp:
            tmp.write(z.read(poscar_name).decode("utf-8"))
            ppath = tmp.name
        try:
            structure = atoms_to_structure(Poscar.from_file(ppath).atoms)
        finally:
            os.remove(ppath)
    if structure is None:
        structure = atoms_to_structure(vrun.all_structures[-1])

    return energy, real_d, imag_d, structure


def _gap_float(x):
    if x in (None, "na"):
        return None
    return float(x)


def make_lmdb_record(
    jid, mpid, opt_gap, mbj_gap, structure, energy, real_d, imag_d,
    atom_init, gdf, radius, max_num_nbr, n_xi_points,
):
    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(structure, radius, max_num_nbr, gdf)

    targets = {}
    for comp in COMPONENTS:
        targets[f"real_{comp}"] = real_d[comp]
        targets[f"imag_{comp}"] = imag_d[comp]

    kkr_targets = {}
    xi = None
    for comp in COMPONENTS:
        xi, kkr_vals = apply_kkr(energy, imag_d[comp], n_xi_points=n_xi_points)
        kkr_targets[f"kkr_{comp}"] = kkr_vals

    return {
        "mpid": mpid or "",
        "jid": jid,
        "atom_fea": atom_fea,
        "nbr_fea": nbr_fea,
        "nbr_fea_idx": nbr_fea_idx,
        "positions": np.array(structure.cart_coords, dtype=np.float32),
        "cell": np.array(structure.lattice.matrix, dtype=np.float32),
        "atomic_numbers": np.array([site.specie.Z for site in structure], dtype=np.int64),
        "pbc": np.array(structure.pbc, dtype=bool),
        "targets": targets,
        "opt_en": energy,
        "opt_gap": opt_gap,
        "mbj_gap": mbj_gap,
        "n_atoms": len(atom_fea),
        "formula": structure.formula,
        "kkr_targets": kkr_targets,
        "kkr_xi": xi,
    }


def download_raw(entry, meta, source, timeout=120):
    """Download + parse only (graph/KKR done in the writer thread)."""
    jid = entry["name"].split(".zip")[0]
    url = _fix_url(entry["download_url"])
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    energy, real_d, imag_d, structure = parse_zip(r.content)

    info = meta.get(jid, {})
    if structure is None and info.get("atoms"):
        structure = atoms_to_structure(info["atoms"])
    opt_gap = _gap_float(info.get("opt_gap"))
    mbj_gap = _gap_float(info.get("mbj_gap"))
    if source == "mbj":
        gap_for_metal = mbj_gap if mbj_gap is not None else opt_gap
    else:
        gap_for_metal = opt_gap if opt_gap is not None else mbj_gap

    return {
        "jid": jid,
        "mpid": info.get("mpid") or "",
        "opt_gap": gap_for_metal,
        "mbj_gap": mbj_gap,
        "energy": energy,
        "real_d": real_d,
        "imag_d": imag_d,
        "structure": structure,
    }


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


def write_metadata(txn, total_samples, energy_array, kkr_xi_array, atom_init, gdf, radius, max_num_nbr):
    metadata = {
        "total_samples": total_samples,
        "energy_array": energy_array,
        "kkr_xi_array": kkr_xi_array,
        "radius": radius,
        "max_num_nbr": max_num_nbr,
        "n_frequencies": len(energy_array) if energy_array is not None else 0,
        "n_xi_points": len(kkr_xi_array) if kkr_xi_array is not None else 0,
        "atom_fea_len": len(next(iter(atom_init.values()))),
        "nbr_fea_len": len(gdf.filter),
        "has_kkr": kkr_xi_array is not None,
    }
    txn.put(b"__metadata__", pickle.dumps(metadata))
    return metadata


def build_dataset(
    source,
    out_dir,
    atom_init_path="atom_init.json",
    limit=None,
    workers=4,
    resume=True,
    radius=8.0,
    max_num_nbr=12,
    n_xi_points=500,
    map_size=200 * 1024**3,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / "crystal_graphs.lmdb"

    key, entries = get_download_entries(source)
    done, start_idx = load_existing_jids(lmdb_path) if resume else (set(), 0)
    if done:
        print(f"Resume: {len(done)} JIDs already in {lmdb_path} (next idx={start_idx})")

    todo = []
    for e in entries:
        jid = e["name"].split(".zip")[0]
        if jid in done:
            continue
        todo.append(e)
        if limit is not None and len(todo) >= limit:
            break
    print(f"To download: {len(todo)}")

    meta = load_dft3d_index()
    atom_init = load_atom_init(atom_init_path)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)

    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=False)
    idx = start_idx
    ok = 0
    fail = 0
    energy_array = None
    kkr_xi_array = None

    if start_idx > 0:
        with env.begin() as txn:
            prev = pickle.loads(txn.get(b"__metadata__"))
            energy_array = prev.get("energy_array")
            kkr_xi_array = prev.get("kkr_xi_array")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_raw, e, meta, source): e for e in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"build-{source}"):
            entry = futures[fut]
            try:
                raw = fut.result()
                record = make_lmdb_record(
                    raw["jid"], raw["mpid"], raw["opt_gap"], raw["mbj_gap"],
                    raw["structure"], raw["energy"], raw["real_d"], raw["imag_d"],
                    atom_init, gdf, radius, max_num_nbr, n_xi_points,
                )
                if energy_array is None:
                    energy_array = record["opt_en"]
                if kkr_xi_array is None:
                    kkr_xi_array = record["kkr_xi"]
                with env.begin(write=True) as txn:
                    txn.put(f"{idx}".encode("utf-8"), pickle.dumps(record))
                    write_metadata(
                        txn, idx + 1, energy_array, kkr_xi_array,
                        atom_init, gdf, radius, max_num_nbr,
                    )
                idx += 1
                ok += 1
            except Exception as exc:
                fail += 1
                jid = entry["name"].split(".zip")[0]
                print(f"  FAIL {jid}: {exc}")

    env.close()
    summary = {
        "source": source,
        "catalog_key": key,
        "ok": ok,
        "fail": fail,
        "lmdb_path": str(lmdb_path),
        "total_samples": idx,
    }
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Build latest JARVIS optics LMDB datasets (graphs + KKR)"
    )
    parser.add_argument("--source", choices=("opt", "mbj", "both"), default="both")
    parser.add_argument("--opt_dir", default="latest_data_opt")
    parser.add_argument("--mbj_dir", default="latest_data_mbj")
    parser.add_argument("--atom_init", default="atom_init.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max_num_nbr", type=int, default=12)
    parser.add_argument("--n_xi_points", type=int, default=500)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--list_keys", action="store_true")
    args = parser.parse_args()

    if args.list_keys:
        fls = load_raw_catalog()
        for k, v in fls.items():
            n = len(v) if isinstance(v, list) else "?"
            print(f"{k}: {n}")
        return

    common = dict(
        atom_init_path=args.atom_init,
        limit=args.limit,
        workers=args.workers,
        resume=not args.no_resume,
        radius=args.radius,
        max_num_nbr=args.max_num_nbr,
        n_xi_points=args.n_xi_points,
    )
    if args.source in ("opt", "both"):
        build_dataset("opt", args.opt_dir, **common)
    if args.source in ("mbj", "both"):
        build_dataset("mbj", args.mbj_dir, **common)


if __name__ == "__main__":
    main()
