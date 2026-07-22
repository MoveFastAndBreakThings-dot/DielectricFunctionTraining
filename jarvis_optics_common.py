"""
Shared JARVIS optics helpers: Figshare download, zip parse, LMDB record build.
"""

import io
import os
import pickle
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests

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
FIGSHARE_PROJECT = {"opt": 86378, "mbj": 84020}
UA = {"User-Agent": "TianTianResearch-optics-builder/1.0"}


def _file_id_from_url(url):
    parts = url.rstrip("/").split("/")
    for i, p in enumerate(parts):
        if p == "files" and i + 1 < len(parts):
            return parts[i + 1].split("?")[0]
    raise ValueError(f"Cannot parse Figshare file id from {url!r}")


def figshare_download_url(url):
    """api.figshare.com bypasses ndownloader WAF empty-202 responses."""
    return f"https://api.figshare.com/v2/file/download/{_file_id_from_url(url)}"


def download_bytes(url, timeout=120):
    api_url = figshare_download_url(url)
    r = requests.get(api_url, timeout=timeout, headers=UA, allow_redirects=True)
    r.raise_for_status()
    data = r.content
    if len(data) < 4 or data[:2] != b"PK":
        raise RuntimeError(
            f"Not a zip (status={r.status_code}, len={len(data)}, "
            f"ctype={r.headers.get('content-type')}, url={api_url})"
        )
    return data


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
        arts = requests.get(url, timeout=60, headers=UA).json()
        if not arts:
            break
        for art in arts:
            detail = requests.get(art["url_public_api"], timeout=60, headers=UA).json()
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
        }
    print(f"  indexed {len(index)} dft_3d entries")
    return index


def gap_float(x):
    if x in (None, "na"):
        return None
    return float(x)


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


def parse_zip_bytes(zip_bytes):
    """Parse a material zip (bytes) → energy, real_d, imag_d, structure."""
    from jarvis.io.vasp.inputs import Poscar
    from jarvis.io.vasp.outputs import Vasprun

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
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


def parse_zip_path(zip_path):
    """Parse a on-disk JVASP-*.zip → energy, real_d, imag_d, structure."""
    with open(zip_path, "rb") as f:
        return parse_zip_bytes(f.read())


def is_material_zip_name(name):
    """True for JVASP-12345.zip, false for article-range archives."""
    base = Path(name).name
    return (
        base.startswith("JVASP-")
        and base.endswith(".zip")
        and "_to_" not in base.lower()
        and " to " not in base.lower()
        and base.count("-") == 1
    )


def discover_material_jobs(raw_dir):
    """
    Find per-material optics sources under raw_dir.

    Supports:
      - flat JVASP-XXXXX.zip on disk
      - Figshare article zips that contain nested JVASP-XXXXX.zip
      - article zips that contain JVASP-XXXXX/vasprun.xml folders
    """
    raw_dir = Path(raw_dir)
    jobs = {}  # jid -> job tuple

    # Flat per-material zips
    for p in raw_dir.rglob("*.zip"):
        if not p.is_file() or not is_material_zip_name(p.name):
            continue
        jid = p.stem
        jobs[jid] = ("file", jid, str(p))

    # Article-level zips (everything else)
    for p in sorted(raw_dir.rglob("*.zip")):
        if not p.is_file() or is_material_zip_name(p.name):
            continue
        try:
            with zipfile.ZipFile(p, "r") as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            print(f"  skip bad zip: {p}")
            continue

        nested = [n for n in names if is_material_zip_name(n)]
        if nested:
            for n in nested:
                jid = Path(n).stem
                if jid not in jobs:
                    jobs[jid] = ("nested_zip", jid, str(p), n)
            continue

        by_jid = {}
        for n in names:
            for part in Path(n).parts:
                if part.startswith("JVASP-") and "_to_" not in part and " to " not in part:
                    jid = part
                    if is_material_zip_name(jid + ".zip") or (
                        jid.startswith("JVASP-") and jid.count("-") == 1
                    ):
                        by_jid.setdefault(jid, []).append(n)
                    break
        for jid, members in by_jid.items():
            if jid in jobs:
                continue
            if not any(m.endswith("vasprun.xml") for m in members):
                continue
            jobs[jid] = ("nested_dir", jid, str(p), members)

    return [jobs[k] for k in sorted(jobs)]


def load_job_zip_bytes(job):
    """Return zip bytes for a discover_material_jobs entry."""
    kind = job[0]
    if kind == "file":
        with open(job[2], "rb") as f:
            return f.read()
    if kind == "nested_zip":
        _, _, article, member = job
        with zipfile.ZipFile(article, "r") as z:
            return z.read(member)
    if kind == "nested_dir":
        _, jid, article, members = job
        buf = io.BytesIO()
        with zipfile.ZipFile(article, "r") as zin:
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zout:
                for m in members:
                    if m.endswith("/"):
                        continue
                    zout.writestr(Path(m).name, zin.read(m))
        return buf.getvalue()
    raise ValueError(f"Unknown job kind: {kind}")


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


def write_lmdb_metadata(txn, total_samples, energy_array, kkr_xi_array, atom_init, gdf, radius, max_num_nbr):
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
