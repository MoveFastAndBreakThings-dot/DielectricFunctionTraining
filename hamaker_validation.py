"""
Compute Hamaker constant A (material/vac/material) from mace_kkr_v2 predicted ε(iξ).
Uses reference materials (COD CIFs) and compares to literature values.
"""

import os
import urllib.request
import numpy as np
import torch
from scipy.interpolate import interp1d

from preprocess_dataset import (
    load_atom_init,
    get_atom_features,
    get_neighbor_features,
    GaussianDistance,
)
from model_loader import load_model
from lmdb_dataset import collate_pool

# COD IDs from Crystallography Open Database (crystallography.net/cod)
# Si: diamond 298 K | MgO: rocksalt | AlN: wurtzite | Al2O3: corundum | SiO2: alpha-quartz 298 K
COD_IDS = {
    "Si": 9011998,
    "MgO": 1011116,
    "AlN": 1010514,
    "Al2O3": 1000032,
    "SiO2": 9005017,
}
COD_CIF_URL = "https://www.crystallography.net/cod/{cod_id}.cif"
_cif_cache = {}


def get_reference_cif(name):
    """Return CIF string for a reference material (fetched from COD)."""
    if name not in COD_IDS:
        raise ValueError(f"Unknown material {name}. Choose from {list(COD_IDS)}")
    if name in _cif_cache:
        return _cif_cache[name]
    url = COD_CIF_URL.format(cod_id=COD_IDS[name])
    with urllib.request.urlopen(url) as resp:
        cif_string = resp.read().decode("utf-8")
    _cif_cache[name] = cif_string
    return cif_string


def get_reference_structure(name):
    """Return pymatgen Structure for a reference material (from COD CIF).
    Species are converted to Element only (no oxidation state) so atom_init.json works.
    """
    from pymatgen.core import Structure, Element
    from pymatgen.io.cif import CifParser
    cif_string = get_reference_cif(name)
    parser = CifParser.from_str(cif_string)
    struct = parser.parse_structures(primitive=True)[0]
    # COD CIFs often have Mg2+, O2-, etc.; atom_init has Mg, O. Use symbol so Element/Specie both work.
    struct = Structure(
        struct.lattice,
        [Element(site.specie.symbol) for site in struct],
        [site.frac_coords for site in struct],
    )
    return struct


# Physical constants: 300 K, k_B in eV/K for Matsubara spacing
T_K = 300.0
K_B_J = 1.380649e-23  # J/K
K_B_EV = 8.617333e-5  # eV/K
# Matsubara frequency ξ_n (eV) = n * 2π * k_B*T (in eV)
XI_STEP_EV = 2 * np.pi * K_B_EV * T_K  # ~0.162 eV
ZJ_PER_J = 1e21  # 1 zJ = 10^-21 J
# Number of terms in the Matsubara sum: n=0 (weight 1/2) + n=1..N_MATSUBARA-1 → N_MATSUBARA terms total
N_MATSUBARA = 5000

# Reference Hamaker constants A (zJ) for material/vacuum/material from literature
REFERENCE_HAMAKER_ZJ = {
    "Si": 212.6,
    "MgO": 114.5,
    "AlN": 228.5,
    "Al2O3": 168.7,
    "SiO2": 66.6,
}


def run_model_on_structure(structure, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr):
    """Run model on a pymatgen Structure; return ε(iξ) array and xi (eV)."""
    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(structure, radius, max_num_nbr, gdf)

    n_frequencies = checkpoint.get("n_frequencies", checkpoint.get("n_xi_points", 500))
    n_output_channels = checkpoint.get("n_output_channels", 1)
    max_idx = len(atom_fea) - 1
    if nbr_fea_idx.max() > max_idx:
        nbr_fea_idx = np.clip(nbr_fea_idx, 0, max_idx)

    dummy_target = np.zeros((n_output_channels, n_frequencies), dtype=np.float32)
    dataset_item = ((atom_fea, nbr_fea, nbr_fea_idx), dummy_target, 0)
    inputs, _, _ = collate_pool([dataset_item])
    atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx = inputs

    atom_fea_t = atom_fea_t.to(device)
    nbr_fea_t = nbr_fea_t.to(device)
    nbr_fea_idx_t = nbr_fea_idx_t.to(device)
    crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

    model.eval()
    with torch.no_grad():
        pred = model(atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx)

    use_log = checkpoint.get("use_log", False)
    pred_real = torch.exp(pred) if use_log else pred
    eps_xi = pred_real[0, 0].cpu().numpy()

    if "kkr_xi_array" in checkpoint or "xi_array" in checkpoint:
        xi = np.array(checkpoint.get("kkr_xi_array", checkpoint.get("xi_array")))
    else:
        xi = np.logspace(np.log10(0.01), np.log10(100.0), len(eps_xi))

    return eps_xi, xi


def hamaker_from_eps_xi(xi_ev, eps_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=None):
    """
    Lifshitz Hamaker constant A for two identical media in vacuum.
    A = (3/2) k_B T Σ' Δ(iξ_n)²,  Δ = (ε(iξ)-1)/(ε(iξ)+1).
    ξ_n = n * 2π k_B T / ℏ in eV (Matsubara frequencies).

    Uses cubic spline interpolation (xi_ev, eps_xi) → ε at each Matsubara ξ_n.
    n_max = number of terms in the sum (n=0 with weight 1/2, plus n=1..n_max-1). Default N_MATSUBARA.
    Returns (A in zJ, n_max).
    """
    if n_max is None:
        n_max = N_MATSUBARA
    xi_ev = np.asarray(xi_ev)
    eps_xi = np.asarray(eps_xi)
    order = np.argsort(xi_ev)
    xi_ev = xi_ev[order]
    eps_xi = eps_xi[order]

    n_pts = len(xi_ev)
    # fill_value=(eps_xi[0], 1.0): below grid → static dielectric; above grid → vacuum (ε→1).
    # Cubic extrapolation outside the KKR grid is unphysical and can go negative.
    if n_pts >= 4:
        interp = interp1d(xi_ev, eps_xi, kind="cubic", bounds_error=False, fill_value=(eps_xi[0], 1.0))
    else:
        interp = interp1d(xi_ev, eps_xi, kind="linear", bounds_error=False, fill_value=(eps_xi[0], 1.0))

    eps_0 = float(interp(0.0))
    delta_0 = (eps_0 - 1.0) / (eps_0 + 1.0)

    total = 0.5 * delta_0 ** 2
    for n in range(1, n_max):  # n=1..n_max-1 so total terms = 1 + (n_max-1) = n_max
        xi_n = n * xi_step_ev
        eps_n = float(interp(xi_n))
        delta_n = (eps_n - 1.0) / (eps_n + 1.0)
        total += delta_n ** 2

    a_j = 1.5 * K_B_J * t_k * total
    return a_j * ZJ_PER_J, n_max


def get_matsubara_sequences(xi_ev, eps_xi, t_k=T_K, xi_step_ev=XI_STEP_EV, n_max=None):
    """
    Return (d_n, delta_n) arrays for n=0..n_max-1 with d_n = ε(iξ_n), Δ_n = (d_n-1)/(d_n+1).
    Used for error propagation: S(d_n) model and ∂A/∂d_n.
    """
    if n_max is None:
        n_max = N_MATSUBARA
    xi_ev = np.asarray(xi_ev)
    eps_xi = np.asarray(eps_xi)
    order = np.argsort(xi_ev)
    xi_ev = xi_ev[order]
    eps_xi = eps_xi[order]
    n_pts = len(xi_ev)
    if n_pts >= 4:
        interp = interp1d(xi_ev, eps_xi, kind="cubic", bounds_error=False, fill_value=(eps_xi[0], 1.0))
    else:
        interp = interp1d(xi_ev, eps_xi, kind="linear", bounds_error=False, fill_value=(eps_xi[0], 1.0))

    d_n = np.zeros(n_max)
    delta_n = np.zeros(n_max)
    d_n[0] = float(interp(0.0))
    delta_n[0] = (d_n[0] - 1.0) / (d_n[0] + 1.0)
    for n in range(1, n_max):
        xi_n = n * xi_step_ev
        d_n[n] = float(interp(xi_n))
        delta_n[n] = (d_n[n] - 1.0) / (d_n[n] + 1.0)
    return d_n, delta_n


def main():
    model_path = "models/mace_kkr_v2.pt"
    if not os.path.exists(model_path):
        if os.path.exists("models"):
            pts = [f for f in os.listdir("models") if f.endswith(".pt")]
            if pts:
                model_path = os.path.join("models", pts[0])
            else:
                print("No .pt model in models/")
                return
        else:
            print("models/ not found")
            return

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_type = load_model(checkpoint, device)
    atom_init = load_atom_init("atom_init.json")
    radius = checkpoint.get("radius", 8.0)
    max_num_nbr = checkpoint.get("max_num_nbr", 12)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)

    print("Hamaker constant A (material/vac/material) from mace_kkr_v2 ε(iξ)")
    print("Reference values from literature (zJ). T = 300 K. Interpolation: linear.")
    print("-" * 80)
    print(f"{'Material':<10} {'A (pred) zJ':>14} {'A (ref) zJ':>12} {'Ratio':>8} {'n_max':>8}")
    print("-" * 80)

    for name in COD_IDS:
        structure = get_reference_structure(name)
        eps_xi, xi = run_model_on_structure(
            structure, model, checkpoint, atom_init, gdf, device, radius, max_num_nbr
        )
        a_pred, n_max = hamaker_from_eps_xi(xi, eps_xi)
        a_ref = REFERENCE_HAMAKER_ZJ.get(name)
        ratio = a_pred / a_ref if a_ref else float("nan")
        print(f"{name:<10} {a_pred:>14.2f} {a_ref:>12.2f} {ratio:>8.3f} {n_max:>8d}")

    print("-" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
