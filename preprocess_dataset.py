"""
Dataset Preprocessing Script for CGCNN
Converts JSON files with embedded CIF strings into pre-computed crystal graphs
stored in LMDB format for fast training.

Includes KKR (Kramers-Kronig Relation) transformation of dielectric data.

Usage:
    python preprocess_dataset.py
    python preprocess_dataset.py --input_dir ./data --output_dir ./processed_data
"""

import json
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import lmdb
from scipy.constants import pi
from pymatgen.io.cif import CifParser
from pymatgen.core import Element


class GaussianDistance:
    """Expands distances by Gaussian basis functions."""
    
    def __init__(self, dmin=0, dmax=8, step=0.2, var=None):
        self.filter = np.arange(dmin, dmax + step, step)
        self.var = var if var is not None else step

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)


def load_atom_init(atom_init_path):
    """Load atom initialization features and convert Z->symbol."""
    with open(atom_init_path, 'r') as f:
        atom_init_data = json.load(f)
    return {Element.from_Z(int(k)).symbol: np.array(v, dtype=np.float32) 
            for k, v in atom_init_data.items()}


def get_atom_features(structure, atom_init):
    """Extract atom features from structure."""
    atom_fea = []
    for site in structure:
        element = str(site.specie)
        if element not in atom_init:
            raise ValueError(f"Element {element} not in atom_init.json")
        atom_fea.append(atom_init[element])
    return np.array(atom_fea, dtype=np.float32)


def get_neighbor_features(structure, radius, max_num_nbr, gdf):
    """Extract neighbor features and indices from structure."""
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            nbr_fea_idx.append([i[2] for i in nbr] + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append([i[1] for i in nbr] + [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append([i[2] for i in nbr[:max_num_nbr]])
            nbr_fea.append([i[1] for i in nbr[:max_num_nbr]])
    
    nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int64)
    nbr_fea = np.array(nbr_fea, dtype=np.float32)
    nbr_fea = gdf.expand(nbr_fea).astype(np.float32)
    
    return nbr_fea_idx, nbr_fea


# ============================================================================
# KKR (Kramers-Kronig Relation) - Fully Vectorized trapezoidal integration
# ============================================================================

def apply_kkr(omega, epsilon_imag, xi_points=None, n_xi_points=500, cshift=1e-6):
    """
    Fast vectorized KKR: ε₂(ω) → ε(iξ)
    
    ε(iξ) = 1 + (2/π) ∫₀^∞ [ω × ε₂(ω)] / (ω² + ξ²) dω
    
    Uses broadcasting + trapezoidal integration for speed.
    
    Parameters
    ----------
    omega : array (N,)
        Frequency/energy values (eV)
    epsilon_imag : array (N,)
        Imaginary part of dielectric function ε₂(ω)
    xi_points : array, optional
        Specific ξ points to evaluate. If None, uses logarithmic spacing.
    n_xi_points : int
        Number of ξ points if xi_points is None
    cshift : float
        Small shift for numerical stability
    
    Returns
    -------
    xi : array (M,)
        Imaginary frequency points
    epsilon_kkr : array (M,)
        ε(iξ) values at each ξ point
    """
    omega = np.array(omega, dtype=np.float64)
    epsilon_imag = np.array(epsilon_imag, dtype=np.float64)
    
    # Define ξ points (imaginary frequency)
    if xi_points is None:
        xi_min = max(omega[0], 0.01)
        xi_max = omega[-1]
        xi = np.logspace(np.log10(xi_min), np.log10(xi_max), n_xi_points)
    else:
        xi = np.array(xi_points, dtype=np.float64)
    
    # Fully vectorized: broadcast omega[N] and xi[M] to [M, N]
    omega_2d = omega[np.newaxis, :]      # [1, N]
    eps_2d = epsilon_imag[np.newaxis, :] # [1, N]
    xi_2d = xi[:, np.newaxis]            # [M, 1]
    
    # factor shape: [M, N]
    cshift_c = complex(0, cshift)
    factor = omega_2d * eps_2d / (omega_2d**2 + xi_2d**2 + cshift_c)
    
    # Integrate along omega axis (axis=1) for each xi
    int_res = np.real(np.trapezoid(factor, omega, axis=1))
    epsilon_kkr = 1.0 + 2.0 / pi * int_res
    
    return xi.astype(np.float32), epsilon_kkr.astype(np.float32)


def compute_kkr_for_sample(sample, n_xi_points=500):
    """
    Compute KKR transformation for all dielectric tensor components.
    
    Returns dict with:
        - kkr_xi: imaginary frequency points
        - kkr_{component}: KKR-transformed values for each component
    """
    omega = sample.get('opt_en', [])
    if not omega:
        return None
    
    omega = np.array(omega)
    
    # Compute KKR for each component
    kkr_data = {}
    xi = None
    
    for component in ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']:
        imag_key = f'opt_imag{component}'
        if imag_key in sample:
            epsilon_imag = np.array(sample[imag_key])
            xi, kkr_values = apply_kkr(omega, epsilon_imag, n_xi_points=n_xi_points)
            kkr_data[f'kkr_{component}'] = kkr_values
    
    if xi is not None:
        kkr_data['kkr_xi'] = xi
    
    return kkr_data


# ============================================================================
# Main Processing
# ============================================================================

def process_single_material(sample, atom_init, gdf, radius, max_num_nbr, compute_kkr=True, n_xi_points=500):
    """Process a single material entry from JSON."""
    cif_string = sample.get('cif', '')
    if not cif_string:
        return None, f"Missing CIF for {sample.get('mpid', 'unknown')}"
    
    parser = CifParser.from_str(cif_string)
    structure = parser.parse_structures(primitive=True)[0]
    
    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(structure, radius, max_num_nbr, gdf)
    
    # Extract original optical targets - all 6 dielectric tensor components
    targets = {}
    for component in ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']:
        real_key = f'opt_real{component}'
        imag_key = f'opt_imag{component}'
        if real_key in sample and imag_key in sample:
            targets[f'real_{component}'] = np.array(sample[real_key], dtype=np.float32)
            targets[f'imag_{component}'] = np.array(sample[imag_key], dtype=np.float32)
    
    # Get energy array (frequency/energy points)
    opt_en = np.array(sample.get('opt_en', []), dtype=np.float32) if 'opt_en' in sample else None
    
    result = {
        'mpid': sample.get('mpid', ''),
        'jid': sample.get('jid', ''),
        'atom_fea': atom_fea,
        'nbr_fea': nbr_fea,
        'nbr_fea_idx': nbr_fea_idx,
        'positions': np.array(structure.cart_coords, dtype=np.float32),
        'cell': np.array(structure.lattice.matrix, dtype=np.float32),
        'atomic_numbers': np.array([site.specie.Z for site in structure], dtype=np.int64),
        'pbc': np.array(structure.pbc, dtype=bool),
        'targets': targets,
        'opt_en': opt_en,
        'opt_gap': sample.get('opt_gap', None),
        'mbj_gap': sample.get('mbj_gap', None),
        'n_atoms': len(atom_fea),
        'formula': structure.formula,
    }
    
    # Compute KKR transformation
    if compute_kkr and opt_en is not None:
        kkr_data = compute_kkr_for_sample(sample, n_xi_points=n_xi_points)
        if kkr_data:
            result['kkr_targets'] = {k: v for k, v in kkr_data.items() if k != 'kkr_xi'}
            result['kkr_xi'] = kkr_data.get('kkr_xi')
    
    return result, None


def process_json_file(json_path, atom_init, gdf, radius, max_num_nbr, compute_kkr=True, n_xi_points=500):
    """Process a single JSON file and yield processed samples."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for sample in data:
        result, error = process_single_material(
            sample, atom_init, gdf, radius, max_num_nbr, 
            compute_kkr=compute_kkr, n_xi_points=n_xi_points
        )
        if error:
            yield None, error
            continue
        yield result, None


def create_lmdb_dataset(input_dir, output_dir, atom_init_path, 
                        radius=8.0, max_num_nbr=12, compute_kkr=True,
                        n_xi_points=500, map_size=100*1024**3):
    """
    Create LMDB dataset from JSON files.
    
    Parameters
    ----------
    input_dir : str
        Directory containing JSON files
    output_dir : str
        Output directory for LMDB database
    atom_init_path : str
        Path to atom_init.json
    radius : float
        Cutoff radius for neighbor search
    max_num_nbr : int
        Maximum number of neighbors
    compute_kkr : bool
        Whether to compute KKR-transformed dielectric data
    n_xi_points : int
        Number of imaginary frequency points for KKR
    map_size : int
        LMDB map size in bytes (default 100GB)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    atom_init = load_atom_init(atom_init_path)
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)
    
    # Find all JSON files
    json_files = sorted(input_dir.glob('*.json'))
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    if len(json_files) == 0:
        print(f"Error: No JSON files found in {input_dir}")
        return None
    
    # Create LMDB environment
    lmdb_path = output_dir / 'crystal_graphs.lmdb'
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=False)
    
    # Process all files
    total_samples = 0
    failed_samples = 0
    energy_array = None
    kkr_xi_array = None
    
    with env.begin(write=True) as txn:
        for json_file in json_files:
            print(f"\nProcessing {json_file.name}...")
            
            pbar = tqdm(process_json_file(
                json_file, atom_init, gdf, radius, max_num_nbr,
                compute_kkr=compute_kkr, n_xi_points=n_xi_points
            ), desc=f"  {json_file.name}")
            
            for sample_data, error in pbar:
                if error:
                    failed_samples += 1
                    continue
                
                # Store energy arrays once (same for all samples)
                if energy_array is None and sample_data['opt_en'] is not None:
                    energy_array = sample_data['opt_en']
                if kkr_xi_array is None and sample_data.get('kkr_xi') is not None:
                    kkr_xi_array = sample_data['kkr_xi']
                
                key = f'{total_samples}'.encode('utf-8')
                value = pickle.dumps(sample_data)
                txn.put(key, value)
                total_samples += 1
        
        # Store metadata
        metadata = {
            'total_samples': total_samples,
            'energy_array': energy_array,
            'kkr_xi_array': kkr_xi_array,
            'radius': radius,
            'max_num_nbr': max_num_nbr,
            'n_frequencies': len(energy_array) if energy_array is not None else 0,
            'n_xi_points': len(kkr_xi_array) if kkr_xi_array is not None else 0,
            'atom_fea_len': len(next(iter(atom_init.values()))),
            'nbr_fea_len': len(gdf.filter),
            'has_kkr': compute_kkr and kkr_xi_array is not None,
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))
    
    print(f"\n{'='*60}")
    print(f"Done! Processed {total_samples} materials ({failed_samples} failed)")
    print(f"LMDB database saved to: {lmdb_path}")
    print(f"\nMetadata:")
    print(f"  - Original frequency points: {metadata['n_frequencies']}")
    print(f"  - KKR ξ points: {metadata['n_xi_points']}")
    print(f"  - Atom features: {metadata['atom_fea_len']}")
    print(f"  - Neighbor features: {metadata['nbr_fea_len']}")
    print(f"  - Has KKR data: {metadata['has_kkr']}")
    
    return lmdb_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess JSON data for CGCNN')
    parser.add_argument('--input_dir', type=str, default='data',
                        help='Directory containing JSON files (default: data)')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Output directory (default: processed_data)')
    parser.add_argument('--atom_init', type=str, default='atom_init.json',
                        help='Path to atom_init.json')
    parser.add_argument('--radius', type=float, default=8.0,
                        help='Cutoff radius for neighbors')
    parser.add_argument('--max_num_nbr', type=int, default=12,
                        help='Maximum number of neighbors')
    parser.add_argument('--no_kkr', action='store_true',
                        help='Skip KKR computation (faster preprocessing)')
    parser.add_argument('--n_xi_points', type=int, default=500,
                        help='Number of imaginary frequency points for KKR (default: 500)')
    
    args = parser.parse_args()
    
    create_lmdb_dataset(
        args.input_dir, 
        args.output_dir, 
        args.atom_init,
        args.radius, 
        args.max_num_nbr,
        compute_kkr=not args.no_kkr,
        n_xi_points=args.n_xi_points,
    )
