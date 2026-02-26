"""
Simulate H2O response using mace_kkr_v2 model.
Creates H2O structure using ASE and runs inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

from preprocess_dataset import (
    load_atom_init, get_atom_features, get_neighbor_features, GaussianDistance
)
from model_loader import load_model
from lmdb_dataset import collate_pool


def create_h2o_structure():
    """
    Create H2O molecule structure using ASE.
    Uses standard H2O geometry: O-H bond length ~0.96 Å, H-O-H angle ~104.5°
    """
    # H2O molecule coordinates (in Angstroms)
    # O at origin, H atoms at standard positions
    positions = [
        [0.0, 0.0, 0.0],           # O
        [0.757, 0.587, 0.0],       # H1
        [-0.757, 0.587, 0.0],      # H2
    ]
    
    symbols = ['O', 'H', 'H']
    
    # Create ASE Atoms object
    h2o_ase = Atoms(symbols=symbols, positions=positions)
    
    # Convert to pymatgen Structure
    # For a molecule, we need to create a periodic structure with a large cell
    # to avoid interactions between periodic images
    # Use a very large cell to ensure no periodic neighbors within cutoff
    cell_size = 50.0  # Very large cell to isolate molecule
    h2o_ase.set_cell([cell_size, cell_size, cell_size])
    h2o_ase.set_pbc([True, True, True])
    h2o_ase.center()
    
    # Convert to pymatgen Structure
    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(h2o_ase)
    
    # Ensure structure is properly set up
    # For a small molecule, we want to make sure we only get real neighbors
    # not periodic image neighbors
    
    return structure


def process_structure_for_model(structure, atom_init, gdf, radius=8.0, max_num_nbr=12):
    """
    Process a structure to get features needed for the model.
    """
    atom_fea = get_atom_features(structure, atom_init)
    nbr_fea_idx, nbr_fea = get_neighbor_features(structure, radius, max_num_nbr, gdf)
    
    return atom_fea, nbr_fea, nbr_fea_idx


def simulate_h2o_response(model_path="models/mace_kkr_v2.pt", atom_init_path="atom_init.json"):
    """
    Main function to simulate H2O response using mace_kkr_v2 model.
    """
    print("=" * 70)
    print("H2O Response Simulation using mace_kkr_v2")
    print("=" * 70)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        print("Please ensure the model file exists in the models/ directory.")
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    use_log = checkpoint.get('use_log', False)
    print(f"Model trained in log-space: {use_log}")
    
    # Load model
    model, model_type = load_model(checkpoint, device)
    print(f"Model type: {model_type}")
    
    # Load atom initialization
    print(f"\nLoading atom features from: {atom_init_path}")
    atom_init = load_atom_init(atom_init_path)
    
    # Get preprocessing parameters from checkpoint or use defaults
    radius = checkpoint.get('radius', 8.0)
    max_num_nbr = checkpoint.get('max_num_nbr', 12)
    print(f"Using radius: {radius} Å, max_num_nbr: {max_num_nbr}")
    
    # Initialize Gaussian distance function
    gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)
    
    # Create H2O structure
    print("\nCreating H2O structure...")
    structure = create_h2o_structure()
    print(f"H2O structure created: {structure.formula}")
    print(f"Number of atoms: {len(structure)}")
    print(f"Cell: {structure.lattice.abc}")
    
    # Process structure
    print("\nProcessing structure for model input...")
    atom_fea, nbr_fea, nbr_fea_idx = process_structure_for_model(
        structure, atom_init, gdf, radius, max_num_nbr
    )
    
    print(f"Atom features shape: {atom_fea.shape}")
    print(f"Neighbor features shape: {nbr_fea.shape}")
    print(f"Neighbor indices shape: {nbr_fea_idx.shape}")
    
    # Get n_frequencies for dummy target
    n_frequencies = checkpoint.get('n_frequencies', checkpoint.get('n_xi_points', 500))
    n_output_channels = checkpoint.get('n_output_channels', 1)
    
    # Validate neighbor indices are within bounds
    max_idx = len(atom_fea) - 1
    if nbr_fea_idx.max() > max_idx:
        print(f"Warning: Some neighbor indices ({nbr_fea_idx.max()}) exceed max atom index ({max_idx})")
        # Clamp indices to valid range (padding indices should be 0)
        nbr_fea_idx = np.clip(nbr_fea_idx, 0, max_idx)
    
    # Prepare input for model using collate function format
    # Create a dummy target (not used for inference, but needed for collate format)
    dummy_target = np.zeros((n_output_channels, n_frequencies), dtype=np.float32)
    
    # Format as dataset item: (atom_fea, nbr_fea, nbr_fea_idx), target, idx
    dataset_item = ((atom_fea, nbr_fea, nbr_fea_idx), dummy_target, 0)
    
    # Use collate function to properly format inputs
    inputs, _, _ = collate_pool([dataset_item])
    atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx = inputs
    
    # Move to device
    atom_fea_t = atom_fea_t.to(device)
    nbr_fea_t = nbr_fea_t.to(device)
    nbr_fea_idx_t = nbr_fea_idx_t.to(device)
    crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]
    
    # Run inference
    print("\nRunning inference...")
    model.eval()
    with torch.no_grad():
        pred = model(atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx)
    
    # Convert to real-space if needed
    if use_log:
        pred_real = torch.exp(pred)
    else:
        pred_real = pred
    
    # Get frequency array (xi points for KKR)
    n_frequencies = checkpoint.get('n_frequencies', checkpoint.get('n_xi_points', 500))
    print(f"\nPrediction shape: {pred_real.shape}")
    print(f"Number of frequency points: {n_frequencies}")
    
    # Extract prediction (assuming single output channel for xx component)
    prediction = pred_real[0, 0].cpu().numpy()  # Shape: (n_frequencies,)
    
    # Get or generate xi array for plotting
    # Try to get from checkpoint, otherwise generate
    if 'kkr_xi_array' in checkpoint or 'xi_array' in checkpoint:
        xi = checkpoint.get('kkr_xi_array', checkpoint.get('xi_array'))
    else:
        # Generate xi array using logarithmic spacing (typical range)
        xi_min = 0.01
        xi_max = 100.0
        xi = np.logspace(np.log10(xi_min), np.log10(xi_max), len(prediction))
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"Predicted ε(iξ) for H2O:")
    print(f"  Shape: {prediction.shape}")
    print(f"  ξ range: {xi[0]:.4f} to {xi[-1]:.4f} eV")
    print(f"  Min value: {prediction.min():.4f}")
    print(f"  Max value: {prediction.max():.4f}")
    print(f"  Mean value: {prediction.mean():.4f}")
    print(f"  Static dielectric constant (first point): {prediction[0]:.4f}")
    
    # Create plot
    print("\nCreating plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, prediction, 'b-', linewidth=2, label='Predicted ε(iξ)')
    ax.set_xlabel('ξ (eV)', fontsize=12)
    ax.set_ylabel('ε(iξ)', fontsize=12)
    ax.set_title(f'H2O Dielectric Function ε(iξ) - {model_type} Prediction', fontsize=14)
    ax.set_xlim(0,40)
    ax.set_ylim(1,2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = 'h2o_simulation_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")    
    return prediction, xi


if __name__ == '__main__':
    # Try to find the model file
    import os
    model_path = "models/mace_kkr_v2.pt"
    
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Searching for alternative model files...")
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            if model_files:
                print(f"Found model files: {model_files}")
                if 'mace' in str(model_files).lower():
                    mace_models = [f for f in model_files if 'mace' in f.lower()]
                    if mace_models:
                        model_path = os.path.join(models_dir, mace_models[0])
                        print(f"Using: {model_path}")
                    else:
                        print("No MACE model found. Using first available model.")
                        model_path = os.path.join(models_dir, model_files[0])
                else:
                    print("No MACE model found. Using first available model.")
                    model_path = os.path.join(models_dir, model_files[0])
            else:
                print("No model files found in models/ directory.")
                print("Please ensure a trained model exists.")
                exit(1)
        else:
            print("models/ directory not found.")
            print("Please ensure a trained model exists.")
            exit(1)
    
    simulate_h2o_response(model_path=model_path)

