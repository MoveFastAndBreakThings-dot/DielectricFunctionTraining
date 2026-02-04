"""
Plot KKR values for first 10 materials from processed LMDB.
Throwaway script for visualization.
"""

import pickle
import lmdb
import matplotlib.pyplot as plt
import numpy as np

lmdb_path = 'processed_data/crystal_graphs.lmdb'

# Open LMDB (subdir=False since it's a single file, not a directory)
env = lmdb.open(lmdb_path, readonly=True, lock=False, subdir=False)

with env.begin() as txn:
    # Get metadata (contains shared ξ array)
    metadata = pickle.loads(txn.get(b'__metadata__'))
    xi = metadata['kkr_xi_array']
    
    print(f"ξ range: {xi[0]:.4f} to {xi[-1]:.4f} eV ({len(xi)} points)")
    print(f"Total materials: {metadata['total_samples']}")
    print("\n" + "="*60)
    print("First 10 materials:")
    print("="*60)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get first 10 materials
    for i in range(min(10, metadata['total_samples'])):
        data = pickle.loads(txn.get(f'{i}'.encode()))
        
        formula = data['formula']
        mpid = data['mpid']
        kkr_xx = data['kkr_targets']['kkr_xx']
        
        print(f"  [{i}] {mpid}: {formula}")
        
        # Plot
        ax.plot(xi, kkr_xx, label=f"{mpid} ({formula})", alpha=0.8)
    
    ax.set_xlabel('ξ (eV)', fontsize=12)
    ax.set_ylabel('ε(iξ)', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('KKR Dielectric Function ε(iξ) - First 10 Materials')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kkr_first_10.png', dpi=150)
    print(f"\nPlot saved to: kkr_first_10.png")
    plt.show()

env.close()

