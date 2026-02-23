"""
Plot predictions vs actual for 20 materials from validation set.
Saves each plot to plots/ folder.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model

# ============================================================================
# CHANGE MODEL NAME HERE
# ============================================================================
MODEL_NAME = "cgcnn_kkr_v1"
# ============================================================================

os.makedirs('plots', exist_ok=True)

# Load model
model_path = f"models/{MODEL_NAME}.pt"
print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = LMDBCrystalDataset('processed_data/crystal_graphs.lmdb', use_kkr=True)

# Load model (automatically detects type)
model, model_type = load_model(checkpoint, device)
print(f"Model type: {model_type}")

xi = dataset.kkr_xi_array

# Load validation indices from checkpoint (guaranteed same as training)
if 'val_idx' in checkpoint:
    val_indices = np.array(checkpoint['val_idx'])
    print("Loaded validation indices from checkpoint ✓")
else:
    # Fallback: regenerate with same seed
    print("Warning: checkpoint missing val_idx, regenerating with seed=42")
    np.random.seed(42)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    val_indices = indices[train_size:train_size + val_size]

# Sample 20 materials from validation set
n_plot = min(20, len(val_indices))
plot_indices = val_indices[np.linspace(0, len(val_indices) - 1, n_plot, dtype=int)]

print(f"Plotting {n_plot} materials from validation set ({len(val_indices)} total)...")
print("="*60)

for i, idx in enumerate(plot_indices):
    (atom_fea, nbr_fea, nbr_fea_idx), target, _ = dataset[idx]
    info = dataset.get_sample_info(idx)
    
    # Predict
    inputs, tgt, _ = collate_pool([dataset[idx]])
    atom_fea_t, nbr_fea_t, nbr_fea_idx_t, crystal_atom_idx = inputs
    
    with torch.no_grad():
        pred = model(
            atom_fea_t.to(device), 
            nbr_fea_t.to(device), 
            nbr_fea_idx_t.to(device), 
            crystal_atom_idx
        )
    
    actual = target[0]  # kkr_xx
    predicted = pred[0, 0].cpu().numpy()
    
    # Calculate error
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, actual, 'b-', label='Actual', linewidth=2)
    ax.plot(xi, predicted, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('ξ (eV)', fontsize=12)
    ax.set_ylabel('ε(iξ)', fontsize=12)
    ax.set_title(f"{info['mpid']}: {info['formula']}\nRMSE: {rmse:.3f}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save
    safe_name = info['mpid'].replace('/', '_')
    filepath = f"plots/{i:02d}_{safe_name}_val.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[{i+1:2d}/{n_plot}] {info['mpid']:12s} {info['formula']:20s} RMSE: {rmse:.3f} → {filepath}")

dataset.close()
print("="*60)
print("Done! Plots saved to plots/ folder")
