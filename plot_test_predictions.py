"""
Plot predictions vs actual for ALL materials from test set.
Saves each plot to plots/ folder and creates a loss distribution histogram.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model
from losses import FrequencyWeightedMSELoss


# ============================================================================
# CHANGE MODEL NAME HERE
# ============================================================================
MODEL_NAME = "cgcnn_kkr_v2"
# ============================================================================

os.makedirs('plots', exist_ok=True)

# Load model
model_path = f"models/{MODEL_NAME}.pt"
print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if model was trained in log-space
use_log = checkpoint.get('use_log', False)
print(f"Log-space model: {use_log}")

dataset = LMDBCrystalDataset('processed_data/crystal_graphs.lmdb', use_kkr=True)

# Load model (automatically detects type)
model, model_type = load_model(checkpoint, device)
print(f"Model type: {model_type}")

xi = dataset.kkr_xi_array

# Create criterion (same as training)
criterion = FrequencyWeightedMSELoss(dataset.frequency_array).to(device)

# Load test indices from checkpoint (guaranteed same as training)
if 'test_idx' in checkpoint:
    test_indices = np.array(checkpoint['test_idx'])
    print("Loaded test indices from checkpoint ✓")
else:
    # Fallback: regenerate with same seed
    print("Warning: checkpoint missing test_idx, regenerating with seed=42")
    np.random.seed(42)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    test_indices = indices[train_size + val_size:]

n_test = len(test_indices)
print(f"Plotting ALL {n_test} materials from test set...")
print("=" * 60)

# Store loss values for distribution plot
loss_values = []
mpids = []

for i, idx in enumerate(test_indices):
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
    
    # Convert prediction back to real-space if model was trained in log-space
    if use_log:
        pred_real = torch.exp(pred)
    else:
        pred_real = pred
    
    actual = target[0]  # kkr_xx
    predicted = pred_real[0, 0].cpu().numpy()
    
    # Calculate loss in real-space
    loss = criterion(pred_real.to(device), tgt.to(device)).item()
    loss_values.append(loss)
    mpids.append(info['mpid'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, actual, 'b-', label='Actual', linewidth=2)
    ax.plot(xi, predicted, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('ξ (eV)', fontsize=12)
    ax.set_ylabel('ε(iξ)', fontsize=12)
    ax.set_title(f"{info['mpid']}: {info['formula']}\nLoss: {loss:.6f}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save
    safe_name = info['mpid'].replace('/', '_')
    filepath = f"plots/{i:03d}_{safe_name}_test.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[{i+1:3d}/{n_test}] {info['mpid']:12s} {info['formula']:20s} Loss: {loss:.6f} → {filepath}")

dataset.close()

# Plot loss distribution
print("=" * 60)
print("Creating loss distribution plot...")

loss_values = np.array(loss_values)

fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of loss ranges vs number of materials
counts, bins, patches = ax.hist(loss_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(np.mean(loss_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(loss_values):.6f}')
ax.axvline(np.median(loss_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(loss_values):.6f}')
ax.set_xlabel('Loss (Freq-Weighted MSE)', fontsize=12)
ax.set_ylabel('Number of Materials', fontsize=12)
ax.set_title('Test Set Loss Distribution', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/loss_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Print summary statistics
print("=" * 60)
print("Test Set Summary:")
print(f"  Total materials: {n_test}")
print(f"  Mean Loss:       {np.mean(loss_values):.6f}")
print(f"  Median Loss:     {np.median(loss_values):.6f}")
print(f"  Std Loss:        {np.std(loss_values):.6f}")
print(f"  Min Loss:        {np.min(loss_values):.6f} ({mpids[np.argmin(loss_values)]})")
print(f"  Max Loss:        {np.max(loss_values):.6f} ({mpids[np.argmax(loss_values)]})")
print("=" * 60)
print("Done! All plots saved to plots/ folder")
print("Loss distribution saved to: plots/loss_distribution.png")

