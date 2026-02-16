"""
Detailed analysis of test set predictions.
Identifies outliers and provides breakdown by loss ranges.
Handles both log-space and direct prediction models.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from cgcnn_model import FrequencyCGCNN
from losses import FrequencyWeightedMSELoss


MODEL_NAME = "cgcnn_kkr_v2"

# Load model and dataset
model_path = f"models/{MODEL_NAME}.pt"
print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check if model was trained in log-space
use_log = checkpoint.get('use_log', False)
print(f"Log-space model: {use_log}")

dataset = LMDBCrystalDataset('processed_data/crystal_graphs.lmdb', use_kkr=True)

model = FrequencyCGCNN(
    orig_atom_fea_len=checkpoint['orig_atom_fea_len'],
    nbr_fea_len=checkpoint['nbr_fea_len'],
    n_frequencies=checkpoint['n_frequencies'],
    n_output_channels=checkpoint['n_output_channels']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

criterion = FrequencyWeightedMSELoss(dataset.frequency_array).to(device)
xi = dataset.kkr_xi_array

# Load test indices
if 'test_idx' in checkpoint:
    test_indices = np.array(checkpoint['test_idx'])
else:
    print("Warning: checkpoint missing test_idx, regenerating with seed=42")
    np.random.seed(42)
    n_samples = len(dataset)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)
    test_indices = indices[train_size + val_size:]

print(f"Analyzing {len(test_indices)} test materials...")
print("=" * 70)

# Collect all results
results = []
for idx in test_indices:
    info = dataset.get_sample_info(idx)
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
    
    # Loss computed in real-space for fair comparison
    loss = criterion(pred_real.to(device), tgt.to(device)).item()
    actual = tgt[0, 0].numpy()  # kkr_xx
    predicted = pred_real[0, 0].cpu().numpy()
    results.append({
        'idx': idx,
        'mpid': info['mpid'],
        'formula': info['formula'],
        'loss': loss,
        'actual': actual,
        'predicted': predicted
    })

dataset.close()

# Sort by loss
results = sorted(results, key=lambda x: x['loss'])
losses = np.array([r['loss'] for r in results])

# Percentile analysis
print("\n" + "=" * 70)
print("PERCENTILE BREAKDOWN")
print("=" * 70)
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    val = np.percentile(losses, p)
    count = np.sum(losses <= val)
    print(f"  {p:3d}th percentile: Loss ≤ {val:10.2f}  ({count:4d} materials)")

# Loss range breakdown
print("\n" + "=" * 70)
print("LOSS RANGE BREAKDOWN")
print("=" * 70)
ranges = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 5000), (5000, float('inf'))]
for low, high in ranges:
    count = np.sum((losses >= low) & (losses < high))
    pct = 100 * count / len(losses)
    if high == float('inf'):
        print(f"  Loss ≥ {low:5.0f}:          {count:4d} materials ({pct:5.1f}%)")
    else:
        print(f"  Loss {low:5.0f} - {high:5.0f}:    {count:4d} materials ({pct:5.1f}%)")

# Best predictions
print("\n" + "=" * 70)
print("TOP 10 BEST PREDICTIONS (lowest loss)")
print("=" * 70)
for i, r in enumerate(results[:10]):
    print(f"  {i+1:2d}. {r['mpid']:12s} {r['formula']:25s} Loss: {r['loss']:.4f}")

# Worst predictions
print("\n" + "=" * 70)
print("TOP 20 WORST PREDICTIONS (highest loss)")
print("=" * 70)
for i, r in enumerate(results[-20:][::-1]):
    print(f"  {i+1:2d}. {r['mpid']:12s} {r['formula']:25s} Loss: {r['loss']:.2f}")

# Create detailed plots
os.makedirs('plots', exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Log-scale histogram
ax1 = axes[0, 0]
ax1.hist(np.log10(losses + 1), bins=40, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(np.log10(np.median(losses) + 1), color='orange', linestyle='--', linewidth=2, 
            label=f'Median: {np.median(losses):.1f}')
ax1.set_xlabel('log₁₀(Loss + 1)', fontsize=12)
ax1.set_ylabel('Number of Materials', fontsize=12)
ax1.set_title('Loss Distribution (Log Scale)', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Cumulative distribution
ax2 = axes[0, 1]
sorted_losses = np.sort(losses)
cumulative = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses) * 100
ax2.plot(sorted_losses, cumulative, linewidth=2, color='steelblue')
ax2.axhline(50, color='orange', linestyle='--', alpha=0.7, label='50%')
ax2.axhline(90, color='red', linestyle='--', alpha=0.7, label='90%')
ax2.axhline(95, color='darkred', linestyle='--', alpha=0.7, label='95%')
ax2.set_xlabel('Loss', fontsize=12)
ax2.set_ylabel('Cumulative % of Materials', fontsize=12)
ax2.set_title('Cumulative Distribution', fontsize=14)
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Loss vs rank (sorted)
ax3 = axes[1, 0]
ax3.semilogy(range(len(losses)), sorted_losses, linewidth=1.5, color='steelblue')
ax3.axhline(np.median(losses), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.1f}')
ax3.set_xlabel('Material Rank (sorted by loss)', fontsize=12)
ax3.set_ylabel('Loss (log scale)', fontsize=12)
ax3.set_title('Loss vs Rank', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Zoomed histogram (excluding outliers)
ax4 = axes[1, 1]
p95 = np.percentile(losses, 95)
filtered = losses[losses <= p95]
ax4.hist(filtered, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax4.axvline(np.median(losses), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(losses):.1f}')
ax4.set_xlabel('Loss', fontsize=12)
ax4.set_ylabel('Number of Materials', fontsize=12)
ax4.set_title(f'Loss Distribution (95th percentile, ≤{p95:.0f})', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('plots/detailed_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("Saved: plots/detailed_analysis.png")

# Plot 20 best and 20 worst materials
os.makedirs('plot_test', exist_ok=True)

print("\nGenerating plots for 20 best materials...")
for i, r in enumerate(results[:20]):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, r['actual'], 'b-', label='Actual', linewidth=2)
    ax.plot(xi, r['predicted'], 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('ξ (eV)', fontsize=12)
    ax.set_ylabel('ε(iξ)', fontsize=12)
    ax.set_title(f"{r['mpid']}: {r['formula']}\nLoss: {r['loss']:.4f}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    safe_name = r['mpid'].replace('/', '_')
    filepath = f"plot_test/best_{i+1:02d}_{safe_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{i+1:2d}/20] {r['mpid']:12s} Loss: {r['loss']:.4f} → {filepath}")

print("\nGenerating plots for 20 worst materials...")
worst_20 = results[-20:][::-1]
for i, r in enumerate(worst_20):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xi, r['actual'], 'b-', label='Actual', linewidth=2)
    ax.plot(xi, r['predicted'], 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('ξ (eV)', fontsize=12)
    ax.set_ylabel('ε(iξ)', fontsize=12)
    ax.set_title(f"{r['mpid']}: {r['formula']}\nLoss: {r['loss']:.2f}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    safe_name = r['mpid'].replace('/', '_')
    filepath = f"plot_test/worst_{i+1:02d}_{safe_name}.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [{i+1:2d}/20] {r['mpid']:12s} Loss: {r['loss']:.2f} → {filepath}")

print("\n" + "=" * 70)
print("Done! Plots saved to plot_test/ folder")
print("=" * 70)
