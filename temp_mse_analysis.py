"""
Temporary script to analyze MSE and interpret it in context.
Loads a model, makes predictions, computes MSE, and provides statistics.
"""

import torch
import numpy as np
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from model_loader import load_model
from losses import FrequencyWeightedMSELoss


MODEL_NAME = "cgcnn_kkr_v2"

# Load model
model_path = f"models/{MODEL_NAME}.pt"
print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

use_log = checkpoint.get('use_log', False)
print(f"Log-space model: {use_log}")

dataset = LMDBCrystalDataset('processed_data/crystal_graphs.lmdb', use_kkr=True)

# Load model (automatically detects type)
model, model_type = load_model(checkpoint, device)
print(f"Model type: {model_type}")

criterion = FrequencyWeightedMSELoss(dataset.frequency_array).to(device)
xi = dataset.kkr_xi_array

# Load test indices
if 'test_idx' in checkpoint:
    test_indices = np.array(checkpoint['test_idx'])
else:
    print("Warning: checkpoint missing test_idx, using first 100 samples")
    test_indices = np.arange(100)

print(f"Analyzing {len(test_indices)} test samples...")
print("=" * 70)

# Collect predictions and compute errors
all_mse_per_sample = []
all_actual_values = []
all_predicted_values = []
all_relative_errors = []

# Specifically for first frequency point (lowest ξ, static dielectric constant)
first_point_actual = []
first_point_predicted = []
first_point_errors = []
first_point_relative_errors = []

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
    
    # Convert to real-space if needed
    if use_log:
        pred_real = torch.exp(pred)
    else:
        pred_real = pred
    
    # Compute MSE for this sample
    mse = criterion(pred_real.to(device), tgt.to(device)).item()
    all_mse_per_sample.append(mse)
    
    # Get actual and predicted values (flatten across frequencies)
    actual_flat = tgt[0, 0].numpy().flatten()
    predicted_flat = pred_real[0, 0].cpu().numpy().flatten()
    
    all_actual_values.extend(actual_flat)
    all_predicted_values.extend(predicted_flat)
    
    # Compute relative errors (avoid division by zero)
    mask = actual_flat > 1e-6
    if mask.sum() > 0:
        relative_errors = np.abs((predicted_flat[mask] - actual_flat[mask]) / actual_flat[mask])
        all_relative_errors.extend(relative_errors)
    
    # Collect first frequency point (index 0) - static dielectric constant
    first_actual = actual_flat[0]
    first_predicted = predicted_flat[0]
    first_point_actual.append(first_actual)
    first_point_predicted.append(first_predicted)
    first_point_errors.append((first_predicted - first_actual) ** 2)
    if first_actual > 1e-6:
        first_point_relative_errors.append(np.abs((first_predicted - first_actual) / first_actual))

dataset.close()

# Convert to numpy arrays
all_mse_per_sample = np.array(all_mse_per_sample)
all_actual_values = np.array(all_actual_values)
all_predicted_values = np.array(all_predicted_values)
all_relative_errors = np.array(all_relative_errors)

# Compute overall MSE (mean of per-sample MSEs)
overall_mse = np.mean(all_mse_per_sample)

# Compute per-point MSE (across all frequency points)
per_point_mse = np.mean((all_predicted_values - all_actual_values) ** 2)

# Statistics for interpretation
print("\n" + "=" * 70)
print("MSE STATISTICS")
print("=" * 70)
print(f"Overall MSE (mean of per-sample MSEs): {overall_mse:.6f}")
print(f"Per-point MSE (across all frequency points): {per_point_mse:.6f}")
print(f"RMSE (Root Mean Squared Error): {np.sqrt(overall_mse):.6f}")
print(f"Per-point RMSE: {np.sqrt(per_point_mse):.6f}")

print("\n" + "=" * 70)
print("TARGET VALUE STATISTICS (to interpret MSE scale)")
print("=" * 70)
print(f"Mean of actual values: {np.mean(all_actual_values):.4f}")
print(f"Median of actual values: {np.median(all_actual_values):.4f}")
print(f"Std of actual values: {np.std(all_actual_values):.4f}")
print(f"Min actual value: {np.min(all_actual_values):.4f}")
print(f"Max actual value: {np.max(all_actual_values):.4f}")
print(f"25th percentile: {np.percentile(all_actual_values, 25):.4f}")
print(f"75th percentile: {np.percentile(all_actual_values, 75):.4f}")

print("\n" + "=" * 70)
print("RELATIVE ERROR STATISTICS")
print("=" * 70)
print(f"Mean relative error: {np.mean(all_relative_errors) * 100:.2f}%")
print(f"Median relative error: {np.median(all_relative_errors) * 100:.2f}%")
print(f"RMSE relative to mean: {(np.sqrt(overall_mse) / np.mean(all_actual_values)) * 100:.2f}%")
print(f"RMSE relative to median: {(np.sqrt(overall_mse) / np.median(all_actual_values)) * 100:.2f}%")

print("\n" + "=" * 70)
print("PER-SAMPLE MSE STATISTICS")
print("=" * 70)
print(f"Mean MSE per sample: {np.mean(all_mse_per_sample):.6f}")
print(f"Median MSE per sample: {np.median(all_mse_per_sample):.6f}")
print(f"Std MSE per sample: {np.std(all_mse_per_sample):.6f}")
print(f"Min MSE per sample: {np.min(all_mse_per_sample):.6f}")
print(f"Max MSE per sample: {np.max(all_mse_per_sample):.6f}")
print(f"25th percentile: {np.percentile(all_mse_per_sample, 25):.6f}")
print(f"75th percentile: {np.percentile(all_mse_per_sample, 75):.6f}")
print(f"95th percentile: {np.percentile(all_mse_per_sample, 95):.6f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
mean_actual = np.mean(all_actual_values)
median_actual = np.median(all_actual_values)
rmse = np.sqrt(overall_mse)

print(f"RMSE = {rmse:.4f}")
print(f"Mean actual value = {mean_actual:.4f}")
print(f"Median actual value = {median_actual:.4f}")
print(f"\nRMSE is {rmse/mean_actual*100:.1f}% of mean actual value")
print(f"RMSE is {rmse/median_actual*100:.1f}% of median actual value")
print(f"\nTypical prediction error: ±{rmse:.4f} (absolute)")
print(f"Typical relative error: ±{np.median(all_relative_errors)*100:.1f}% (median)")

# Convert first point data to numpy arrays
first_point_actual = np.array(first_point_actual)
first_point_predicted = np.array(first_point_predicted)
first_point_errors = np.array(first_point_errors)
first_point_relative_errors = np.array(first_point_relative_errors)

# Compute statistics for first point
first_point_mse = np.mean(first_point_errors)
first_point_rmse = np.sqrt(first_point_mse)

print("\n" + "=" * 70)
print("FIRST FREQUENCY POINT STATISTICS (ξ_min, Static Dielectric Constant)")
print("=" * 70)
print(f"ξ value at first point: {xi[0]:.6f} eV")
print(f"\nMSE at first point: {first_point_mse:.6f}")
print(f"RMSE at first point: {first_point_rmse:.6f}")
print(f"Mean absolute error: {np.mean(np.abs(first_point_predicted - first_point_actual)):.6f}")

print(f"\nActual values at first point:")
print(f"  Mean: {np.mean(first_point_actual):.4f}")
print(f"  Median: {np.median(first_point_actual):.4f}")
print(f"  Std: {np.std(first_point_actual):.4f}")
print(f"  Min: {np.min(first_point_actual):.4f}")
print(f"  Max: {np.max(first_point_actual):.4f}")
print(f"  25th percentile: {np.percentile(first_point_actual, 25):.4f}")
print(f"  75th percentile: {np.percentile(first_point_actual, 75):.4f}")

print(f"\nPredicted values at first point:")
print(f"  Mean: {np.mean(first_point_predicted):.4f}")
print(f"  Median: {np.median(first_point_predicted):.4f}")
print(f"  Std: {np.std(first_point_predicted):.4f}")

print(f"\nRelative errors at first point:")
print(f"  Mean relative error: {np.mean(first_point_relative_errors) * 100:.2f}%")
print(f"  Median relative error: {np.median(first_point_relative_errors) * 100:.2f}%")
print(f"  Std relative error: {np.std(first_point_relative_errors) * 100:.2f}%")

mean_first_actual = np.mean(first_point_actual)
median_first_actual = np.median(first_point_actual)

print(f"\nInterpretation for first point:")
print(f"  RMSE = {first_point_rmse:.4f}")
print(f"  Mean actual value = {mean_first_actual:.4f}")
print(f"  Median actual value = {median_first_actual:.4f}")
print(f"  RMSE is {first_point_rmse/mean_first_actual*100:.1f}% of mean actual value")
print(f"  RMSE is {first_point_rmse/median_first_actual*100:.1f}% of median actual value")
print(f"  Typical relative error: ±{np.median(first_point_relative_errors)*100:.1f}% (median)")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)

