import json
import os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt

def check_isotropy(epsilon_xx, epsilon_yy, epsilon_zz, tolerance=0.1):
    """
    Check if epsilon xx, yy, and zz are same or similar.
    If not, classify as non-isotropic.
    
    Args:
        epsilon_xx: Array of epsilon xx values
        epsilon_yy: Array of epsilon yy values
        epsilon_zz: Array of epsilon zz values
        tolerance: Relative tolerance for similarity check (default 0.1 = 10%)
    
    Returns:
        str: 'isotropic' or 'non-isotropic'
    """
    epsilon_xx = np.array(epsilon_xx)
    epsilon_yy = np.array(epsilon_yy)
    epsilon_zz = np.array(epsilon_zz)
    
    mean_xx = np.mean(np.abs(epsilon_xx))
    mean_yy = np.mean(np.abs(epsilon_yy))
    mean_zz = np.mean(np.abs(epsilon_zz))
    
    avg_mean = (mean_xx + mean_yy + mean_zz) / 3.0
    
    if avg_mean == 0:
        return 'isotropic'
    
    rel_diff_xy = abs(mean_xx - mean_yy) / avg_mean
    rel_diff_xz = abs(mean_xx - mean_zz) / avg_mean
    rel_diff_yz = abs(mean_yy - mean_zz) / avg_mean
    
    max_rel_diff = max(rel_diff_xy, rel_diff_xz, rel_diff_yz)
    
    if max_rel_diff <= tolerance:
        return 'isotropic'
    else:
        return 'non-isotropic'

def kkr_integrand(omega_prime, xi, epsilon2_interp):
    """
    Integrand for Kramers-Kronig relation.
    ε(iξ) = 1 + (2/π) ∫₀^∞ [ω Im[ε(ω)]] / (ω² + ξ²) dω
    """
    epsilon2_val = epsilon2_interp(omega_prime)
    numerator = omega_prime * epsilon2_val
    denominator = omega_prime**2 + xi**2
    return numerator / denominator

def apply_kkr_relation(omega, epsilon2, omega_range=None, max_points=1000):
    """
    Apply Kramers-Kronig relation using cubic spline interpolation and quad integration.
    
    Args:
        omega: Array of frequency/energy values
        epsilon2: Array of imaginary part of dielectric function
        omega_range: Tuple (min, max) for integration range. If None, uses data range.
        max_points: Maximum number of omega points to process (default 1000)
    
    Returns:
        epsilon1_kkr: Real part calculated from KKR relation
    """
    omega = np.array(omega)
    epsilon2 = np.array(epsilon2)
    
    cs = CubicSpline(omega, epsilon2, extrapolate='linear')
    
    if omega_range is None:
        omega_min = omega[0]
        omega_max = omega[-1]
    else:
        omega_min, omega_max = omega_range
    
    num_points = min(len(omega), max_points)
    omega_subset = omega[:num_points].copy()
    epsilon1_kkr = np.zeros(num_points)
    
    print(f"    - Processing {num_points} omega points (max batch size: {max_points})")
    
    for i, xi in enumerate(omega_subset):
        def integrand(wp):
            return kkr_integrand(wp, xi, cs)
        
        result, _ = quad(integrand, omega_min, omega_max, limit=1000)
        epsilon1_kkr[i] = 1.0 + (2.0 / np.pi) * result
        print(f"      Processing omega[{i}]: {xi:.4f} eV ({i + 1}/{num_points})")
    
    return epsilon1_kkr, omega_subset

def plot_kkr_results(omega, epsilon1_kkr, epsilon2, 
                     material_id=None, w_range=None):
    """
    Plot KKR-calculated real part and imaginary part.
    Saves plot to plots folder.
    
    Args:
        omega: Array of frequency/energy values
        epsilon1_kkr: Real part calculated from KKR relation
        epsilon2: Imaginary part of dielectric function
        material_id: Material identifier for plot title and filename
        w_range: Tuple (min, max) for x-axis range. If None, uses full range.
    
    Returns:
        str: Path to saved plot file
    """
    os.makedirs('plots', exist_ok=True)
    
    omega = np.array(omega)
    epsilon1_kkr = np.array(epsilon1_kkr)
    epsilon2 = np.array(epsilon2)
    
    if w_range is not None:
        mask = (omega >= w_range[0]) & (omega <= w_range[1])
        omega_plot = omega[mask]
        epsilon1_kkr_plot = epsilon1_kkr[mask]
        epsilon2_plot = epsilon2[mask]
    else:
        omega_plot = omega
        epsilon1_kkr_plot = epsilon1_kkr
        epsilon2_plot = epsilon2
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(omega_plot, epsilon1_kkr_plot, 'r-', label='KKR ε₁', linewidth=2)
    axes[0].set_xlabel('Energy (eV)')
    axes[0].set_ylabel('ε₁ (Real part)')
    axes[0].set_title(f'Real Part of Dielectric Function (KKR){" - " + material_id if material_id else ""}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(omega_plot, epsilon2_plot, 'g-', label='ε₂ (Imaginary part)', linewidth=2)
    axes[1].set_xlabel('Energy (eV)')
    axes[1].set_ylabel('ε₂ (Imaginary part)')
    axes[1].set_title('Imaginary Part of Dielectric Function')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{material_id if material_id else 'unknown'}_kkr_plot.png"
    filepath = os.path.join('plots', filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

def validate_material(data_entry, material_index=None):
    """
    Validate a single material entry with all three checks.
    
    Args:
        data_entry: Dictionary containing material data
        material_index: Index of material being processed (for logging)
    
    Returns:
        dict: Validation results
    """
    results = {}
    
    material_id = data_entry.get('mpid', 'unknown')
    if material_index is not None:
        print(f"\n[{material_index + 1}] Processing material: {material_id}")
    else:
        print(f"\nProcessing material: {material_id}")
    
    print(f"  Step 1: Extracting dielectric tensor data...")
    epsilon_xx_real = data_entry.get('opt_realxx', [])
    epsilon_yy_real = data_entry.get('opt_realyy', [])
    epsilon_zz_real = data_entry.get('opt_realzz', [])
    epsilon_xx_imag = data_entry.get('opt_imagxx', [])
    omega = data_entry.get('opt_en', [])
    
    if not all([epsilon_xx_real, epsilon_yy_real, epsilon_zz_real, omega]):
        print(f"  ERROR: Missing required data fields")
        return {'error': 'Missing required data fields', 'material_id': material_id}
    
    results['material_id'] = material_id
    print(f"  Step 2: Checking isotropy (comparing ε_xx, ε_yy, ε_zz)...")
    isotropy_result = check_isotropy(epsilon_xx_real, epsilon_yy_real, epsilon_zz_real)
    results['isotropy'] = isotropy_result
    print(f"  Result: {isotropy_result}")
    
    print(f"  Step 3: Applying KKR relation with cubic spline interpolation...")
    print(f"    - Creating cubic spline interpolation...")
    omega = np.array(omega)
    epsilon1_kkr, omega_subset = apply_kkr_relation(omega, epsilon_xx_imag, 
                                      omega_range=(omega[0], omega[-1]), max_points=1000)
    results['epsilon1_kkr'] = epsilon1_kkr.tolist()
    print(f"    - KKR integration complete")
    
    print(f"  Step 4: Generating plot...")
    epsilon2_subset = np.array(epsilon_xx_imag)[:len(omega_subset)]
    plot_path = plot_kkr_results(omega_subset, epsilon1_kkr, epsilon2_subset,
                    material_id=results['material_id'], w_range=(0, 10))
    print(f"    - Plot saved to: {plot_path}")
    print(f"  ✓ Material {material_id} validation complete")
    
    return results

if __name__ == '__main__':
    print("Loading data from ENdiele20.json...")
    with open('ENdiele20.json', 'r') as f:
        data = json.load(f)
    
    num_materials = min(10, len(data))
    print(f"\nProcessing KKR integrand and plotting for first {num_materials} materials...")
    print("=" * 60)
    
    results = []
    for i in range(num_materials):
        material_result = validate_material(data[i], material_index=i)
        results.append(material_result)
    
    print("\n" + "=" * 60)
    print("Summary of Results:")
    print("=" * 60)
    for r in results:
        material_id = r.get('material_id', 'unknown')
        isotropy = r.get('isotropy', 'N/A')
        print(f"  {material_id}: {isotropy}")
    
    print(f"\nPlots saved to 'plots/' directory")
