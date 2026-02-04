# ENdiele20.json Structure Explanation

## Overview

This JSON file contains a list of 285 materials, each with electronic structure and optical property calculations from Density Functional Theory (DFT) computations. Each entry represents a material with its calculated properties.

## Data Structure

The file is structured as a list of dictionaries:
```json
[
  {
    "material_property_1": value,
    "material_property_2": value,
    ...
  },
  ...
]
```

## Field Descriptions

### Material Identifiers

#### `mpid` (string)
- **Meaning**: Materials Project ID
- **Description**: Unique identifier for the material from the Materials Project database
- **Example**: "mp-1234"

#### `jid` (string)
- **Meaning**: Job ID
- **Description**: Unique identifier for the computational job/calculation
- **Example**: "jid_abc123"

#### `cif` (string)
- **Meaning**: Crystallographic Information File
- **Description**: Crystal structure information in CIF format (may be a file path or encoded structure data)
- **Example**: "path/to/structure.cif" or encoded CIF string

---

### MBJ (Modified Becke-Johnson) Calculations

The MBJ functional is a meta-GGA exchange-correlation functional used in DFT to improve band gap predictions. These fields contain dielectric tensor components calculated using the MBJ functional.

#### Real Components (string)
- `mbj_realxx`: Real part of dielectric tensor component ε_xx
- `mbj_realyy`: Real part of dielectric tensor component ε_yy
- `mbj_realzz`: Real part of dielectric tensor component ε_zz
- `mbj_realxy`: Real part of dielectric tensor component ε_xy
- `mbj_realyz`: Real part of dielectric tensor component ε_yz
- `mbj_realzx`: Real part of dielectric tensor component ε_zx

#### Imaginary Components (string)
- `mbj_imagxx`: Imaginary part of dielectric tensor component ε_xx
- `mbj_imagyy`: Imaginary part of dielectric tensor component ε_yy
- `mbj_imagzz`: Imaginary part of dielectric tensor component ε_zz
- `mbj_imagxy`: Imaginary part of dielectric tensor component ε_xy
- `mbj_imagyz`: Imaginary part of dielectric tensor component ε_yz
- `mbj_imagzx`: Imaginary part of dielectric tensor component ε_zx

**Note**: The dielectric tensor is a 3×3 symmetric matrix describing how a material responds to an electric field. The xx, yy, zz components are diagonal, while xy, yz, zx are off-diagonal components.

#### MBJ Energy and Gap
- `mbj_en` (string): Energy value from MBJ calculation (likely Fermi energy or band edge)
- `mbj_gap` (string): Band gap calculated using the MBJ functional (in eV)

---

### Optical Properties (Frequency-Dependent)

These fields contain frequency-dependent optical properties calculated from the dielectric function. Each array has 4999 data points representing different energy/frequency values.

#### Real Parts of Dielectric Tensor (list of floats, 4999 items)
- `opt_realxx`: Real part of ε_xx as a function of frequency/energy
- `opt_realyy`: Real part of ε_yy as a function of frequency/energy
- `opt_realzz`: Real part of ε_zz as a function of frequency/energy
- `opt_realxy`: Real part of ε_xy as a function of frequency/energy
- `opt_realyz`: Real part of ε_yz as a function of frequency/energy
- `opt_realzx`: Real part of ε_zx as a function of frequency/energy

#### Imaginary Parts of Dielectric Tensor (list of floats, 4999 items)
- `opt_imagxx`: Imaginary part of ε_xx as a function of frequency/energy
- `opt_imagyy`: Imaginary part of ε_yy as a function of frequency/energy
- `opt_imagzz`: Imaginary part of ε_zz as a function of frequency/energy
- `opt_imagxy`: Imaginary part of ε_xy as a function of frequency/energy
- `opt_imagyz`: Imaginary part of ε_yz as a function of frequency/energy
- `opt_imagzx`: Imaginary part of ε_zx as a function of frequency/energy

**Physical Meaning**: 
- Real part (ε_real): Related to refraction and dispersion
- Imaginary part (ε_imag): Related to absorption and optical losses
- These are typically calculated over a range of photon energies (e.g., 0-20 eV)

#### Optical Energy Array
- `opt_en` (list of floats, 4999 items): Energy values (in eV) corresponding to the optical property arrays above. Each index in this array corresponds to the same index in the `opt_real*` and `opt_imag*` arrays.

#### Optical Band Gap
- `opt_gap` (float): Optical band gap calculated from the dielectric function (in eV). This represents the minimum energy required for an electron to be excited across the band gap.

---

### Density of States (DOS) Data

Density of States describes the number of electronic states available at each energy level.

#### `opt_dos_energies` (string)
- **Meaning**: Energy values for DOS calculation (likely a file path or reference)
- **Description**: May contain a path to DOS energy data or be a placeholder

#### `opt_dos_enenrgies` (list of floats, 5000 items)
- **Meaning**: Energy array for DOS (note: appears to be a typo of "energies")
- **Description**: Energy values (in eV) at which DOS is calculated. This array has 5000 points, one more than the optical arrays.

#### `opt_dos_up` (list of floats, 5000 items)
- **Meaning**: Spin-up density of states
- **Description**: DOS values for spin-up electrons at each energy in `opt_dos_enenrgies`. In non-magnetic materials, this would be half the total DOS.

**Note**: There may be a corresponding `opt_dos_down` field for spin-down DOS, or the material may be non-magnetic.

---

### Computational Metadata

#### `opt_totcputime_per_core` (float)
- **Meaning**: Total CPU time per core
- **Description**: Computational cost of the optical property calculation, measured in seconds or hours. Useful for benchmarking and resource estimation.

---

## Tensor Notation

The dielectric tensor components use standard crystallographic notation:
- **xx, yy, zz**: Diagonal components (along principal axes)
- **xy, yz, zx**: Off-diagonal components (coupling between axes)

For cubic or isotropic materials, only diagonal components are non-zero. For lower symmetry crystals, off-diagonal components may be present.

## Data Characteristics

- **Total entries**: 285 materials
- **Optical data points**: 4999 energy points per tensor component
- **DOS data points**: 5000 energy points
- **Data types**: Mix of strings (for MBJ results, possibly stored as formatted strings) and arrays of floats (for frequency-dependent properties)

## Usage Notes

1. The MBJ fields are stored as strings, which may indicate they contain formatted output or file paths rather than numerical values.
2. The optical arrays are frequency-dependent and should be paired with `opt_en` for proper interpretation.
3. The DOS arrays have 5000 points while optical arrays have 4999, suggesting different energy grids were used.
4. This dataset appears to be from high-throughput computational materials screening, likely for optical/electronic applications.

