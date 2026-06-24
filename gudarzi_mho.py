"""
Modified harmonic oscillator ε(iξ) model from Moazzami Gudarzi & Aboutalebi,
Sci. Adv. 2021, eabg2272 (abg2272).

Electronic part (Eq. 5), parameters predicted from optical bandgap only:

    ε(iξ) = 1 + C_UV / (1 + (ξ / ω_UV)^α)
    C_UV  = A · Eg^{-1.2}
    ω_UV  = B · Eg^{0.736}

Defaults: A ≈ 20 (inorganic), B = 3.05 (inorganic) / 3.55 (organic), α = 1.7.
"""

import re

import numpy as np

B_INORGANIC = 3.05
B_ORGANIC = 3.55
A_INORGANIC = 20.0
DEFAULT_ALPHA = 1.7

ORGANIC_ELEMENTS = frozenset(
    "H C N O F P S Cl Br I Se B Si".split(),
)


def is_organic_formula(formula):
    """Heuristic: all elements are typical organic/light nonmetal constituents."""
    tokens = re.findall(r"[A-Z][a-z]?", str(formula))
    if not tokens:
        return False
    return all(el in ORGANIC_ELEMENTS for el in tokens)


def eps_mho_from_bandgap(xi, eg_ev, formula=None, alpha=DEFAULT_ALPHA, a_pref=A_INORGANIC):
    """
    Predict ε(iξ) on ascending xi grid (eV) using bandgap-only MHO model.

    Returns array same shape as xi.
    """
    xi = np.asarray(xi, dtype=np.float64)
    eg = float(eg_ev)
    if eg <= 0:
        raise ValueError(f"bandgap must be positive, got {eg}")

    organic = is_organic_formula(formula) if formula is not None else False
    b = B_ORGANIC if organic else B_INORGANIC
    c_uv = a_pref * (eg ** -1.2)
    omega_uv = b * (eg ** 0.736)
    return 1.0 + c_uv / (1.0 + (xi / omega_uv) ** alpha)
