"""
FrequencyMACE variant that directly predicts per-point σ(ξ) at every frequency.

Architecture is identical to FrequencyMACE; the only difference is that the
output is forced positive via Softplus so it can be interpreted as uncertainty.

Outputs: sigma_pred  (batch, 1, n_frequencies)  — all values > 0
"""

import torch.nn.functional as F
from mace_model import FrequencyMACE


class FrequencyMACEDirectSigma(FrequencyMACE):
    """Identical to FrequencyMACE but output is Softplus-activated (all positive)."""

    def __init__(self, *args, **kwargs):
        # Force n_output_channels=1: we always predict a single σ(ξ) curve
        kwargs["n_output_channels"] = 1
        super().__init__(*args, **kwargs)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        out = super().forward(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        return F.softplus(out)  # (batch, 1, n_frequencies), all > 0
