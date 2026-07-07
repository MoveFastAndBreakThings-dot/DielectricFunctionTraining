"""
Official ACEsuit MACE (E(3)-equivariant) + custom ε(iξ) spectrum readout.

Uses ``mace.modules.ScaleShiftMACE`` as the encoder and pools ``node_feats``
from the forward pass (``compute_force=False``) into a frequency head.

Install separately (not done in this repo):
    pip install mace-torch
"""

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3

try:
    from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes
    from mace.tools.scatter import scatter_mean

    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    ScaleShiftMACE = None


def require_mace():
    if not MACE_AVAILABLE:
        raise ImportError(
            "Official MACE requires mace-torch. Install in your environment:\n"
            "  pip install mace-torch"
        )


def node_feature_dim(num_interactions, hidden_irreps, keep_last_layer_irreps=False):
    """
    Size of ``node_feats`` returned by ScaleShiftMACE (concat of all layers).

    MACE keeps only scalar irreps on the final interaction by default, so the
    last block contributes fewer channels than earlier ones.
    """
    ir = o3.Irreps(hidden_irreps)
    full_dim = ir.dim
    scalar_dim = ir.count(o3.Irrep(0, 1))
    if keep_last_layer_irreps:
        return num_interactions * full_dim
    if num_interactions == 1:
        return scalar_dim
    return (num_interactions - 1) * full_dim + scalar_dim


def build_mace_backbone(
    z_table,
    r_max=8.0,
    num_interactions=2,
    hidden_irreps="128x0e + 128x1o",
    max_ell=3,
    correlation=3,
    avg_num_neighbors=20.0,
    num_radial_basis=8,
    num_cutoff_basis=5,
    dtype=torch.float32,
):
    """Construct a fresh ScaleShiftMACE encoder (energy readout unused)."""
    require_mace()
    atomic_numbers = list(z_table.zs)
    num_elements = len(atomic_numbers)
    hidden = o3.Irreps(hidden_irreps)
    atomic_energies = np.zeros(num_elements, dtype=np.float64)

    torch.set_default_dtype(dtype)
    model = ScaleShiftMACE(
        r_max=r_max,
        num_bessel=num_radial_basis,
        num_polynomial_cutoff=num_cutoff_basis,
        max_ell=max_ell,
        interaction_cls=interaction_classes["RealAgnosticInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticInteractionBlock"],
        num_interactions=num_interactions,
        num_elements=num_elements,
        hidden_irreps=hidden,
        MLP_irreps=o3.Irreps("16x0e"),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=correlation,
        gate=gate_dict["silu"],
        atomic_inter_scale=[1.0],
        atomic_inter_shift=[0.0],
        radial_MLP=[64, 64, 64],
        heads=["Default"],
        use_last_readout_only=True,
    )
    return model


class OfficialFrequencyMACE(nn.Module):
    """
    Official MACE encoder + mean-pooled node features → ε(iξ) curve.

    Forward accepts a MACE ``Batch`` (from ``mace_data.collate_mace_pool``).
    """

    def __init__(
        self,
        mace_encoder,
        n_frequencies,
        n_output_channels=1,
        h_fea_len=256,
        dropout=0.1,
        num_interactions=2,
        hidden_irreps="128x0e + 128x1o",
    ):
        require_mace()
        super().__init__()
        self.mace = mace_encoder
        self.n_frequencies = n_frequencies
        self.n_output_channels = n_output_channels
        self.num_interactions = num_interactions
        self.hidden_irreps = hidden_irreps

        feat_dim = node_feature_dim(num_interactions, hidden_irreps)
        self.pool_to_fc = nn.Linear(feat_dim, h_fea_len)
        self.pool_activation = nn.SiLU()
        self.fc_freq = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_fea_len * 2, h_fea_len * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_fea_len * 2, n_frequencies * n_output_channels),
        )

    def forward(self, batch):
        out = self.mace(
            batch,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )
        node_feats = out["node_feats"]
        crys_fea = scatter_mean(node_feats, batch.batch, dim=0)
        crys_fea = self.pool_activation(self.pool_to_fc(crys_fea))
        flat = self.fc_freq(crys_fea)
        batch_size = flat.shape[0]
        return flat.view(batch_size, self.n_output_channels, self.n_frequencies)


def build_official_frequency_mace(
    z_table,
    n_frequencies,
    n_output_channels=1,
    r_max=8.0,
    num_interactions=2,
    hidden_irreps="128x0e + 128x1o",
    max_ell=3,
    h_fea_len=256,
    dropout=0.1,
    dtype=torch.float32,
):
    encoder = build_mace_backbone(
        z_table=z_table,
        r_max=r_max,
        num_interactions=num_interactions,
        hidden_irreps=hidden_irreps,
        max_ell=max_ell,
        dtype=dtype,
    )
    return OfficialFrequencyMACE(
        mace_encoder=encoder,
        n_frequencies=n_frequencies,
        n_output_channels=n_output_channels,
        h_fea_len=h_fea_len,
        dropout=dropout,
        num_interactions=num_interactions,
        hidden_irreps=hidden_irreps,
    )
