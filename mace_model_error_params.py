"""
MACE variant that predicts ε(iξ) and power-law error parameters (a,b,c).

Outputs:
  - eps_pred: (batch, n_output_channels, n_frequencies)
  - abc_pred: (batch, 3) with positive entries via softplus
"""

import torch
import torch.nn as nn

from mace_model import MACELayer


class FrequencyMACEWithABC(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=128,
        n_layers=4,
        h_fea_len=256,
        n_h=1,
        n_frequencies=4999,
        n_output_channels=1,
        dropout=0.1,
        abc_hidden=128,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_output_channels = n_output_channels

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.layers = nn.ModuleList([MACELayer(atom_fea_len, nbr_fea_len, dropout) for _ in range(n_layers)])

        self.pool_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.pool_activation = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(h_fea_len, h_fea_len),
                        nn.Softplus(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(n_h - 1)
                ]
            )
        else:
            self.fcs = None

        self.fc_freq = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(h_fea_len * 2, h_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(h_fea_len * 2, n_frequencies * n_output_channels),
        )

        self.fc_abc = nn.Sequential(
            nn.Linear(h_fea_len, abc_hidden),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(abc_hidden, 3),
        )
        self.abc_pos = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for layer in self.layers:
            atom_fea = layer(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.pool_to_fc(crys_fea)
        crys_fea = self.pool_activation(crys_fea)
        if self.fcs is not None:
            for fc in self.fcs:
                crys_fea = fc(crys_fea)

        out_freq = self.fc_freq(crys_fea)
        batch = out_freq.shape[0]
        eps_pred = out_freq.view(batch, self.n_output_channels, self.n_frequencies)

        abc_raw = self.fc_abc(crys_fea)
        abc_pred = self.abc_pos(abc_raw)
        return eps_pred, abc_pred

    @staticmethod
    def pooling(atom_fea, crystal_atom_idx):
        assert sum(len(idx) for idx in crystal_atom_idx) == atom_fea.shape[0]
        summed_fea = [torch.mean(atom_fea[idx], dim=0, keepdim=True) for idx in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

