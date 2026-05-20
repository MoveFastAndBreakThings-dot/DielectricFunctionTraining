"""
Frozen backbone encoder + trainable per-frequency error head.

Architecture:
  embedding → MACE layers → pool → pool_to_fc → pool_activation → [fcs]
                                           ↓
                                        crys_fea   (h_fea_len,)
                                           ↓
                                        fc_err + Softplus
                                           ↓
                               err_pred  (batch, 1, n_frequencies)

fc_freq (the ε prediction head) is deliberately NOT part of this model.
The training script loads a separate frozen FrequencyMACE to compute
eps_pred and derives the abs-residual target externally.

forward() takes (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
and returns err_pred: (batch, 1, n_frequencies), all values > 0.
"""

import torch
import torch.nn as nn
from mace_model import MACELayer


class FrequencyMACEWithErrorHead(nn.Module):
    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=128,
        n_layers=4,
        h_fea_len=256,
        n_h=1,
        n_frequencies=4999,
        dropout=0.1,
        err_hidden=256,
    ):
        super().__init__()
        self.n_frequencies = n_frequencies

        # ---- frozen encoder (everything up to crys_fea, no fc_freq) ----
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.layers = nn.ModuleList([
            MACELayer(atom_fea_len, nbr_fea_len, dropout) for _ in range(n_layers)
        ])
        self.pool_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.pool_activation = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(h_fea_len, h_fea_len),
                    nn.Softplus(),
                    nn.Dropout(dropout),
                ) for _ in range(n_h - 1)
            ])
        else:
            self.fcs = None

        # ---- trainable per-frequency error head ----
        self.fc_err = nn.Sequential(
            nn.Linear(h_fea_len, err_hidden),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(err_hidden, err_hidden),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(err_hidden, n_frequencies),
        )
        self.err_pos = nn.Softplus()

    def load_encoder_from_checkpoint(self, ckpt):
        """Copy encoder weights from a FrequencyMACE checkpoint and freeze them.

        Frozen : embedding, layers, pool_to_fc, pool_activation, fcs
        Trainable: fc_err  (err_pos is Softplus — no parameters)
        """
        sd = ckpt["model_state_dict"]
        own = self.state_dict()
        encoder_keys = {k for k in own if not k.startswith("fc_err")}
        filtered = {k: v for k, v in sd.items() if k in encoder_keys}
        missing = encoder_keys - filtered.keys()
        if missing:
            raise ValueError(f"Keys missing from backbone checkpoint: {missing}")
        own.update(filtered)
        self.load_state_dict(own)
        # freeze every parameter that is NOT part of fc_err
        for name, p in self.named_parameters():
            p.requires_grad = name.startswith("fc_err")

    @staticmethod
    def _pool(atom_fea, crystal_atom_idx):
        return torch.cat([
            torch.mean(atom_fea[idx], dim=0, keepdim=True) for idx in crystal_atom_idx
        ], dim=0)

    def encode(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """Return crys_fea (h_fea_len,) — the shared crystal representation."""
        x = self.embedding(atom_fea)
        for layer in self.layers:
            x = layer(x, nbr_fea, nbr_fea_idx)
        crys_fea = self._pool(x, crystal_atom_idx)
        crys_fea = self.pool_to_fc(crys_fea)
        crys_fea = self.pool_activation(crys_fea)
        if self.fcs is not None:
            for fc in self.fcs:
                crys_fea = fc(crys_fea)
        return crys_fea

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        with torch.no_grad():
            crys_fea = self.encode(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        err_pred = self.err_pos(self.fc_err(crys_fea.detach())).unsqueeze(1)
        return err_pred  # (batch, 1, n_frequencies)
