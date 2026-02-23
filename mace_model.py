"""
MACE Model Implementation
For frequency-dependent property prediction on crystal structures.

MACE (Many-body Atomic Cluster Expansion) uses message passing with
equivariant features and many-body interactions.
Compatible with the same data format as CGCNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MACELayer(nn.Module):
    """
    MACE message passing layer with many-body interactions.
    """
    def __init__(self, atom_fea_len, nbr_fea_len, dropout=0.1):
        """
        Parameters
        ----------
        atom_fea_len: int
            Number of atom hidden features
        nbr_fea_len: int
            Number of bond/edge features
        dropout: float
            Dropout probability
        """
        super(MACELayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # Message network: processes edge features and atom features
        self.message_net = nn.Sequential(
            nn.Linear(atom_fea_len * 2 + nbr_fea_len, atom_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(atom_fea_len * 2, atom_fea_len)
        )
        
        # Update network for atom features
        self.update_net = nn.Sequential(
            nn.Linear(atom_fea_len * 2, atom_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(atom_fea_len * 2, atom_fea_len)
        )
        
        self.layer_norm = nn.LayerNorm(atom_fea_len)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Edge/neighbor features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
            
        Returns
        -------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Updated atom features
        """
        N, M = nbr_fea.shape[0], nbr_fea.shape[1]
        
        # Get neighbor atom features
        atom_nbr_fea = atom_fea[nbr_fea_idx]  # (N, M, atom_fea_len)
        
        # Expand current atom features
        atom_fea_expanded = atom_fea.unsqueeze(1).expand(-1, M, -1)  # (N, M, atom_fea_len)
        
        # Compute messages: concatenate [atom_i, atom_j, edge_ij]
        message_input = torch.cat([atom_fea_expanded, atom_nbr_fea, nbr_fea], dim=2)  # (N, M, 2*atom_fea_len + nbr_fea_len)
        messages = self.message_net(message_input)  # (N, M, atom_fea_len)
        
        # Aggregate messages (sum aggregation for MACE)
        aggregated_messages = torch.sum(messages, dim=1)  # (N, atom_fea_len)
        
        # Update atom features: concatenate [atom_i, aggregated_messages]
        update_input = torch.cat([atom_fea, aggregated_messages], dim=1)  # (N, 2*atom_fea_len)
        atom_update = self.update_net(update_input)  # (N, atom_fea_len)
        atom_update = self.layer_norm(atom_update)
        
        # Residual connection
        atom_fea = atom_fea + atom_update
        
        return atom_fea


class FrequencyMACE(nn.Module):
    """
    MACE model for frequency-dependent property prediction.
    Uses message passing with many-body interactions.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=128, n_layers=4, h_fea_len=256,
                 n_h=1, n_frequencies=4999, n_output_channels=2, dropout=0.1):
        """
        Initialize FrequencyMACE.
        
        Parameters
        ----------
        orig_atom_fea_len: int
            Number of atom features in the input
        nbr_fea_len: int
            Number of bond/edge features
        atom_fea_len: int
            Number of hidden atom features
        n_layers: int
            Number of message passing layers
        h_fea_len: int
            Hidden dimension after pooling
        n_h: int
            Number of hidden layers after pooling
        n_frequencies: int
            Number of frequency/energy points to predict
        n_output_channels: int
            Number of output channels
        dropout: float
            Dropout probability
        """
        super(FrequencyMACE, self).__init__()
        self.n_frequencies = n_frequencies
        self.n_output_channels = n_output_channels
        
        # Input embedding
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            MACELayer(atom_fea_len, nbr_fea_len, dropout)
            for _ in range(n_layers)
        ])
        
        # Pooling: mean pooling over atoms in each crystal
        self.pool_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.pool_activation = nn.Softplus()
        
        # Additional fully connected layers
        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(h_fea_len, h_fea_len),
                    nn.Softplus(),
                    nn.Dropout(dropout)
                )
                for _ in range(n_h - 1)
            ])
        else:
            self.fcs = None
        
        # Output layers for frequency-dependent prediction
        self.fc_freq = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(h_fea_len * 2, h_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(h_fea_len * 2, n_frequencies * n_output_channels)
        )
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
            Atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Edge/neighbor features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
        crystal_atom_idx: list of torch.LongTensor
            Mapping from crystal idx to atom idx
            
        Returns
        -------
        prediction: torch.Tensor shape (batch_size, n_output_channels, n_frequencies)
            Predicted dielectric properties
        """
        # Embed atoms
        atom_fea = self.embedding(atom_fea)  # (N, atom_fea_len)
        
        # Apply message passing layers
        for layer in self.layers:
            atom_fea = layer(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Pool to crystal features (mean pooling)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        
        # Project to hidden dimension
        crys_fea = self.pool_to_fc(crys_fea)
        crys_fea = self.pool_activation(crys_fea)
        
        # Additional fully connected layers
        if self.fcs is not None:
            for fc in self.fcs:
                crys_fea = fc(crys_fea)
        
        # Predict frequency-dependent values
        output = self.fc_freq(crys_fea)  # (batch_size, n_frequencies * n_output_channels)
        
        # Reshape to (batch_size, n_output_channels, n_frequencies)
        batch_size = output.shape[0]
        output = output.view(batch_size, self.n_output_channels, self.n_frequencies)
        
        return output
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features (mean pooling)
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
            Atom feature vectors
        crystal_atom_idx: list of torch.LongTensor
            Mapping from crystal idx to atom idx
            
        Returns
        -------
        crys_fea: torch.Tensor shape (batch_size, atom_fea_len)
            Crystal feature vectors
        """
        assert sum([len(idx) for idx in crystal_atom_idx]) == atom_fea.shape[0]
        summed_fea = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                      for idx in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

