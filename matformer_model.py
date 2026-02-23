"""
Matformer / Graph Transformer Model Implementation
For frequency-dependent property prediction on crystal structures.

Based on transformer architecture with graph attention mechanisms.
Compatible with the same data format as CGCNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer with edge features.
    Implements multi-head attention over crystal graph.
    """
    def __init__(self, d_model, n_heads, nbr_fea_len, dropout=0.1):
        """
        Parameters
        ----------
        d_model: int
            Model dimension (must be divisible by n_heads)
        n_heads: int
            Number of attention heads
        nbr_fea_len: int
            Dimension of edge/neighbor features
        dropout: float
            Dropout probability
        """
        super(GraphAttentionLayer, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.nbr_fea_len = nbr_fea_len
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Edge feature projection
        self.W_e = nn.Linear(nbr_fea_len, n_heads)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, mask=None):
        """
        Forward pass
        
        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, d_model)
            Atom features
        nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
            Edge/neighbor features
        nbr_fea_idx: torch.LongTensor shape (N, M)
            Neighbor indices
        mask: torch.Tensor shape (N, M), optional
            Attention mask (1 for valid neighbors, 0 for padding)
            
        Returns
        -------
        out: torch.Tensor shape (N, d_model)
            Updated atom features
        """
        N, M = nbr_fea.shape[0], nbr_fea.shape[1]
        residual = atom_fea
        
        # Project to Q, K, V
        Q = self.W_q(atom_fea).view(N, self.n_heads, self.d_k)  # (N, n_heads, d_k)
        K = self.W_k(atom_fea).view(N, self.n_heads, self.d_k)  # (N, n_heads, d_k)
        V = self.W_v(atom_fea).view(N, self.n_heads, self.d_k)  # (N, n_heads, d_k)
        
        # Gather neighbor keys and values
        # For each head, we need to gather neighbors
        # K is (N, n_heads, d_k), nbr_fea_idx is (N, M)
        # We want K_nbr[i, h, m, :] = K[nbr_fea_idx[i, m], h, :]
        K_nbr_list = []
        V_nbr_list = []
        for h in range(self.n_heads):
            # For head h: K[:, h, :] is (N, d_k), index with nbr_fea_idx to get (N, M, d_k)
            K_h = K[:, h, :]  # (N, d_k)
            V_h = V[:, h, :]  # (N, d_k)
            K_nbr_h = K_h[nbr_fea_idx]  # (N, M, d_k)
            V_nbr_h = V_h[nbr_fea_idx]  # (N, M, d_k)
            K_nbr_list.append(K_nbr_h)
            V_nbr_list.append(V_nbr_h)
        
        # Stack: (n_heads, N, M, d_k) -> (N, n_heads, M, d_k)
        K_nbr = torch.stack(K_nbr_list, dim=1)  # (N, n_heads, M, d_k)
        V_nbr = torch.stack(V_nbr_list, dim=1)  # (N, n_heads, M, d_k)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        Q_expanded = Q.unsqueeze(2)  # (N, n_heads, 1, d_k)
        scores = torch.matmul(Q_expanded, K_nbr.transpose(-2, -1)) / math.sqrt(self.d_k)  # (N, n_heads, 1, M)
        scores = scores.squeeze(2)  # (N, n_heads, M)
        
        # Add edge features to attention scores
        edge_attn = self.W_e(nbr_fea)  # (N, M, n_heads)
        edge_attn = edge_attn.transpose(1, 2)  # (N, n_heads, M)
        scores = scores + edge_attn
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand(-1, self.n_heads, -1)  # (N, n_heads, M)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)  # (N, n_heads, M)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: (N, n_heads, M), V_nbr: (N, n_heads, M, d_k)
        # We want: sum_m(attn_weights[i, h, m] * V_nbr[i, h, m, :]) = (N, n_heads, d_k)
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # (N, n_heads, M, 1)
        attn_output = torch.sum(attn_weights_expanded * V_nbr, dim=2)  # (N, n_heads, d_k)
        
        # Concatenate heads
        attn_output = attn_output.view(N, self.d_model)  # (N, d_model)
        
        # Output projection
        out = self.W_o(attn_output)
        out = self.dropout(out)
        
        # Residual connection and layer norm
        out = self.layer_norm(out + residual)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class TransformerBlock(nn.Module):
    """Transformer block: attention + feed-forward."""
    def __init__(self, d_model, n_heads, nbr_fea_len, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = GraphAttentionLayer(d_model, n_heads, nbr_fea_len, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, mask=None):
        atom_fea = self.attention(atom_fea, nbr_fea, nbr_fea_idx, mask)
        atom_fea = self.feed_forward(atom_fea)
        return atom_fea


class FrequencyMatformer(nn.Module):
    """
    Matformer model for frequency-dependent property prediction.
    Uses graph transformer architecture with multi-head attention.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 d_model=256, n_layers=4, n_heads=8,
                 d_ff=1024, h_fea_len=256, n_h=1,
                 n_frequencies=4999, n_output_channels=2, dropout=0.1):
        """
        Initialize FrequencyMatformer.
        
        Parameters
        ----------
        orig_atom_fea_len: int
            Number of atom features in the input
        nbr_fea_len: int
            Number of bond/edge features
        d_model: int
            Model dimension (must be divisible by n_heads)
        n_layers: int
            Number of transformer layers
        n_heads: int
            Number of attention heads
        d_ff: int
            Feed-forward dimension
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
        super(FrequencyMatformer, self).__init__()
        self.n_frequencies = n_frequencies
        self.n_output_channels = n_output_channels
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(orig_atom_fea_len, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, nbr_fea_len, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Pooling: mean pooling over atoms in each crystal
        self.pool_to_fc = nn.Linear(d_model, h_fea_len)
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
        N = atom_fea.shape[0]
        
        # Embed atoms
        atom_fea = self.embedding(atom_fea)  # (N, d_model)
        
        # Create attention mask (1 for valid neighbors, 0 for padding)
        # Neighbor indices should be valid (within [0, N)), but we create mask for safety
        # In practice, collate_pool ensures valid indices, but we mask any that might be out of bounds
        mask = (nbr_fea_idx >= 0) & (nbr_fea_idx < N)
        mask = mask.float()
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            atom_fea = layer(atom_fea, nbr_fea, nbr_fea_idx, mask)
        
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
        atom_fea: torch.Tensor shape (N, d_model)
            Atom feature vectors
        crystal_atom_idx: list of torch.LongTensor
            Mapping from crystal idx to atom idx
            
        Returns
        -------
        crys_fea: torch.Tensor shape (batch_size, d_model)
            Crystal feature vectors
        """
        assert sum([len(idx) for idx in crystal_atom_idx]) == atom_fea.shape[0]
        summed_fea = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                      for idx in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

