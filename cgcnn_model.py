"""
CGCNN Model Implementation
Based on the original implementation from https://github.com/txie-93/cgcnn

Original paper:
Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable 
Prediction of Material Properties
Xie & Grossman, Phys. Rev. Lett. 120, 145301 (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrystalGraphConv(nn.Module):
    """
    Convolutional operation on crystal graphs.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize CrystalGraphConv.

        Parameters
        ----------
        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(CrystalGraphConv, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                  2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features.
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's neighbors.
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom.

        Returns
        -------
        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Updated atom hidden features.
        """
        N, M = nbr_fea.shape[0], nbr_fea.shape[1]
        # convolution
        atom_nbr_fea = atom_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_fea + nbr_sumed)
        return out


class CGCNN(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128,
                 n_h=1, classification=False):
        """
        Initialize CGCNN.

        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        classification: bool
          If True, use classification loss (binary cross entropy)
          If False, use regression loss (MSE)
        """
        super(CGCNN, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CrystalGraphConv(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                            for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------
        prediction: nn.Variable
          Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        crys_fea = self.dropout(crys_fea)
        out = self.fc_out(crys_fea)
        if self.classification:
            out = F.log_softmax(out, dim=1)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx) for idx in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                      for idx in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class FrequencyCGCNN(nn.Module):
    """
    CGCNN adapted for frequency-dependent property prediction.
    Predicts dielectric properties across a frequency range.
    Supports both original (real/imag) and KKR-transformed targets.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128,
                 n_h=1, n_frequencies=4999, n_output_channels=2):
        """
        Initialize FrequencyCGCNN.

        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        n_frequencies: int
          Number of frequency/energy points to predict
        n_output_channels: int
          Number of output channels. For original data: 2 (real + imag).
          For KKR: 1 per component. For multiple components, multiply accordingly.
        """
        super(FrequencyCGCNN, self).__init__()
        self.n_frequencies = n_frequencies
        self.n_output_channels = n_output_channels
        
        # Base CGCNN (without final output layer)
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CrystalGraphConv(
            atom_fea_len=atom_fea_len,
            nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                            for _ in range(n_h-1)])
        else:
            self.fcs = None
        
        # Output layers for frequency-dependent prediction
        self.fc_freq = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(h_fea_len * 2, h_fea_len * 2),
            nn.Softplus(),
            nn.Dropout(0.1),
            nn.Linear(h_fea_len * 2, n_frequencies * n_output_channels)
        )
        
    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------
        prediction: torch.Tensor shape (N0, n_output_channels, n_frequencies)
          Predicted dielectric properties
        """
        # Convolutional layers
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        
        # Pooling to crystal features
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        # Additional fully connected layers
        if self.fcs is not None:
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        # Predict frequency-dependent values
        output = self.fc_freq(crys_fea)  # [batch_size, n_frequencies * n_output_channels]
        
        # Reshape to [batch_size, n_output_channels, n_frequencies]
        batch_size = output.shape[0]
        output = output.view(batch_size, self.n_output_channels, self.n_frequencies)
        
        return output
    
    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        Parameters
        ----------
        atom_fea: torch.Tensor shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx) for idx in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx], dim=0, keepdim=True)
                      for idx in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
