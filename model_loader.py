"""
Model loader utility for loading MACE models from checkpoints.
"""

import torch
from mace_model import FrequencyMACE


def load_model(checkpoint, device):
    """
    Load a FrequencyMACE model from a checkpoint.

    Parameters
    ----------
    checkpoint: dict
        Model checkpoint dictionary
    device: torch.device
        Device to load model on

    Returns
    -------
    model: nn.Module
        Loaded model
    model_type: str
        Always 'MACE'
    """
    model = FrequencyMACE(
        orig_atom_fea_len=checkpoint['orig_atom_fea_len'],
        nbr_fea_len=checkpoint['nbr_fea_len'],
        atom_fea_len=checkpoint.get('atom_fea_len', 128),
        n_layers=checkpoint.get('n_layers', 4),
        h_fea_len=checkpoint.get('h_fea_len', 256),
        n_h=checkpoint.get('n_h', 1),
        n_frequencies=checkpoint['n_frequencies'],
        n_output_channels=checkpoint['n_output_channels'],
        dropout=checkpoint.get('dropout', 0.1),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, "MACE"
