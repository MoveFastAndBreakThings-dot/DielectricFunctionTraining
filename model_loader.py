"""
Model loader utility for loading different model types from checkpoints.
Supports CGCNN, Matformer, and MACE models.
"""

import torch
from cgcnn_model import FrequencyCGCNN
from matformer_model import FrequencyMatformer
from mace_model import FrequencyMACE


def load_model(checkpoint, device):
    """
    Load either CGCNN, Matformer, or MACE model based on checkpoint.
    
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
        Type of model ('CGCNN', 'Matformer', or 'MACE')
    """
    # Check for Matformer first (has d_model or n_heads - most specific)
    if 'd_model' in checkpoint or 'n_heads' in checkpoint:
        # Matformer model
        d_model = checkpoint.get('d_model', 256)
        n_layers = checkpoint.get('n_layers', 4)
        n_heads = checkpoint.get('n_heads', 8)
        d_ff = checkpoint.get('d_ff', 1024)
        h_fea_len = checkpoint.get('h_fea_len', 256)
        n_h = checkpoint.get('n_h', 1)
        dropout = checkpoint.get('dropout', 0.1)
        model = FrequencyMatformer(
            orig_atom_fea_len=checkpoint['orig_atom_fea_len'],
            nbr_fea_len=checkpoint['nbr_fea_len'],
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            h_fea_len=h_fea_len,
            n_h=n_h,
            n_frequencies=checkpoint['n_frequencies'],
            n_output_channels=checkpoint['n_output_channels'],
            dropout=dropout
        ).to(device)
        model_type = "Matformer"
    # Check for MACE (has n_layers but not d_model/n_heads, and not n_conv)
    elif 'mace' in checkpoint.get('model_name', '').lower() or \
         ('n_layers' in checkpoint and 'd_model' not in checkpoint and 'n_heads' not in checkpoint and 'n_conv' not in checkpoint):
        # MACE model
        atom_fea_len = checkpoint.get('atom_fea_len', 128)
        n_layers = checkpoint.get('n_layers', 4)
        h_fea_len = checkpoint.get('h_fea_len', 256)
        n_h = checkpoint.get('n_h', 1)
        dropout = checkpoint.get('dropout', 0.1)
        model = FrequencyMACE(
            orig_atom_fea_len=checkpoint['orig_atom_fea_len'],
            nbr_fea_len=checkpoint['nbr_fea_len'],
            atom_fea_len=atom_fea_len,
            n_layers=n_layers,
            h_fea_len=h_fea_len,
            n_h=n_h,
            n_frequencies=checkpoint['n_frequencies'],
            n_output_channels=checkpoint['n_output_channels'],
            dropout=dropout
        ).to(device)
        model_type = "MACE"
    else:
        # CGCNN model
        atom_fea_len = checkpoint.get('atom_fea_len', 64)
        n_conv = checkpoint.get('n_conv', 3)
        h_fea_len = checkpoint.get('h_fea_len', 128)
        n_h = checkpoint.get('n_h', 1)
        dropout = checkpoint.get('dropout', 0.1)
        model = FrequencyCGCNN(
            orig_atom_fea_len=checkpoint['orig_atom_fea_len'],
            nbr_fea_len=checkpoint['nbr_fea_len'],
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            n_frequencies=checkpoint['n_frequencies'],
            n_output_channels=checkpoint['n_output_channels'],
            dropout=dropout
        ).to(device)
        model_type = "CGCNN"
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, model_type

