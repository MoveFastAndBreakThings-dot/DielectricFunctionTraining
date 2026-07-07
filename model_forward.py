"""
Unified forward pass for CGCNN-style and official MACE models.
"""

import torch


def model_forward(model, model_type, inputs, device, use_log=False, channel=0):
    """
    Run model and return linear ε(iξ) for one channel.

    Parameters
    ----------
    model_type : str
        ``"CGCNN"`` or ``"MACE"``
    inputs : tuple or Batch
        CGCNN: (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        MACE:  Batch from mace_data.collate_mace_pool
    channel : int
        Output channel index (default 0 = xx)

    Returns
    -------
    pred_lin : (B, F) tensor
    """
    model.eval()
    with torch.no_grad():
        if model_type == "MACE":
            batch = inputs.to(device)
            out = model(batch)[:, channel, :]
        else:
            af, nf, nfi, cai = inputs
            af = af.to(device)
            nf = nf.to(device)
            nfi = nfi.to(device)
            cai = [x.to(device) for x in cai]
            out = model(af, nf, nfi, cai)[:, channel, :]
        if use_log:
            return torch.exp(out)
        return out


def model_forward_train(model, model_type, inputs, device):
    """Training forward — returns full (B, C, F) tensor."""
    if model_type == "MACE":
        batch = inputs.to(device)
        return model(batch)
    af, nf, nfi, cai = inputs
    af = af.to(device)
    nf = nf.to(device)
    nfi = nfi.to(device)
    cai = [x.to(device) for x in cai]
    return model(af, nf, nfi, cai)
