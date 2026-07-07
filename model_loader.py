"""
Load CGCNN-style or official MACE frequency models from checkpoints.
"""

import torch

from mace_model import FrequencyCGCNN
from mace_official_model import OfficialFrequencyMACE, build_mace_backbone, require_mace


def _load_cgcnn(checkpoint, device):
    model = FrequencyCGCNN(
        orig_atom_fea_len=checkpoint["orig_atom_fea_len"],
        nbr_fea_len=checkpoint["nbr_fea_len"],
        atom_fea_len=checkpoint.get("atom_fea_len", 128),
        n_layers=checkpoint.get("n_layers", 4),
        h_fea_len=checkpoint.get("h_fea_len", 256),
        n_h=checkpoint.get("n_h", 1),
        n_frequencies=checkpoint["n_frequencies"],
        n_output_channels=checkpoint["n_output_channels"],
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, "CGCNN"


def _load_official_mace(checkpoint, device):
    require_mace()
    from mace_data import z_table_from_atom_init

    z_table = z_table_from_atom_init(checkpoint.get("atom_init_path", "atom_init.json"))
    encoder = build_mace_backbone(
        z_table=z_table,
        r_max=checkpoint.get("mace_r_max", checkpoint.get("radius", 8.0)),
        num_interactions=checkpoint.get("mace_num_interactions", 2),
        hidden_irreps=checkpoint.get("mace_hidden_irreps", "128x0e + 128x1o"),
        max_ell=checkpoint.get("mace_max_ell", 3),
        correlation=checkpoint.get("mace_correlation", 3),
        avg_num_neighbors=checkpoint.get("mace_avg_num_neighbors", 20.0),
    )
    model = OfficialFrequencyMACE(
        mace_encoder=encoder,
        n_frequencies=checkpoint["n_frequencies"],
        n_output_channels=checkpoint["n_output_channels"],
        h_fea_len=checkpoint.get("h_fea_len", 256),
        dropout=checkpoint.get("dropout", 0.1),
        num_interactions=checkpoint.get("mace_num_interactions", 2),
        hidden_irreps=checkpoint.get("mace_hidden_irreps", "128x0e + 128x1o"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, "MACE"


def load_model(checkpoint, device):
    """
    Load a frequency model from a checkpoint.

    Returns (model, model_type) where model_type is ``"CGCNN"`` or ``"MACE"``.
    """
    model_type = checkpoint.get("model_type", "CGCNN")
    if model_type == "MACE":
        return _load_official_mace(checkpoint, device)
    return _load_cgcnn(checkpoint, device)


def build_train_model(args, dataset, device):
    """Construct a fresh model for training."""
    n_frequencies = dataset.n_frequencies
    n_output_channels = dataset.get_output_channels()

    if args.model == "mace":
        require_mace()
        from mace_official_model import build_official_frequency_mace

        z_table = dataset.z_table
        model = build_official_frequency_mace(
            z_table=z_table,
            n_frequencies=n_frequencies,
            n_output_channels=n_output_channels,
            r_max=args.mace_r_max or dataset.radius,
            num_interactions=args.mace_num_interactions,
            hidden_irreps=args.mace_hidden_irreps,
            max_ell=args.mace_max_ell,
            h_fea_len=args.h_fea_len,
            dropout=args.dropout,
        ).to(device)
        return model, "MACE"

    model = FrequencyCGCNN(
        orig_atom_fea_len=dataset.atom_fea_len,
        nbr_fea_len=dataset.nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_layers=args.n_layers,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        n_frequencies=n_frequencies,
        n_output_channels=n_output_channels,
        dropout=args.dropout,
    ).to(device)
    return model, "CGCNN"
