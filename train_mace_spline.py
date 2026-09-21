"""
Train MACE ε(iξ) with a 32-point dielectric head and a cubic spline loss.

The last linear layer emits ε at 32 knots on the KKR ξ grid (evenly spaced in
log ξ, endpoints included). A fixed cubic-spline matrix expands those knots
onto every ξ point, and the usual full-grid MAE is applied to that curve.
The matrix does not depend on the prediction, so the loss gradient reaches
the 32 outputs through a matmul.

Forward returns the splined curve, shape (batch, channels, n_ξ), so Hamaker
eval can consume it the same way as a dense head.

    python train_mace_spline.py --use_log \\
        --lmdb_path latest_data_opt/crystal_graphs.lmdb \\
        --models_dir models_latest_opt --model_name mace_spline32_opt
"""

import argparse
import os

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from losses import FrequencyWeightedMSELoss
from mace_official_model import (
    OfficialFrequencyMACE,
    build_mace_backbone,
    load_mace_mp0_encoder,
    require_mace,
)
from model_loader import set_encoder_trainable
from train_mace_frequency import (
    checkpoint_extra,
    plot_losses,
    train_epoch,
    train_val_test_split,
    validate,
)

MODEL_NAME = "mace_spline32"


def knot_indices(n_freq, n_knots):
    """n_knots indices into a log-spaced ξ grid, including both endpoints."""
    idx = np.round(np.linspace(0, n_freq - 1, n_knots)).astype(np.int64)
    idx[0] = 0
    idx[-1] = n_freq - 1
    if len(np.unique(idx)) != n_knots:
        raise ValueError(f"n_knots={n_knots} does not fit on a grid of {n_freq} points")
    return idx


def cubic_spline_matrix(x_knots, x_query):
    """
    (F, K) matrix S with y(x_query) = S @ y(x_knots).

    Not-a-knot cubic spline. x_knots and x_query must be ascending.
    """
    k = len(x_knots)
    spline = CubicSpline(x_knots, np.eye(k), bc_type="not-a-knot")
    return np.asarray(spline(x_query), dtype=np.float64)


class SplineFrequencyMACE(OfficialFrequencyMACE):
    """
    Official MACE encoder plus a dielectric head whose last layer has
    n_knots outputs per channel. Forward splines those knots onto xi_full.
    """

    def __init__(
        self,
        mace_encoder,
        xi_full,
        n_knots=32,
        n_output_channels=1,
        h_fea_len=256,
        dropout=0.1,
        num_interactions=2,
        hidden_irreps="128x0e + 128x1o",
        feat_dim=None,
    ):
        super().__init__(
            mace_encoder=mace_encoder,
            n_frequencies=n_knots,
            n_output_channels=n_output_channels,
            h_fea_len=h_fea_len,
            dropout=dropout,
            num_interactions=num_interactions,
            hidden_irreps=hidden_irreps,
            feat_dim=feat_dim,
        )
        xi = np.asarray(xi_full, dtype=np.float64)
        knots = knot_indices(len(xi), n_knots)
        weight = cubic_spline_matrix(np.log(xi[knots]), np.log(xi))
        self.n_knots = int(n_knots)
        self.n_full = int(len(xi))
        self.register_buffer("spline_w", torch.tensor(weight, dtype=torch.float32))
        self.register_buffer("knot_index", torch.tensor(knots, dtype=torch.long))

    def forward(self, batch):
        knots = super().forward(batch)
        return torch.matmul(knots, self.spline_w.transpose(0, 1))


def build_spline_mace(args, dataset, device, n_knots):
    """Scratch or MACE-MP-0 encoder, dielectric head ending at n_knots."""
    xi = np.asarray(dataset.frequency_array, dtype=np.float64)
    n_channels = dataset.get_output_channels()
    foundation = getattr(args, "foundation_model", None)
    extra = {}
    if foundation:
        from mace.tools import AtomicNumberTable

        encoder, cfg = load_mace_mp0_encoder(foundation, device=device)
        model = SplineFrequencyMACE(
            mace_encoder=encoder,
            xi_full=xi,
            n_knots=n_knots,
            n_output_channels=n_channels,
            h_fea_len=args.h_fea_len,
            dropout=args.dropout,
            num_interactions=cfg["num_interactions"],
            hidden_irreps=cfg["hidden_irreps"],
            feat_dim=cfg["node_feat_dim"],
        ).to(device)
        dataset.set_mace_graph_config(AtomicNumberTable(zs=cfg["atomic_numbers"]), cfg["r_max"])
        args.mace_r_max = cfg["r_max"]
        args.mace_num_interactions = cfg["num_interactions"]
        args.mace_hidden_irreps = cfg["hidden_irreps"]
        extra = {
            "foundation_model": foundation,
            "mace_atomic_numbers": cfg["atomic_numbers"],
            "mace_r_max": cfg["r_max"],
            "mace_num_interactions": cfg["num_interactions"],
            "mace_hidden_irreps": cfg["hidden_irreps"],
            "mace_node_feat_dim": cfg["node_feat_dim"],
        }
        return model, extra

    encoder = build_mace_backbone(
        z_table=dataset.z_table,
        r_max=args.mace_r_max or dataset.radius,
        num_interactions=args.mace_num_interactions,
        hidden_irreps=args.mace_hidden_irreps,
        max_ell=args.mace_max_ell,
        correlation=args.mace_correlation,
        avg_num_neighbors=args.mace_avg_num_neighbors,
    )
    model = SplineFrequencyMACE(
        mace_encoder=encoder,
        xi_full=xi,
        n_knots=n_knots,
        n_output_channels=n_channels,
        h_fea_len=args.h_fea_len,
        dropout=args.dropout,
        num_interactions=args.mace_num_interactions,
        hidden_irreps=args.mace_hidden_irreps,
    ).to(device)
    extra = {"mace_atomic_numbers": list(dataset.z_table.zs)}
    return model, extra


def load_spline_mace(checkpoint, device):
    """Rebuild a spline head from a checkpoint written by this script."""
    require_mace()
    xi = np.asarray(checkpoint["kkr_xi_array"], dtype=np.float64)
    n_knots = int(checkpoint["n_knots"])
    n_output_channels = checkpoint["n_output_channels"]
    foundation = checkpoint.get("foundation_model")
    if foundation:
        encoder, cfg = load_mace_mp0_encoder(foundation, device=device)
        model = SplineFrequencyMACE(
            mace_encoder=encoder,
            xi_full=xi,
            n_knots=n_knots,
            n_output_channels=n_output_channels,
            h_fea_len=checkpoint.get("h_fea_len", 256),
            dropout=checkpoint.get("dropout", 0.1),
            num_interactions=cfg["num_interactions"],
            hidden_irreps=cfg["hidden_irreps"],
            feat_dim=cfg["node_feat_dim"],
        ).to(device)
    else:
        from mace.tools import AtomicNumberTable
        from mace_data import z_table_from_atom_init

        atomic_numbers = checkpoint.get("mace_atomic_numbers")
        if atomic_numbers is not None:
            z_table = AtomicNumberTable(zs=[int(z) for z in atomic_numbers])
        else:
            z_table = z_table_from_atom_init(checkpoint.get("atom_init_path", "atom_init.json"))
        encoder = build_mace_backbone(
            z_table=z_table,
            r_max=checkpoint.get("mace_r_max", checkpoint.get("radius", 8.0)),
            num_interactions=checkpoint.get("mace_num_interactions", 2),
            hidden_irreps=checkpoint.get("mace_hidden_irreps", "128x0e + 128x1o"),
            max_ell=checkpoint.get("mace_max_ell", 3),
            correlation=checkpoint.get("mace_correlation", 3),
            avg_num_neighbors=checkpoint.get("mace_avg_num_neighbors", 20.0),
            atomic_numbers=atomic_numbers,
        )
        model = SplineFrequencyMACE(
            mace_encoder=encoder,
            xi_full=xi,
            n_knots=n_knots,
            n_output_channels=n_output_channels,
            h_fea_len=checkpoint.get("h_fea_len", 256),
            dropout=checkpoint.get("dropout", 0.1),
            num_interactions=checkpoint.get("mace_num_interactions", 2),
            hidden_irreps=checkpoint.get("mace_hidden_irreps", "128x0e + 128x1o"),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, "MACE"


def main():
    parser = argparse.ArgumentParser(description="Train MACE ε(iξ) with a 32-knot spline head")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--target_components", type=str, default="xx")
    parser.add_argument("--use_log", action="store_true")
    parser.add_argument("--n_knots", type=int, default=32)
    parser.add_argument("--n_low_freq", type=int, default=0)
    parser.add_argument("--metal_low_freq_weight", type=float, default=0.1)
    parser.add_argument("--metal_gap_threshold", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--h_fea_len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mace_r_max", type=float, default=None)
    parser.add_argument("--mace_num_interactions", type=int, default=2)
    parser.add_argument("--mace_hidden_irreps", type=str, default="128x0e + 128x1o")
    parser.add_argument("--mace_max_ell", type=int, default=3)
    parser.add_argument("--mace_correlation", type=int, default=3)
    parser.add_argument("--mace_avg_num_neighbors", type=float, default=20.0)
    parser.add_argument("--foundation_model", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    args = parser.parse_args()

    target_components = tuple(args.target_components.split(","))
    print(f"Loading dataset from {args.lmdb_path}...")
    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=args.metal_gap_threshold,
        use_mace=True,
        mace_r_max=args.mace_r_max,
    )
    print(f"Total samples: {len(dataset)}")
    print(f"ξ points: {dataset.n_frequencies}  knots: {args.n_knots}")
    print(f"Log-space training: {args.use_log}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, foundation_ckpt = build_spline_mace(args, dataset, device, args.n_knots)
    set_encoder_trainable(model, freeze_encoder=args.freeze_encoder)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Head last layer: {args.n_knots} outputs, splined to {model.n_full}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} (trainable: {n_trainable:,})")

    train_idx, val_idx, test_idx = train_val_test_split(len(dataset))
    collate_fn = get_collate_fn(use_mace=True)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    criterion = FrequencyWeightedMSELoss(
        dataset.frequency_array,
        n_low_freq=args.n_low_freq,
        metal_low_freq_weight=args.metal_low_freq_weight,
    ).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience,
    )

    os.makedirs(args.models_dir, exist_ok=True)
    model_path = os.path.join(args.models_dir, f"{args.model_name}.pt")
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    print(f"Model will be saved to: {model_path}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(
            model, "MACE", train_loader, criterion, optimizer, device,
            use_log=args.use_log, grad_clip_norm=args.grad_clip_norm,
        )
        val_loss = validate(model, "MACE", val_loader, criterion, device, use_log=args.use_log)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "n_frequencies": dataset.n_frequencies,
                "n_knots": args.n_knots,
                "kkr_xi_array": np.asarray(dataset.frequency_array, dtype=np.float64),
                "n_output_channels": dataset.get_output_channels(),
                "target_components": target_components,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "model_name": args.model_name,
                "use_log": args.use_log,
                "n_low_freq": args.n_low_freq,
                "metal_low_freq_weight": args.metal_low_freq_weight,
                "metal_gap_threshold": args.metal_gap_threshold,
                **checkpoint_extra(args, "MACE", dataset),
                **foundation_ckpt,
            }, model_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

    plot_losses(
        train_losses, val_losses,
        save_path=os.path.join(args.models_dir, f"{args.model_name}_loss.png"),
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = validate(model, "MACE", test_loader, criterion, device, use_log=args.use_log)
    print(f"Final Test Loss: {test_loss:.6f}")
    dataset.close()


if __name__ == "__main__":
    main()
