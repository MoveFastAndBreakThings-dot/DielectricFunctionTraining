"""
Train a model to predict KKR-transformed dielectric function ε(iξ).

Backends:
    --model cgcnn  CGCNN-style distance graph (default; old checkpoints)
    --model mace   Official ACEsuit MACE (requires mace-torch + geometry in LMDB)

Usage:
    python preprocess_dataset.py --input_dir data --output_dir processed_data
    python train_mace_frequency.py --model cgcnn
    python train_mace_frequency.py --model mace
    # Fine-tune from MACE-MP-0 (downloads once into mace cache on the server):
    python train_mace_frequency.py --model mace --foundation_model medium \\
        --model_name mace_mp0_medium_kkr
    python train_mace_frequency.py --model mace --foundation_model medium \\
        --freeze_encoder --model_name mace_mp0_medium_head_only
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, get_collate_fn
from losses import FrequencyWeightedMSELoss
from model_forward import model_forward_train
from model_loader import build_train_model, set_encoder_trainable

MODEL_NAME = "mace_kkr_v1"


def train_epoch(model, model_type, dataloader, criterion, optimizer, device, use_log=False, grad_clip_norm=1.0):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch_idx, (inputs, target, is_metal) in enumerate(dataloader):
        target = target.to(device)
        if use_log:
            target = torch.log(target)
        if is_metal is not None:
            is_metal = is_metal.to(device)

        optimizer.zero_grad()
        output = model_forward_train(model, model_type, inputs, device)
        loss = criterion(output, target, is_metal)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item() * target.shape[0]
        n_samples += target.shape[0]

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")

    return total_loss / n_samples


def validate(model, model_type, dataloader, criterion, device, use_log=False):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs, target, is_metal in dataloader:
            target = target.to(device)
            if use_log:
                target = torch.log(target)
            if is_metal is not None:
                is_metal = is_metal.to(device)

            output = model_forward_train(model, model_type, inputs, device)
            loss = criterion(output, target, is_metal)

            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]

    return total_loss / n_samples


def plot_losses(train_losses, val_losses, save_path="training_loss.png"):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    plt.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.5)
    plt.scatter([best_epoch], [best_val], color="g", s=100, zorder=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def train_val_test_split(n_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size:train_size + val_size].tolist()
    test_idx = indices[train_size + val_size:].tolist()
    return train_idx, val_idx, test_idx


def checkpoint_extra(args, model_type, dataset):
    base = {
        "model_type": model_type,
        "radius": dataset.radius,
        "max_num_nbr": dataset.max_num_nbr,
        "h_fea_len": args.h_fea_len,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "scheduler_factor": args.scheduler_factor,
        "scheduler_patience": args.scheduler_patience,
    }
    if model_type == "MACE":
        base.update({
            "mace_r_max": args.mace_r_max or dataset.radius,
            "mace_num_interactions": args.mace_num_interactions,
            "mace_hidden_irreps": args.mace_hidden_irreps,
            "mace_max_ell": args.mace_max_ell,
            "mace_correlation": args.mace_correlation,
            "mace_avg_num_neighbors": args.mace_avg_num_neighbors,
            "foundation_model": getattr(args, "foundation_model", None),
            "freeze_encoder": getattr(args, "freeze_encoder", False),
            "mace_atomic_numbers": list(dataset.z_table.zs) if hasattr(dataset, "z_table") else None,
        })
    else:
        base.update({
            "orig_atom_fea_len": dataset.atom_fea_len,
            "nbr_fea_len": dataset.nbr_fea_len,
            "atom_fea_len": args.atom_fea_len,
            "n_layers": args.n_layers,
            "n_h": args.n_h,
        })
    return base


def main():
    parser = argparse.ArgumentParser(description="Train ε(iξ) model on KKR data")
    parser.add_argument("--model", type=str, default="cgcnn", choices=("cgcnn", "mace"),
                        help="cgcnn = distance-graph baseline; mace = official ACEsuit MACE")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--target_components", type=str, default="xx")
    parser.add_argument("--use_log", action="store_true")
    parser.add_argument("--n_low_freq", type=int, default=0)
    parser.add_argument("--metal_low_freq_weight", type=float, default=0.1)
    parser.add_argument("--metal_gap_threshold", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)

    # CGCNN hyperparameters
    parser.add_argument("--atom_fea_len", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--h_fea_len", type=int, default=256)
    parser.add_argument("--n_h", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Official MACE hyperparameters (ignored when --foundation_model is set;
    # hypers are taken from the loaded foundation checkpoint instead)
    parser.add_argument("--mace_r_max", type=float, default=None)
    parser.add_argument("--mace_num_interactions", type=int, default=2)
    parser.add_argument("--mace_hidden_irreps", type=str, default="128x0e + 128x1o")
    parser.add_argument("--mace_max_ell", type=int, default=3)
    parser.add_argument("--mace_correlation", type=int, default=3)
    parser.add_argument("--mace_avg_num_neighbors", type=float, default=20.0)
    parser.add_argument(
        "--foundation_model", type=str, default=None,
        help="Init MACE encoder from MACE-MP-0 (e.g. small/medium/large) "
             "or a local .model path. Requires --model mace.",
    )
    parser.add_argument(
        "--freeze_encoder", action="store_true",
        help="Freeze MACE encoder; train only pool_to_fc + fc_freq readout.",
    )

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    args = parser.parse_args()

    use_mace = args.model == "mace"
    target_components = tuple(args.target_components.split(","))

    if args.foundation_model and not use_mace:
        raise SystemExit("--foundation_model requires --model mace")

    print(f"Loading dataset from {args.lmdb_path}...")
    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=args.metal_gap_threshold,
        use_mace=use_mace,
        mace_r_max=args.mace_r_max,
    )

    print(f"Backend: {args.model.upper()}")
    print(f"Total samples: {len(dataset)}")
    print(f"ξ points: {dataset.n_frequencies}")
    print(f"Target components: {target_components}")
    print(f"Output channels: {dataset.get_output_channels()}")
    print(f"Log-space training: {args.use_log}")
    if args.foundation_model:
        print(f"Foundation init: {args.foundation_model}")
        print(f"Freeze encoder: {args.freeze_encoder}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model before loaders so foundation hypers can rewrite dataset z_table / r_max
    model, model_type, foundation_ckpt = build_train_model(args, dataset, device)
    set_encoder_trainable(model, freeze_encoder=args.freeze_encoder and use_mace)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model type: {model_type}")
    print(f"Model parameters: {n_total:,} (trainable: {n_trainable:,})")

    train_idx, val_idx, test_idx = train_val_test_split(len(dataset))
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    collate_fn = get_collate_fn(use_mace=use_mace)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=False, num_workers=0, collate_fn=collate_fn)

    criterion = FrequencyWeightedMSELoss(
        dataset.frequency_array,
        n_low_freq=args.n_low_freq,
        metal_low_freq_weight=args.metal_low_freq_weight,
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience,
    )

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{args.model_name}.pt"
    n_frequencies = dataset.n_frequencies
    n_output_channels = dataset.get_output_channels()

    print(f"\nModel will be saved to: {model_path}")
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(
            model, model_type, train_loader, criterion, optimizer, device,
            use_log=args.use_log, grad_clip_norm=args.grad_clip_norm,
        )
        val_loss = validate(model, model_type, val_loader, criterion, device, use_log=args.use_log)
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
                "n_frequencies": n_frequencies,
                "n_output_channels": n_output_channels,
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
                **checkpoint_extra(args, model_type, dataset),
                **foundation_ckpt,
            }, model_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

    plot_losses(train_losses, val_losses, save_path=f"models/{args.model_name}_loss.png")

    print("\nEvaluating on test set...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = validate(model, model_type, test_loader, criterion, device, use_log=args.use_log)
    print(f"Final Test Loss: {test_loss:.6f}")

    dataset.close()


if __name__ == "__main__":
    main()
