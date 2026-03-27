"""
Train a MACE model to predict ε(iξ) and per-crystal power-law error params (a,b,c).

Targets:
  - ε(iξ) from LMDB (KKR targets)
Supervision for (a,b,c):
  - match σ(ξ; a,b,c) to |ε_pred - ε_true| within each batch (correlated across ξ).

Usage:
  python train_mace_frequency_with_error_params.py
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from mace_model_error_params import FrequencyMACEWithABC
from losses_mace_error_params import SpectrumAndPowerLawABCLoss


MODEL_NAME = "mace_kkr_with_abc_v1"


def train_epoch(model, dataloader, criterion, optimizer, device, use_log=False, grad_clip_norm=1.0):
    model.train()
    total = 0.0
    n = 0
    last_parts = None
    for batch_idx, (inputs, target, _) in enumerate(dataloader):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [x.to(device) for x in crystal_atom_idx]
        target = target.to(device)

        optimizer.zero_grad()
        eps_out, abc_out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

        criterion.eps_pred_is_log = bool(use_log)
        loss, parts = criterion(eps_out, abc_out, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total += loss.item() * target.shape[0]
        n += target.shape[0]
        last_parts = parts
        if batch_idx % 10 == 0:
            msg = f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}"
            if last_parts is not None:
                msg += f" (eps={last_parts['loss_eps']:.4f}, sigma={last_parts['loss_sigma']:.4f})"
            print(msg)
    return total / max(n, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_log=False):
    model.eval()
    total = 0.0
    n = 0
    for inputs, target, _ in dataloader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [x.to(device) for x in crystal_atom_idx]
        target = target.to(device)

        eps_out, abc_out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        criterion.eps_pred_is_log = bool(use_log)
        loss, _ = criterion(eps_out, abc_out, target)
        total += loss.item() * target.shape[0]
        n += target.shape[0]
    return total / max(n, 1)


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Val", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_val_test_split(n_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size : train_size + val_size].tolist()
    test_idx = indices[train_size + val_size :].tolist()
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Train FrequencyMACEWithABC on KKR ε(iξ)")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target_components", type=str, default="xx")
    parser.add_argument("--use_log", action="store_true", help="Model outputs log(ε); loss uses exp(output).")

    parser.add_argument("--atom_fea_len", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--h_fea_len", type=int, default=256)
    parser.add_argument("--n_h", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--abc_hidden", type=int, default=128)

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--alpha", type=float, default=3.0, help="Fixed α in σ(ξ)=a+b/(ξ+c)^α")
    parser.add_argument("--lambda_eps", type=float, default=1.0)
    parser.add_argument("--lambda_sigma", type=float, default=1.0)

    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=10)

    args = parser.parse_args()
    target_components = tuple(args.target_components.split(","))

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True)
    train_idx, val_idx, test_idx = train_val_test_split(len(dataset))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_pool)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pool)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_pool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FrequencyMACEWithABC(
        orig_atom_fea_len=dataset.atom_fea_len,
        nbr_fea_len=dataset.nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_layers=args.n_layers,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        n_frequencies=dataset.n_frequencies,
        n_output_channels=dataset.get_output_channels(),
        dropout=args.dropout,
        abc_hidden=args.abc_hidden,
    ).to(device)

    criterion = SpectrumAndPowerLawABCLoss(
        xi_array=dataset.frequency_array,
        alpha=args.alpha,
        lambda_eps=args.lambda_eps,
        lambda_sigma=args.lambda_sigma,
        eps_reduction="mae",
        sigma_reduction="mse",
        sigma_transform="log1p" if args.use_log else "none",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val = float("inf")
    train_losses = []
    val_losses = []
    model_path = f"models/{MODEL_NAME}.pt"

    print(f"Saving to {model_path}")
    print(f"alpha={args.alpha}, lambda_eps={args.lambda_eps}, lambda_sigma={args.lambda_sigma}, use_log={args.use_log}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr = train_epoch(model, train_loader, criterion, optimizer, device, use_log=args.use_log, grad_clip_norm=args.grad_clip_norm)
        va = validate(model, val_loader, criterion, device, use_log=args.use_log)
        scheduler.step(va)
        train_losses.append(tr)
        val_losses.append(va)
        print(f"Train Loss: {tr:.6f}, Val Loss: {va:.6f}")

        if va < best_val:
            best_val = va
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va,
                    "n_frequencies": dataset.n_frequencies,
                    "orig_atom_fea_len": dataset.atom_fea_len,
                    "nbr_fea_len": dataset.nbr_fea_len,
                    "n_output_channels": dataset.get_output_channels(),
                    "target_components": target_components,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "test_idx": test_idx,
                    "model_name": MODEL_NAME,
                    "use_log": args.use_log,
                    "atom_fea_len": args.atom_fea_len,
                    "n_layers": args.n_layers,
                    "h_fea_len": args.h_fea_len,
                    "n_h": args.n_h,
                    "dropout": args.dropout,
                    "abc_hidden": args.abc_hidden,
                    "alpha": args.alpha,
                    "lambda_eps": args.lambda_eps,
                    "lambda_sigma": args.lambda_sigma,
                },
                model_path,
            )
            print(f"  -> Saved best model (val_loss: {va:.6f})")

    plot_losses(train_losses, val_losses, save_path=f"models/{MODEL_NAME}_loss.png")

    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device, use_log=args.use_log)
    print(f"Test Loss: {test_loss:.6f}")
    dataset.close()


if __name__ == "__main__":
    main()

