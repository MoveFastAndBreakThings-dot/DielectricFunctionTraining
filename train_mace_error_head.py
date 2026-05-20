"""
Train a per-frequency error head on top of a frozen FrequencyMACE backbone.

The frozen backbone predicts ε(iξ). The new head predicts the per-frequency
absolute residual:

    err_pred(ξ) ≈ |ε_pred_frozen(ξ) − ε_true(ξ)|

The target is computed on-the-fly from the frozen model, so no extra labels
are needed. Only the error head (fc_err + err_pos) is trained.

Loss (matching SpectrumAndPowerLawABCLoss sigma term convention):
  transform(err_pred) vs transform(abs_err)
  where transform = log1p if --loss_transform log1p (default), else none.

Usage:
  python3 train_mace_error_head.py --backbone models/mace_kkr_v2.pt
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from mace_model import FrequencyMACE
from mace_model_error_head import FrequencyMACEWithErrorHead


def train_val_test_split(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return idx[:t].tolist(), idx[t:t+v].tolist(), idx[t+v:].tolist()


def load_frozen_backbone(ckpt, device):
    """Load a FrequencyMACE from checkpoint, fully frozen, eval mode."""
    model = FrequencyMACE(
        orig_atom_fea_len=ckpt["orig_atom_fea_len"],
        nbr_fea_len=ckpt["nbr_fea_len"],
        atom_fea_len=ckpt.get("atom_fea_len", 128),
        n_layers=ckpt.get("n_layers", 4),
        h_fea_len=ckpt.get("h_fea_len", 256),
        n_h=ckpt.get("n_h", 1),
        n_frequencies=ckpt["n_frequencies"],
        n_output_channels=ckpt["n_output_channels"],
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_epoch(error_model, eps_model, loader, optimizer, device, transform, use_log, train, log_every=20):
    error_model.train() if train else error_model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    tag = "train" if train else "val"

    with ctx:
        for batch_idx, (inputs, target, _) in enumerate(loader):
            af, nf, nfi, cai = inputs
            af = af.to(device); nf = nf.to(device)
            nfi = nfi.to(device)
            cai = [x.to(device) for x in cai]
            target = target.to(device)

            with torch.no_grad():
                eps_raw = eps_model(af, nf, nfi, cai)
                eps_pred = torch.exp(eps_raw) if use_log else eps_raw

            abs_err = torch.abs(eps_pred - target).mean(dim=1, keepdim=True)
            err_pred = error_model(af, nf, nfi, cai)

            if transform == "log1p":
                loss = nn.functional.mse_loss(torch.log1p(err_pred), torch.log1p(abs_err))
            elif transform == "log":
                loss = nn.functional.mse_loss(
                    torch.log(err_pred + 1e-12), torch.log(abs_err + 1e-12)
                )
            else:
                loss = nn.functional.mse_loss(err_pred, abs_err)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total += loss.item() * target.shape[0]
            n += target.shape[0]

            if (batch_idx + 1) % log_every == 0:
                print(f"  [{tag}] batch {batch_idx+1}/{len(loader)}  loss={loss.item():.6f}")

    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train per-frequency error head on frozen FrequencyMACE")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Path to trained FrequencyMACE checkpoint")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Output model name (default: <backbone_stem>_errhead)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--err_hidden", type=int, default=256,
                        help="Hidden dim of the error head MLP")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss_transform", type=str, default="log1p",
                        choices=("none", "log1p", "log"),
                        help="Transform applied to both err_pred and abs_err before MSE "
                             "(log1p matches SpectrumAndPowerLawABCLoss default)")
    parser.add_argument("--scheduler_patience", type=int, default=8)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load backbone checkpoint ----
    ckpt = torch.load(args.backbone, map_location=device)
    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))
    metal_gap_threshold = ckpt.get("metal_gap_threshold", None)

    stem = os.path.splitext(os.path.basename(args.backbone))[0]
    model_name = args.model_name or f"{stem}_errhead"
    model_path = f"models/{model_name}.pt"

    print(f"Backbone  : {args.backbone}")
    print(f"use_log   : {use_log}  (backbone output space)")
    print(f"transform : {args.loss_transform}")
    print(f"Saving to : {model_path}")
    print(f"Device: {device}")
    
    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=metal_gap_threshold,
    )

    # reuse backbone's exact split if saved
    train_idx = ckpt.get("train_idx")
    val_idx   = ckpt.get("val_idx")
    test_idx  = ckpt.get("test_idx")
    if train_idx is None:
        train_idx, val_idx, test_idx = train_val_test_split(len(dataset))
        print("  (split not in backbone checkpoint — regenerated with seed=42)")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, collate_fn=collate_pool)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, collate_fn=collate_pool)

    # ---- frozen FrequencyMACE — only used to compute eps_pred targets ----
    eps_model = load_frozen_backbone(ckpt, device)

    # ---- error head model (encoder weights copied from backbone, fc_freq excluded) ----
    error_model = FrequencyMACEWithErrorHead(
        orig_atom_fea_len=dataset.atom_fea_len,
        nbr_fea_len=dataset.nbr_fea_len,
        atom_fea_len=ckpt.get("atom_fea_len", 128),
        n_layers=ckpt.get("n_layers", 4),
        h_fea_len=ckpt.get("h_fea_len", 256),
        n_h=ckpt.get("n_h", 1),
        n_frequencies=dataset.n_frequencies,
        dropout=args.dropout,
        err_hidden=args.err_hidden,
    ).to(device)
    error_model.load_encoder_from_checkpoint(ckpt)

    n_trainable = sum(p.numel() for p in error_model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in error_model.parameters())
    print(f"Trainable : {n_trainable:,} / {n_total:,} params (fc_err + err_pos only)")

    trainable = [p for p in error_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr = run_epoch(error_model, eps_model, train_loader, optimizer, device,
                       args.loss_transform, use_log, train=True)
        va = run_epoch(error_model, eps_model, val_loader, optimizer, device,
                       args.loss_transform, use_log, train=False)
        scheduler.step(va)
        train_losses.append(tr)
        val_losses.append(va)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={tr:.6f}  val={va:.6f}  lr={lr_now:.2e}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state_dict": error_model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "model_name": model_name,
                "backbone_path": args.backbone,
                "loss_transform": args.loss_transform,
                "use_log": use_log,
                "orig_atom_fea_len": dataset.atom_fea_len,
                "nbr_fea_len": dataset.nbr_fea_len,
                "n_frequencies": dataset.n_frequencies,
                "n_output_channels": dataset.get_output_channels(),
                "atom_fea_len": ckpt.get("atom_fea_len", 128),
                "n_layers": ckpt.get("n_layers", 4),
                "h_fea_len": ckpt.get("h_fea_len", 256),
                "n_h": ckpt.get("n_h", 1),
                "dropout": args.dropout,
                "err_hidden": args.err_hidden,
                "target_components": target_components,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }, model_path)
            print(f"  -> saved (val={va:.6f})")

    plt.figure(figsize=(9, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel(f"MSE({args.loss_transform})")
    plt.title(f"{model_name}  — error-head training")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"models/{model_name}_loss.png", dpi=150)
    plt.close()

    dataset.close()
    print(f"\nDone. Best val = {best_val:.6f}  →  {model_path}")


if __name__ == "__main__":
    main()
