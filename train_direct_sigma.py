"""
Train a fresh FrequencyMACEDirectSigma to predict per-point σ(ξ) directly.

The frozen backbone (FrequencyMACE) is used only to compute the per-batch target:
    target(ξ) = |ε_backbone(ξ) − ε_true(ξ)|   averaged over tensor components

The sigma loss is identical to the stage-2 (b,c) sigma loss so the two models
are directly comparable:
    loss = MSE( transform(σ_pred), transform(target) )
where transform = log1p (default), log, or none.

Metal low-frequency down-weighting is also supported (same flags as stage-2).

Usage:
    python3 train_direct_sigma.py --backbone models/mace_kkr_v1.pt
"""

import argparse
import os
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from lmdb_dataset import LMDBCrystalDataset, collate_pool
from mace_model import FrequencyMACE
from mace_model_direct_sigma import FrequencyMACEDirectSigma


def train_val_test_split(n, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    t = int(n * train_ratio)
    v = int(n * val_ratio)
    return idx[:t].tolist(), idx[t:t+v].tolist(), idx[t+v:].tolist()


def load_frozen_backbone(ckpt, device):
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


def sigma_loss(sigma_pred, eps_backbone, eps_true, is_metal,
               transform, n_low_freq, metal_low_freq_weight):
    """
    Exact same formula as SpectrumAndPowerLawABCLoss sigma term (stage-2).

    sigma_pred : (B, 1, F)
    eps_backbone: (B, C, F)  — already in ε-space (not log)
    eps_true    : (B, C, F)
    """
    abs_err = torch.abs(eps_backbone.detach() - eps_true)
    abs_err = abs_err.mean(dim=1, keepdim=True)   # (B, 1, F)

    if is_metal is not None and n_low_freq > 0:
        w = torch.ones_like(abs_err)
        w[is_metal.to(abs_err.device), :, :n_low_freq] = metal_low_freq_weight
        abs_err = abs_err * w
        sigma_pred = sigma_pred * w

    if transform == "log1p":
        abs_err = torch.log1p(abs_err)
        sigma_pred = torch.log1p(sigma_pred)
    elif transform == "log":
        abs_err = torch.log(abs_err + 1e-12)
        sigma_pred = torch.log(sigma_pred + 1e-12)

    return torch.mean((sigma_pred - abs_err) ** 2)


def run_epoch(sigma_model, backbone, loader, optimizer, device,
              transform, use_log, n_low_freq, metal_low_freq_weight,
              train, grad_clip, log_every=20):
    sigma_model.train() if train else sigma_model.eval()
    total, n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    tag = "train" if train else "val"

    with ctx:
        for batch_idx, (inputs, target, is_metal) in enumerate(loader):
            af, nf, nfi, cai = inputs
            af = af.to(device); nf = nf.to(device)
            nfi = nfi.to(device)
            cai = [x.to(device) for x in cai]
            target = target.to(device)

            with torch.no_grad():
                eps_raw = backbone(af, nf, nfi, cai)
                eps_pred = torch.exp(eps_raw) if use_log else eps_raw

            sigma_pred = sigma_model(af, nf, nfi, cai)   # (B, 1, F)

            loss = sigma_loss(sigma_pred, eps_pred, target, is_metal,
                              transform, n_low_freq, metal_low_freq_weight)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sigma_model.parameters(), grad_clip)
                optimizer.step()

            total += loss.item() * target.shape[0]
            n += target.shape[0]

            if train and (batch_idx + 1) % log_every == 0:
                print(f"  [{tag}] batch {batch_idx+1}/{len(loader)}  loss={loss.item():.6f}")

    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Train direct per-point σ(ξ) model")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Path to trained FrequencyMACE checkpoint (used only for loss target)")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Output name (default: <backbone_stem>_direct_sigma)")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--scheduler_patience", type=int, default=8)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)

    # sigma model architecture (independent of backbone — fresh model)
    parser.add_argument("--atom_fea_len", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--h_fea_len", type=int, default=256)
    parser.add_argument("--n_h", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--n_low_freq", type=int, default=0)
    parser.add_argument("--metal_low_freq_weight", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.backbone, map_location=device)
    use_log = ckpt.get("use_log", False)
    target_components = ckpt.get("target_components", ("xx",))
    metal_gap_threshold = ckpt.get("metal_gap_threshold", None)
    n_low_freq = ckpt.get("n_low_freq", 0)
    metal_low_freq_weight = ckpt.get("metal_low_freq_weight", 0.1)
    train_transform = "none"   # train with raw MSE — no compression
    eval_transform  = "log1p"  # eval with log1p for apples-to-apples vs (b,c) head

    stem = os.path.splitext(os.path.basename(args.backbone))[0]
    model_name = args.model_name or f"{stem}_direct_sigma"
    model_path = f"models/{model_name}.pt"

    print(f"Backbone   : {args.backbone}  (frozen, target only)")
    print(f"use_log    : {use_log}")
    print(f"train loss : raw MSE (no transform)")
    print(f"eval loss  : log1p MSE (for comparison with (b,c) head)")
    print(f"Saving to  : {model_path}")
    print(f"Device     : {device}")

    dataset = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=metal_gap_threshold,
    )

    train_idx = ckpt.get("train_idx")
    val_idx   = ckpt.get("val_idx")
    test_idx  = ckpt.get("test_idx")
    if train_idx is None:
        train_idx, val_idx, test_idx = train_val_test_split(len(dataset))
        print("  (split not in backbone checkpoint — regenerated with seed=42)")

    num_workers = min(8, multiprocessing.cpu_count())
    print(f"DataLoader workers: {num_workers}")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True,  num_workers=num_workers, collate_fn=collate_pool)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size,
                              shuffle=False, num_workers=num_workers, collate_fn=collate_pool)

    backbone = load_frozen_backbone(ckpt, device)

    sigma_model = FrequencyMACEDirectSigma(
        orig_atom_fea_len=dataset.atom_fea_len,
        nbr_fea_len=dataset.nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_layers=args.n_layers,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        n_frequencies=dataset.n_frequencies,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in sigma_model.parameters())
    print(f"Sigma model: {n_params:,} trainable params (fully fresh)")

    optimizer = torch.optim.Adam(sigma_model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr = run_epoch(sigma_model, backbone, train_loader, optimizer, device,
                       train_transform, use_log, n_low_freq, metal_low_freq_weight,
                       train=True, grad_clip=args.grad_clip)
        va = run_epoch(sigma_model, backbone, val_loader, optimizer, device,
                       train_transform, use_log, n_low_freq, metal_low_freq_weight,
                       train=False, grad_clip=args.grad_clip)
        va_log1p = run_epoch(sigma_model, backbone, val_loader, optimizer, device,
                             eval_transform, use_log, n_low_freq, metal_low_freq_weight,
                             train=False, grad_clip=args.grad_clip)
        scheduler.step(va)
        train_losses.append(tr)
        val_losses.append(va)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  train_mse={tr:.6f}  val_mse={va:.6f}  val_log1p={va_log1p:.6f}  lr={lr_now:.2e}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state_dict": sigma_model.state_dict(),
                "epoch": epoch,
                "val_loss": va,
                "model_name": model_name,
                "backbone_path": args.backbone,
                "loss_transform": train_transform,
                "use_log": use_log,
                "orig_atom_fea_len": dataset.atom_fea_len,
                "nbr_fea_len": dataset.nbr_fea_len,
                "n_frequencies": dataset.n_frequencies,
                "n_output_channels": 1,
                "atom_fea_len": args.atom_fea_len,
                "n_layers": args.n_layers,
                "h_fea_len": args.h_fea_len,
                "n_h": args.n_h,
                "dropout": args.dropout,
                "n_low_freq": n_low_freq,
                "metal_low_freq_weight": metal_low_freq_weight,
                "target_components": target_components,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }, model_path)
            print(f"  -> saved (val={va:.6f})")

    plt.figure(figsize=(9, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Sigma MSE (raw)")
    plt.title(f"{model_name}  —  direct σ(ξ) training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"models/{model_name}_loss.png", dpi=150)
    plt.close()

    dataset.close()
    print(f"\nDone. Best val = {best_val:.6f}  →  {model_path}")


if __name__ == "__main__":
    main()
