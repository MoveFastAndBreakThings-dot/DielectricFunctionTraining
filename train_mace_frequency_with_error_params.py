"""
Train a MACE model to predict ε(iξ) and per-crystal a=0 power-law error params (b,c).

  σ(ξ; b, c) = b / (ξ + c)^α   (α fixed, b,c > 0 via softplus head)

Targets:
  - ε(iξ) from LMDB (KKR targets)
Supervision for (b,c):
  - match σ(ξ; b,c) to |ε_pred - ε_true| within each batch (correlated across ξ).

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

# kept here so both scripts can import it
def train_val_test_split_fn(n_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    return (indices[:train_size].tolist(),
            indices[train_size:train_size + val_size].tolist(),
            indices[train_size + val_size:].tolist())


def train_epoch(model, dataloader, criterion, optimizer, device, use_log=False, grad_clip_norm=1.0):
    model.train()
    total = 0.0
    n = 0
    last_parts = None
    for batch_idx, (inputs, target, is_metal) in enumerate(dataloader):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [x.to(device) for x in crystal_atom_idx]
        target = target.to(device)

        optimizer.zero_grad()
        eps_out, abc_out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

        criterion.eps_pred_is_log = bool(use_log)
        loss, parts = criterion(eps_out, abc_out, target, is_metal)

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
    for inputs, target, is_metal in dataloader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [x.to(device) for x in crystal_atom_idx]
        target = target.to(device)

        eps_out, abc_out = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        criterion.eps_pred_is_log = bool(use_log)
        loss, _ = criterion(eps_out, abc_out, target, is_metal)
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
    return train_val_test_split_fn(n_samples, train_ratio, val_ratio, seed)


def main():
    parser = argparse.ArgumentParser(description="Train FrequencyMACEWithABC on KKR ε(iξ)")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target_components", type=str, default="xx")
    parser.add_argument("--use_log", action="store_true", help="Model outputs log(ε); loss uses exp(output).")
    parser.add_argument("--n_low_freq", type=int, default=0,
                        help="Number of low-frequency points to discount for metals (0 = disabled)")
    parser.add_argument("--metal_low_freq_weight", type=float, default=0.1,
                        help="Loss weight for discounted low-freq points in metals")
    parser.add_argument("--metal_gap_threshold", type=float, default=0.1,
                        help="Band gap threshold (eV) below which a material is treated as metal")

    parser.add_argument("--atom_fea_len", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--h_fea_len", type=int, default=256)
    parser.add_argument("--n_h", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--abc_hidden", type=int, default=128)

    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--alpha", type=float, default=3.0, help="Fixed α in σ(ξ)=b/(ξ+c)^α")
    parser.add_argument("--lambda_eps", type=float, default=1.0)
    parser.add_argument("--lambda_sigma", type=float, default=1.0)

    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--scheduler_patience", type=int, default=10)

    args = parser.parse_args()
    target_components = tuple(args.target_components.split(","))

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True,
                                  metal_gap_threshold=args.metal_gap_threshold)
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
        n_low_freq=args.n_low_freq,
        metal_low_freq_weight=args.metal_low_freq_weight,
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
                    "n_low_freq": args.n_low_freq,
                    "metal_low_freq_weight": args.metal_low_freq_weight,
                    "metal_gap_threshold": args.metal_gap_threshold,
                    "atom_fea_len": args.atom_fea_len,
                    "n_layers": args.n_layers,
                    "h_fea_len": args.h_fea_len,
                    "n_h": args.n_h,
                    "dropout": args.dropout,
                    "abc_hidden": args.abc_hidden,
                    "alpha": args.alpha,
                    "lambda_eps": args.lambda_eps,
                    "lambda_sigma": args.lambda_sigma,
                    "sigma_transform": "log1p" if args.use_log else "none",
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
    import sys
    if "--stage2" in sys.argv:
        main_stage2()
    else:
        main()


# ---------------------------------------------------------------------------
# Stage-2 entry point  (kept in this file for import convenience)
# Run via:  python3 train_mace_frequency_with_error_params.py --stage2 ...
# Or:       python3 train_bc_head_stage2.py ...  (thin wrapper that imports this)
# ---------------------------------------------------------------------------

def main_stage2():
    """
    Freeze the entire backbone + eps head of a pretrained FrequencyMACEWithABC,
    then fine-tune only the (b,c) head (fc_bc + bc_pos) with pure sigma loss.

    The key insight: with the eps head frozen, the self-supervised target
    |eps_pred - eps_true| is stable across batches, making it much easier
    for the (b,c) head to learn crystal-specific error magnitudes.

    Usage:
        python3 train_mace_frequency_with_error_params.py --stage2 \\
            --checkpoint models/mace_kkr_with_abc_v3.pt \\
            --model_name mace_kkr_with_abc_v3_stage2 \\
            --epochs 50 --lr 1e-3
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to stage-1 checkpoint to load")
    parser.add_argument("--lmdb_path", type=str, default="processed_data/crystal_graphs.lmdb")
    parser.add_argument("--model_name", type=str, default="mace_kkr_with_abc_stage2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="LR for (b,c) head only — can be higher than stage-1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lambda_sigma", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_patience", type=int, default=8)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    args = parser.parse_args()

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from lmdb_dataset import LMDBCrystalDataset, collate_pool
    from mace_model_error_params import FrequencyMACEWithABC
    from losses_mace_error_params import SpectrumAndPowerLawABCLoss

    os.makedirs("models", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load stage-1 checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    use_log = ckpt.get("use_log", False)
    alpha = ckpt.get("alpha", 3.0)
    sigma_transform = ckpt.get("sigma_transform", "log1p" if use_log else "none")
    target_components = ckpt.get("target_components", ("xx",))
    n_low_freq = ckpt.get("n_low_freq", 0)
    metal_low_freq_weight = ckpt.get("metal_low_freq_weight", 0.1)
    metal_gap_threshold = ckpt.get("metal_gap_threshold", 0.1)
    print(f"Loaded: {args.checkpoint}")
    print(f"  use_log={use_log}  alpha={alpha}  sigma_transform={sigma_transform}")
    print(f"  n_low_freq={n_low_freq}  metal_low_freq_weight={metal_low_freq_weight}")

    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=target_components, use_kkr=True,
                                  metal_gap_threshold=metal_gap_threshold)

    # reuse exact same split as stage 1 so test set is untouched
    train_idx = ckpt.get("train_idx")
    val_idx = ckpt.get("val_idx")
    test_idx = ckpt.get("test_idx")
    if train_idx is None:
        train_idx, val_idx, test_idx = train_val_test_split_fn(len(dataset))
        print("  (split not in checkpoint, regenerated with seed=42)")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=0, collate_fn=collate_pool)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate_pool)

    # ---- build model and load weights ----
    model = FrequencyMACEWithABC(
        orig_atom_fea_len=ckpt["orig_atom_fea_len"],
        nbr_fea_len=ckpt["nbr_fea_len"],
        atom_fea_len=ckpt.get("atom_fea_len", 128),
        n_layers=ckpt.get("n_layers", 4),
        h_fea_len=ckpt.get("h_fea_len", 256),
        n_h=ckpt.get("n_h", 1),
        n_frequencies=ckpt["n_frequencies"],
        n_output_channels=ckpt["n_output_channels"],
        dropout=ckpt.get("dropout", 0.1),
        abc_hidden=ckpt.get("abc_hidden", 128),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # ---- freeze everything except fc_bc and bc_pos ----
    frozen_modules = [model.embedding, model.layers, model.pool_to_fc,
                      model.pool_activation, model.fc_freq]
    if model.fcs is not None:
        frozen_modules.append(model.fcs)

    for mod in frozen_modules:
        for p in mod.parameters():
            p.requires_grad = False

    trainable_params = list(model.fc_bc.parameters()) + list(model.bc_pos.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nFrozen backbone. Trainable: {n_trainable:,} / {n_total:,} parameters "
          f"(fc_bc + bc_pos only)")

    # ---- loss: pure sigma, lambda_eps=0 ----
    criterion = SpectrumAndPowerLawABCLoss(
        xi_array=dataset.frequency_array,
        alpha=alpha,
        lambda_eps=0.0,
        lambda_sigma=args.lambda_sigma,
        eps_reduction="mae",
        sigma_reduction="mse",
        sigma_transform=sigma_transform,
        n_low_freq=n_low_freq,
        metal_low_freq_weight=metal_low_freq_weight,
    ).to(device)
    criterion.eps_pred_is_log = use_log

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor,
        patience=args.scheduler_patience)

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total = 0.0
        n = 0
        last_sigma = None
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_idx, (inputs, target, is_metal) in enumerate(loader):
                af, nf, nfi, cai = inputs
                af = af.to(device); nf = nf.to(device)
                nfi = nfi.to(device)
                cai = [x.to(device) for x in cai]
                target = target.to(device)

                if train:
                    optimizer.zero_grad()

                eps_out, bc_out = model(af, nf, nfi, cai)
                loss, parts = criterion(eps_out, bc_out, target, is_metal)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)
                    optimizer.step()

                total += loss.item() * target.shape[0]
                n += target.shape[0]
                last_sigma = parts["loss_sigma"]

                if train and batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(loader)}, "
                          f"sigma_loss={parts['loss_sigma']:.4f}")
        return total / max(n, 1), last_sigma

    model_path = f"models/{args.model_name}.pt"
    best_val = float("inf")
    train_losses, val_losses = [], []

    print(f"\nStage-2 training  ({args.epochs} epochs, lr={args.lr}, "
          f"lambda_sigma={args.lambda_sigma})")
    print(f"Saving to {model_path}\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        tr, _ = run_epoch(train_loader, train=True)
        va, _ = run_epoch(val_loader, train=False)
        scheduler.step(va)
        train_losses.append(tr)
        val_losses.append(va)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Train sigma_loss={tr:.6f}  Val sigma_loss={va:.6f}  lr={lr_now:.2e}")

        if va < best_val:
            best_val = va
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": va,
                "stage": 2,
                "stage1_checkpoint": args.checkpoint,
                "n_frequencies": ckpt["n_frequencies"],
                "orig_atom_fea_len": ckpt["orig_atom_fea_len"],
                "nbr_fea_len": ckpt["nbr_fea_len"],
                "n_output_channels": ckpt["n_output_channels"],
                "target_components": target_components,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "model_name": args.model_name,
                "use_log": use_log,
                "atom_fea_len": ckpt.get("atom_fea_len", 128),
                "n_layers": ckpt.get("n_layers", 4),
                "h_fea_len": ckpt.get("h_fea_len", 256),
                "n_h": ckpt.get("n_h", 1),
                "dropout": ckpt.get("dropout", 0.1),
                "abc_hidden": ckpt.get("abc_hidden", 128),
                "alpha": alpha,
                "lambda_eps": 0.0,
                "lambda_sigma": args.lambda_sigma,
                "sigma_transform": sigma_transform,
            }, model_path)
            print(f"  -> Saved best (val_loss={va:.6f})")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Sigma loss"); plt.legend()
    plt.title(f"Stage-2: (b,c) head only  —  {args.model_name}")
    plt.tight_layout()
    plt.savefig(f"models/{args.model_name}_loss.png", dpi=150)
    plt.close()

    dataset.close()
    print(f"\nDone. Best val sigma_loss = {best_val:.6f}")
    print(f"Checkpoint: {model_path}")

