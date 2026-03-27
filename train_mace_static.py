"""
Train MACE model to predict only the static dielectric constant (first ξ point).
Same architecture as mace_kkr_v2, but output is a single value per crystal.
Saves model and then reports actual vs predicted for the first frequency point.
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, Dataset
from mace_model import FrequencyMACE
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses import FrequencyWeightedMSELoss


MODEL_NAME = "mace_kkr_static"


class StaticOnlyWrapper(Dataset):
    """Wraps LMDBCrystalDataset to return only the first frequency point (static ε)."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        inputs, target, _ = self.base[idx]
        # target shape (n_components, n_xi) -> take first freq only
        target_static = target[:, 0:1].copy()
        return inputs, target_static, idx

    @property
    def n_frequencies(self):
        return 1

    def get_output_channels(self):
        return self.base.get_output_channels()

    def get_sample_info(self, idx):
        return self.base.get_sample_info(idx)

    @property
    def frequency_array(self):
        return self.base.kkr_xi_array[0:1]

    def close(self):
        self.base.close()


def train_epoch(model, dataloader, criterion, optimizer, device, use_log=False, grad_clip_norm=1.0):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch_idx, (inputs, target, _) in enumerate(dataloader):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        target = target.to(device)
        if use_log:
            target = torch.log(target)
        optimizer.zero_grad()
        output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * target.shape[0]
        n_samples += target.shape[0]
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
    return total_loss / n_samples


def validate(model, dataloader, criterion, device, use_log=False):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for inputs, target, _ in dataloader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            target = target.to(device)
            if use_log:
                target = torch.log(target)
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(output, target)
            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]
    return total_loss / n_samples


def train_val_test_split(n_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size:train_size + val_size].tolist()
    test_idx = indices[train_size + val_size:].tolist()
    return train_idx, val_idx, test_idx


def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('MACE static: Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    plt.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def evaluate_static_actual_vs_predicted(model, dataset, test_idx, device, use_log, model_path):
    """Compute and print actual vs predicted for first frequency point on test set."""
    model.eval()
    actuals = []
    preds = []
    for idx in test_idx:
        inputs, tgt, _ = collate_pool([dataset[idx]])
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        with torch.no_grad():
            out = model(
                atom_fea.to(device),
                nbr_fea.to(device),
                nbr_fea_idx.to(device),
                crystal_atom_idx
            )
        if use_log:
            out = torch.exp(out)
        a = tgt[0, 0, 0].item()
        p = out[0, 0, 0].cpu().item()
        actuals.append(a)
        preds.append(p)
    actuals = np.array(actuals)
    preds = np.array(preds)
    errors = preds - actuals
    rel_errors = np.abs(errors) / (np.abs(actuals) + 1e-10)

    print("\n" + "=" * 70)
    print("STATIC (FIRST FREQUENCY POINT) — ACTUAL VS PREDICTED")
    print("=" * 70)
    print(f"Test samples: {len(actuals)}")
    print(f"ξ at first point: {dataset.base.kkr_xi_array[0]:.6f} eV")
    print(f"\nActual  — mean: {np.mean(actuals):.4f}, median: {np.median(actuals):.4f}, std: {np.std(actuals):.4f}")
    print(f"Pred    — mean: {np.mean(preds):.4f}, median: {np.median(preds):.4f}, std: {np.std(preds):.4f}")
    print(f"\nMSE:     {np.mean(errors**2):.6f}")
    print(f"RMSE:    {np.sqrt(np.mean(errors**2)):.6f}")
    print(f"MAE:     {np.mean(np.abs(errors)):.6f}")
    print(f"Mean relative error: {np.mean(rel_errors)*100:.2f}%")
    print(f"Median relative error: {np.median(rel_errors)*100:.2f}%")
    print("=" * 70)

    # Plot actual vs predicted
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(actuals, preds, alpha=0.5, s=20)
    lims = [min(actuals.min(), preds.min()), max(actuals.max(), preds.max())]
    ax.plot(lims, lims, 'k--', label='y=x')
    ax.set_xlabel('Actual ε(0)', fontsize=12)
    ax.set_ylabel('Predicted ε(0)', fontsize=12)
    ax.set_title('Static dielectric: actual vs predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plot_path = model_path.replace('.pt', '_actual_vs_pred.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Actual vs predicted plot saved to: {plot_path}")
    return actuals, preds


def main():
    parser = argparse.ArgumentParser(description='Train MACE on static dielectric only (first ξ point)')
    parser.add_argument('--lmdb_path', type=str, default='processed_data/crystal_graphs.lmdb')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--target_components', type=str, default='xx')
    parser.add_argument('--use_log', action='store_true', help='Train in log-space')
    parser.add_argument('--atom_fea_len', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--h_fea_len', type=int, default=256)
    parser.add_argument('--n_h', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    args = parser.parse_args()

    target_components = tuple(args.target_components.split(','))
    print("Loading dataset (full KKR)...")
    base_ds = LMDBCrystalDataset(
        args.lmdb_path,
        target_components=target_components,
        use_kkr=True
    )
    dataset = StaticOnlyWrapper(base_ds)
    n_frequencies = 1
    n_output_channels = dataset.get_output_channels()

    print(f"Total samples: {len(dataset)}, static only (n_frequencies=1), output channels: {n_output_channels}")
    print(f"Log-space: {args.use_log}")

    train_idx, val_idx, test_idx = train_val_test_split(len(dataset))
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_pool)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_pool)

    orig_atom_fea_len = base_ds.atom_fea_len
    nbr_fea_len = base_ds.nbr_fea_len
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = FrequencyMACE(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_layers=args.n_layers,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        n_frequencies=n_frequencies,
        n_output_channels=n_output_channels,
        dropout=args.dropout
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    xi_first = np.array([base_ds.kkr_xi_array[0]])
    criterion = FrequencyWeightedMSELoss(xi_first).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    model_path = f"models/{MODEL_NAME}.pt"
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print(f"\nSaving best model to: {model_path}")
    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                 use_log=args.use_log, grad_clip_norm=args.grad_clip_norm)
        val_loss = validate(model, val_loader, criterion, device, use_log=args.use_log)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'n_frequencies': n_frequencies,
                'orig_atom_fea_len': orig_atom_fea_len,
                'nbr_fea_len': nbr_fea_len,
                'n_output_channels': n_output_channels,
                'target_components': target_components,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx,
                'model_name': MODEL_NAME,
                'use_log': args.use_log,
                'atom_fea_len': args.atom_fea_len,
                'n_layers': args.n_layers,
                'h_fea_len': args.h_fea_len,
                'n_h': args.n_h,
                'dropout': args.dropout,
                'weight_decay': args.weight_decay,
                'grad_clip_norm': args.grad_clip_norm,
                'scheduler_factor': args.scheduler_factor,
                'scheduler_patience': args.scheduler_patience,
            }, model_path)
            print(f"  -> Saved best model (val_loss: {val_loss:.6f})")

    plot_losses(train_losses, val_losses, save_path=f"models/{MODEL_NAME}_loss.png")

    print("\nLoading best checkpoint for evaluation...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=0, collate_fn=collate_pool),
                       criterion, device, use_log=args.use_log)
    print(f"Test Loss: {test_loss:.6f}")

    evaluate_static_actual_vs_predicted(
        model, dataset, test_idx, device, args.use_log, model_path
    )
    dataset.close()
    print("Done.")


if __name__ == '__main__':
    main()
