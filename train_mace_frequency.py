"""
Train MACE model to predict KKR-transformed dielectric function ε(iξ).

Usage:
    1. First run: python preprocess_dataset.py --input_dir data --output_dir processed_data
    2. Then run: python train_mace_frequency.py

Log-space training:
    Use --use_log to train in log-space. This predicts log(ε) and transforms
    back to real values via exp() for evaluation. Helps with high dynamic range.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from mace_model import FrequencyMACE
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses import FrequencyWeightedMSELoss

# ============================================================================
# CHANGE MODEL NAME HERE
# ============================================================================
MODEL_NAME = "mace_kkr_v1"
# ============================================================================


def train_epoch(model, dataloader, criterion, optimizer, device, use_log=False, grad_clip_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for batch_idx, (inputs, target, is_metal) in enumerate(dataloader):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        target = target.to(device)
        
        # Transform target to log-space if enabled
        if use_log:
            target = torch.log(target)
        
        optimizer.zero_grad()
        output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        loss = criterion(output, target, is_metal)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        
        total_loss += loss.item() * target.shape[0]
        n_samples += target.shape[0]
        
        if batch_idx % 10 == 0 and not hasattr(train_epoch, '_suppress_output'):
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
    
    return total_loss / n_samples


def validate(model, dataloader, criterion, device, use_log=False):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    with torch.no_grad():
        for inputs, target, is_metal in dataloader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            target = target.to(device)
            
            if use_log:
                target = torch.log(target)
            
            output = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss = criterion(output, target, is_metal)
            
            total_loss += loss.item() * target.shape[0]
            n_samples += target.shape[0]
    
    return total_loss / n_samples


def plot_losses(train_losses, val_losses, save_path='training_loss.png'):
    """Plot train and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Mark best validation loss
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best: epoch {best_epoch}')
    plt.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to: {save_path}")


def train_val_test_split(n_samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Split indices into train/val/test sets."""
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size:train_size + val_size].tolist()
    test_idx = indices[train_size + val_size:].tolist()
    
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description='Train FrequencyMACE on KKR data')
    parser.add_argument('--lmdb_path', type=str, default='processed_data/crystal_graphs.lmdb',
                        help='Path to LMDB database')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--target_components', type=str, default='xx',
                        help='Dielectric components to predict (comma-separated: xx,yy,zz)')
    parser.add_argument('--use_log', action='store_true',
                        help='Train in log-space (predict log(ε), better for high dynamic range)')
    parser.add_argument('--n_low_freq', type=int, default=0,
                        help='Number of low-frequency points to discount for metals (0 = disabled)')
    parser.add_argument('--metal_low_freq_weight', type=float, default=0.1,
                        help='Loss weight for discounted low-freq points in metals')
    parser.add_argument('--metal_gap_threshold', type=float, default=0.1,
                        help='Band gap threshold (eV) below which a material is treated as metal')
    
    # Model architecture hyperparameters
    parser.add_argument('--atom_fea_len', type=int, default=128,
                        help='Number of hidden atom features')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of message passing layers')
    parser.add_argument('--h_fea_len', type=int, default=256,
                        help='Hidden dimension after pooling')
    parser.add_argument('--n_h', type=int, default=1,
                        help='Number of hidden layers after pooling')
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training hyperparameters
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    
    # Scheduler hyperparameters
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_patience', type=int, default=10)
    
    args = parser.parse_args()
    
    target_components = tuple(args.target_components.split(','))
    
    # Load LMDB dataset with KKR targets
    print(f"Loading dataset from {args.lmdb_path}...")
    dataset = LMDBCrystalDataset(
        args.lmdb_path, 
        target_components=target_components,
        use_kkr=True,
        metal_gap_threshold=args.metal_gap_threshold,
    )
    
    print(f"Total samples: {len(dataset)}")
    print(f"ξ points: {dataset.n_frequencies}")
    print(f"Target components: {target_components}")
    print(f"Output channels: {dataset.get_output_channels()}")
    print(f"Log-space training: {args.use_log}")
    
    # Split into train/val/test
    train_idx, val_idx, test_idx = train_val_test_split(len(dataset))
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=0, collate_fn=collate_pool)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=0, collate_fn=collate_pool)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             num_workers=0, collate_fn=collate_pool)
    
    # Get model parameters from dataset
    orig_atom_fea_len = dataset.atom_fea_len
    nbr_fea_len = dataset.nbr_fea_len
    n_frequencies = dataset.n_frequencies
    n_output_channels = dataset.get_output_channels()
    
    print(f"Atom features: {orig_atom_fea_len}, Neighbor features: {nbr_fea_len}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Training setup - frequency-weighted loss
    criterion = FrequencyWeightedMSELoss(
        dataset.frequency_array,
        n_low_freq=args.n_low_freq,
        metal_low_freq_weight=args.metal_low_freq_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    model_path = f"models/{MODEL_NAME}.pt"
    
    print(f"\nModel will be saved to: {model_path}")
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
                'n_low_freq': args.n_low_freq,
                'metal_low_freq_weight': args.metal_low_freq_weight,
                'metal_gap_threshold': args.metal_gap_threshold,
                # Hyperparameters
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
    
    # Plot losses
    plot_losses(train_losses, val_losses, save_path=f"models/{MODEL_NAME}_loss.png")
    
    # Final evaluation
    print("\nEvaluating on test set...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, test_loader, criterion, device, use_log=args.use_log)
    print(f"Final Test Loss (log-space): {test_loss:.6f}" if args.use_log else f"Final Test Loss: {test_loss:.6f}")
    
    dataset.close()


if __name__ == '__main__':
    main()

