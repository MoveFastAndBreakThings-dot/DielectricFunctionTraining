"""
Train CGCNN model to predict KKR-transformed dielectric function ε(iξ).

Usage:
    1. First run: python preprocess_dataset.py --input_dir data --output_dir processed_data
    2. Then run: python train_cgcnn_frequency.py

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
from cgcnn_model import FrequencyCGCNN
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses import FrequencyWeightedMSELoss

# ============================================================================
# CHANGE MODEL NAME HERE
# ============================================================================
MODEL_NAME = "cgcnn_kkr_v2"
# ============================================================================


def train_epoch(model, dataloader, criterion, optimizer, device, use_log=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for batch_idx, (inputs, target, _) in enumerate(dataloader):
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
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * target.shape[0]
        n_samples += target.shape[0]
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')
    
    return total_loss / n_samples


def validate(model, dataloader, criterion, device, use_log=False):
    """Validate the model."""
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
    parser = argparse.ArgumentParser(description='Train FrequencyCGCNN on KKR data')
    parser.add_argument('--lmdb_path', type=str, default='processed_data/crystal_graphs.lmdb',
                        help='Path to LMDB database')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--target_components', type=str, default='xx',
                        help='Dielectric components to predict (comma-separated: xx,yy,zz)')
    parser.add_argument('--use_log', action='store_true',
                        help='Train in log-space (predict log(ε), better for high dynamic range)')
    args = parser.parse_args()
    
    target_components = tuple(args.target_components.split(','))
    
    # Load LMDB dataset with KKR targets
    print(f"Loading dataset from {args.lmdb_path}...")
    dataset = LMDBCrystalDataset(
        args.lmdb_path, 
        target_components=target_components,
        use_kkr=True
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
    
    model = FrequencyCGCNN(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        n_frequencies=n_frequencies,
        n_output_channels=n_output_channels
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup - frequency-weighted loss (low freq weighted more than high freq)
    criterion = FrequencyWeightedMSELoss(dataset.frequency_array).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_log=args.use_log)
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
                'use_log': args.use_log,  # Important: save this for inference
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
