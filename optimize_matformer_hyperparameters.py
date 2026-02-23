"""
Hyperparameter optimization using Optuna for Matformer frequency prediction.

Usage:
    python optimize_matformer_hyperparameters.py --n_trials 50 --epochs 50
"""

import argparse
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from matformer_model import FrequencyMatformer
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses import FrequencyWeightedMSELoss
from train_matformer_frequency import train_epoch, validate, train_val_test_split


def objective(trial, dataset, device, epochs, use_log=True):
    """
    Objective function for Optuna optimization.
    Returns validation loss to minimize.
    """
    
    # Suggest hyperparameters for Matformer
    # Learning rate
    lr = trial.suggest_float('lr', 1e-6, 2e-3, log=True)
    
    # Architecture hyperparameters
    d_model = trial.suggest_categorical('d_model', [128, 256, 384])
    n_layers = trial.suggest_categorical('n_layers', [3, 4, 5, 6])
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    
    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        # Adjust d_model to be divisible by n_heads
        d_model = (d_model // n_heads) * n_heads
        if d_model < n_heads:
            d_model = n_heads * 4
    
    d_ff = trial.suggest_categorical('d_ff', [512, 1024, 2048])
    h_fea_len = trial.suggest_categorical('h_fea_len', [128, 256, 384])
    
    # Regularization
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3, 0.4])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3])
    
    # Fixed hyperparameters
    batch_size = 16
    n_h = 1
    scheduler_patience = 10
    scheduler_factor = 0.5
    grad_clip_norm = 1.0
    
    print(f"\nTrial {trial.number}:")
    print(f"  lr={lr:.6f}, d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}")
    print(f"  d_ff={d_ff}, h_fea_len={h_fea_len}, dropout={dropout}, weight_decay={weight_decay}")
    
    # Split dataset (same seed for consistency)
    train_idx, val_idx, _ = train_val_test_split(len(dataset), seed=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate_pool)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_pool)
    
    # Get model parameters
    orig_atom_fea_len = dataset.atom_fea_len
    nbr_fea_len = dataset.nbr_fea_len
    n_frequencies = dataset.n_frequencies
    n_output_channels = dataset.get_output_channels()
    
    # Initialize model
    model = FrequencyMatformer(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        h_fea_len=h_fea_len,
        n_h=n_h,
        n_frequencies=n_frequencies,
        n_output_channels=n_output_channels,
        dropout=dropout
    ).to(device)
    
    # Training setup
    criterion = FrequencyWeightedMSELoss(dataset.frequency_array).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
    )
    
    # Suppress verbose output during optimization
    train_epoch._suppress_output = True
    
    # Training loop with overfitting detection
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20  # Early stopping patience
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                use_log=use_log, grad_clip_norm=grad_clip_norm)
        val_loss = validate(model, val_loader, criterion, device, use_log=use_log)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break
        
        if epoch % 10 == 0:
            overfitting_ratio = val_loss / (train_loss + 1e-8) if train_loss > 0 else 1.0
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, Ratio={overfitting_ratio:.2f}")
    
    # Clean up
    delattr(train_epoch, '_suppress_output')
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna for Matformer')
    parser.add_argument('--lmdb_path', type=str, default='processed_data/crystal_graphs.lmdb',
                        help='Path to LMDB database')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per trial')
    parser.add_argument('--use_log', action='store_true', default=True,
                        help='Train in log-space')
    parser.add_argument('--study_name', type=str, default='matformer_hyperopt',
                        help='Name for Optuna study (for resuming)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Storage URL for Optuna study (e.g., sqlite:///study.db)')
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.lmdb_path}...")
    dataset = LMDBCrystalDataset(args.lmdb_path, target_components=('xx',), use_kkr=True)
    print(f"Total samples: {len(dataset)}")
    print(f"Log-space training: {args.use_log}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create or load study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            load_if_exists=True,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    else:
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
    
    # Enqueue reasonable defaults as first trial
    current_defaults = {
        'lr': 0.0001,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 1024,
        'h_fea_len': 256,
        'dropout': 0.1,
        'weight_decay': 1e-5,
    }
    study.enqueue_trial(current_defaults)
    print(f"Enqueued default hyperparameters as first trial")
    
    # Optimize
    print(f"\nStarting optimization with {args.n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, dataset, device, args.epochs, args.use_log),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters
    import json
    best_params = {
        'best_val_loss': study.best_value,
        'best_trial': study.best_trial.number,
        'hyperparameters': study.best_params
    }
    with open('best_matformer_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters saved to: best_matformer_hyperparameters.json")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image('matformer_hyperopt_history.png')
        print("Optimization history plot saved to: matformer_hyperopt_history.png")
    except:
        print("Could not generate optimization history plot (install plotly)")
    
    dataset.close()


if __name__ == '__main__':
    main()

