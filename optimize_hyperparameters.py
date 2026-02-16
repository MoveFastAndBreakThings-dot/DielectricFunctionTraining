"""
Hyperparameter optimization using Optuna for CGCNN frequency prediction.

Usage:
    python optimize_hyperparameters.py --n_trials 50 --epochs 50
"""

import argparse
import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from cgcnn_model import FrequencyCGCNN
from lmdb_dataset import LMDBCrystalDataset, collate_pool
from losses import FrequencyWeightedMSELoss
from train_cgcnn_frequency import train_epoch, validate, train_val_test_split


def objective(trial, dataset, device, epochs, use_log=True):
    """
    Objective function for Optuna optimization.
    Returns validation loss to minimize.
    """
    
    # Suggest hyperparameters
    # High priority
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    atom_fea_len = trial.suggest_categorical('atom_fea_len', [64, 128])
    h_fea_len = trial.suggest_categorical('h_fea_len', [128, 256])
    n_conv = trial.suggest_categorical('n_conv', [3, 4])
    dropout = trial.suggest_categorical('dropout', [0.1, 0.2])
    
    # Medium priority
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4])
    n_h = trial.suggest_categorical('n_h', [1, 2])
    
    # Scheduler and training
    scheduler_patience = trial.suggest_categorical('scheduler_patience', [5, 10, 15])
    scheduler_factor = trial.suggest_categorical('scheduler_factor', [0.3, 0.5, 0.7])
    grad_clip_norm = trial.suggest_categorical('grad_clip_norm', [0.5, 1.0, 2.0])
    
    print(f"\nTrial {trial.number}:")
    print(f"  lr={lr:.6f}, atom_fea_len={atom_fea_len}, h_fea_len={h_fea_len}, n_conv={n_conv}")
    print(f"  dropout={dropout}, batch_size={batch_size}, weight_decay={weight_decay}")
    print(f"  n_h={n_h}, scheduler_patience={scheduler_patience}, scheduler_factor={scheduler_factor}")
    print(f"  grad_clip_norm={grad_clip_norm}")
    
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
    model = FrequencyCGCNN(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=atom_fea_len,
        n_conv=n_conv,
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
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20  # Early stopping patience
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                use_log=use_log, grad_clip_norm=grad_clip_norm)
        val_loss = validate(model, val_loader, criterion, device, use_log=use_log)
        scheduler.step(val_loss)
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                break
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
    
    # Clean up
    delattr(train_epoch, '_suppress_output')
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization with Optuna')
    parser.add_argument('--lmdb_path', type=str, default='processed_data/crystal_graphs.lmdb',
                        help='Path to LMDB database')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per trial')
    parser.add_argument('--use_log', action='store_true', default=True,
                        help='Train in log-space')
    parser.add_argument('--study_name', type=str, default='cgcnn_hyperopt',
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
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest hyperparameters saved to: best_hyperparameters.json")
    
    # Plot optimization history
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image('hyperopt_history.png')
        print("Optimization history plot saved to: hyperopt_history.png")
    except:
        print("Could not generate optimization history plot (install plotly)")
    
    dataset.close()


if __name__ == '__main__':
    main()

