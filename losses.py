"""
Loss functions for dielectric function prediction.
Single source of truth - import from here in all scripts.
"""

import torch
import torch.nn as nn


class FrequencyWeightedMSELoss(nn.Module):
    """
    Weighted MSE loss: Loss = mean( exp(-xi/A) * (pred - true)^2 )
    
    Lower frequencies (small xi) get weight ~1, higher frequencies decay exponentially.
    This allows less accuracy at high frequencies where it matters less.
    """
    
    def __init__(self, xi_array, decay_scale=None):
        super().__init__()
        xi = torch.tensor(xi_array, dtype=torch.float32)
        if decay_scale is None:
            decay_scale = float(torch.median(xi))
        weights = torch.exp(-xi / decay_scale)
        weights = weights * len(xi) / weights.sum()
        self.register_buffer('weights', weights)
    
    def forward(self, pred, target):
        abs_error = torch.abs(pred - target)
        return abs_error.mean()
