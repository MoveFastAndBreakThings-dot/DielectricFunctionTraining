"""
Loss functions for dielectric function prediction.
Single source of truth - import from here in all scripts.
"""

import torch
import torch.nn as nn


class FrequencyWeightedMSELoss(nn.Module):
    """
    MAE loss with optional low-frequency discounting for metals.

    For samples flagged as metals (is_metal=True), the first n_low_freq
    frequency points are multiplied by metal_low_freq_weight before averaging.
    This prevents the Drude divergence at small xi from dominating training.
    """
    
    def __init__(self, xi_array, decay_scale=None, n_low_freq=0, metal_low_freq_weight=0.1):
        super().__init__()
        xi = torch.tensor(xi_array, dtype=torch.float32)
        if decay_scale is None:
            decay_scale = float(torch.median(xi))
        weights = torch.exp(-xi / decay_scale)
        weights = weights * len(xi) / weights.sum()
        self.register_buffer('weights', weights)
        self.n_low_freq = n_low_freq
        self.metal_low_freq_weight = metal_low_freq_weight
    
    def forward(self, pred, target, is_metal=None):
        abs_error = torch.abs(pred - target)  # (B, C, F)
        if is_metal is not None and self.n_low_freq > 0:
            w = torch.ones(pred.shape[0], 1, pred.shape[2], device=pred.device)
            w[is_metal.to(pred.device), :, :self.n_low_freq] = self.metal_low_freq_weight
            abs_error = abs_error * w
        return abs_error.mean()
