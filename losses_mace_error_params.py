"""
Loss for joint prediction of ε(iξ) and a=0 power-law error parameters (b,c).

We do not assume labeled (b,c). Instead we supervise (b,c) by matching the
predicted uncertainty curve σ(ξ) to the observed absolute residual |ε_pred-ε_true|
within each batch.

Power-law uncertainty curve (same convention as fit_error_models.model_power_law_a0):
  σ(ξ) = b / (ξ + c)^α    [a=0 so σ→0 as ξ→∞]

α is fixed (set in training script), while b,c are predicted per crystal.
The model head outputs 2 values (b, c); abc_pred below refers to (b, c).
"""

import torch
import torch.nn as nn


class SpectrumAndPowerLawABCLoss(nn.Module):
    def __init__(
        self,
        xi_array,
        alpha=3.0,
        lambda_eps=1.0,
        lambda_sigma=1.0,
        eps_reduction="mae",
        sigma_reduction="mse",
        sigma_transform="none",
        eps_min_c=1e-6,
        n_low_freq=0,
        metal_low_freq_weight=0.1,
    ):
        super().__init__()
        xi = torch.tensor(xi_array, dtype=torch.float32)
        self.register_buffer("xi", xi)
        self.alpha = float(alpha)
        self.lambda_eps = float(lambda_eps)
        self.lambda_sigma = float(lambda_sigma)
        self.eps_reduction = eps_reduction
        self.sigma_reduction = sigma_reduction
        self.sigma_transform = sigma_transform
        self.eps_min_c = float(eps_min_c)
        self.n_low_freq = n_low_freq
        self.metal_low_freq_weight = metal_low_freq_weight

    def sigma_from_abc(self, abc):
        """
        abc: (B, 2) with (b, c) >= 0  [a=0 so σ→0 as ξ→∞]
        returns sigma: (B, 1, F) broadcast-ready
        """
        b = abc[:, 0:1]
        c = abc[:, 1:2].clamp_min(self.eps_min_c)
        xi = self.xi.view(1, -1)
        denom = (xi + c) ** self.alpha
        sigma = b / denom
        return sigma.unsqueeze(1)

    def forward(self, eps_pred, abc_pred, eps_true, is_metal=None):
        """
        eps_pred: (B, C, F) (either ε or log(ε); see eps_pred_is_log)
        abc_pred: (B, 2)  — (b, c) with b,c > 0
        eps_true: (B, C, F)
        is_metal: (B,) bool, optional — discounts first n_low_freq freqs for metals
        """
        eps_pred_is_log = bool(getattr(self, "eps_pred_is_log", False))
        if eps_pred_is_log:
            eps_true_for_eps_loss = torch.log(eps_true)
        else:
            eps_true_for_eps_loss = eps_true

        raw_eps_err = (torch.abs(eps_pred - eps_true_for_eps_loss)
                       if self.eps_reduction == "mae"
                       else (eps_pred - eps_true_for_eps_loss) ** 2)
        if is_metal is not None and self.n_low_freq > 0:
            w = torch.ones(eps_pred.shape[0], 1, eps_pred.shape[2], device=eps_pred.device)
            w[is_metal.to(eps_pred.device), :, :self.n_low_freq] = self.metal_low_freq_weight
            raw_eps_err = raw_eps_err * w
        loss_eps = raw_eps_err.mean()

        if eps_pred_is_log:
            eps_pred_real = torch.exp(eps_pred)
        else:
            eps_pred_real = eps_pred

        abs_err = torch.abs(eps_pred_real.detach() - eps_true)
        abs_err = torch.mean(abs_err, dim=1, keepdim=True)
        sigma = self.sigma_from_abc(abc_pred)

        if is_metal is not None and self.n_low_freq > 0:
            freq_w = torch.ones(abs_err.shape[0], 1, abs_err.shape[2], device=abs_err.device)
            freq_w[is_metal.to(abs_err.device), :, :self.n_low_freq] = self.metal_low_freq_weight
            abs_err = abs_err * freq_w
            sigma = sigma * freq_w

        if self.sigma_transform == "log1p":
            abs_err = torch.log1p(abs_err)
            sigma = torch.log1p(sigma)
        elif self.sigma_transform == "log":
            abs_err = torch.log(abs_err + 1e-12)
            sigma = torch.log(sigma + 1e-12)

        if self.sigma_reduction == "mae":
            loss_sigma = torch.mean(torch.abs(sigma - abs_err))
        else:
            loss_sigma = torch.mean((sigma - abs_err) ** 2)

        return self.lambda_eps * loss_eps + self.lambda_sigma * loss_sigma, {
            "loss_eps": float(loss_eps.item()),
            "loss_sigma": float(loss_sigma.item()),
        }

