"""
Loss for joint prediction of ε(iξ) and power-law error parameters (a,b,c).

We do not assume labeled (a,b,c). Instead we supervise (a,b,c) by matching the
predicted uncertainty curve σ(ξ) to the observed absolute residual |ε_pred-ε_true|
within each batch.

Power-law uncertainty curve (same convention as fit_error_models.model_power_law):
  σ(ξ) = a + b / (ξ + c)^α

α is fixed (set in training script), while a,b,c are predicted per crystal.
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

    def sigma_from_abc(self, abc):
        """
        abc: (B, 3) with (a,b,c) >= 0
        returns sigma: (B, 1, F) broadcast-ready
        """
        a = abc[:, 0:1]
        b = abc[:, 1:2]
        c = abc[:, 2:3].clamp_min(self.eps_min_c)
        xi = self.xi.view(1, -1)
        denom = (xi + c) ** self.alpha
        sigma = a + b / denom
        return sigma.unsqueeze(1)

    def forward(self, eps_pred, abc_pred, eps_true):
        """
        eps_pred: (B, C, F) (either ε or log(ε); see eps_pred_is_log)
        abc_pred: (B, 3)
        eps_true: (B, C, F)
        """
        eps_pred_is_log = bool(getattr(self, "eps_pred_is_log", False))
        if eps_pred_is_log:
            eps_true_for_eps_loss = torch.log(eps_true)
        else:
            eps_true_for_eps_loss = eps_true

        if self.eps_reduction == "mae":
            loss_eps = torch.mean(torch.abs(eps_pred - eps_true_for_eps_loss))
        else:
            loss_eps = torch.mean((eps_pred - eps_true_for_eps_loss) ** 2)

        if eps_pred_is_log:
            eps_pred_real = torch.exp(eps_pred)
        else:
            eps_pred_real = eps_pred

        abs_err = torch.abs(eps_pred_real.detach() - eps_true)
        abs_err = torch.mean(abs_err, dim=1, keepdim=True)
        sigma = self.sigma_from_abc(abc_pred)

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

