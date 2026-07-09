"""
GeoAct: Geometry-Aware Action Head for VLA Models.

Drop-in replacement for any VLA model's action head that respects SE(3) manifold structure.
Uses geodesic loss on SO(3), von Mises-Fisher mixture density, and residual geometric refinement.

Based on GeoAct-Paper: "Geometry-Aware Action Prediction for VLA Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from ..manifold.se3 import (
    exp_so3, log_so3, hat_so3,
    geodesic_distance_so3, geodesic_interpolation_so3,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
)


class VonMisesFisher:
    """Von Mises-Fisher distribution on SO(3) for rotation modeling."""

    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor):
        """
        Args:
            mu: (..., 3) mean axis (unit vector)
            kappa: (..., 1) concentration parameter (>0)
        """
        self.mu = F.normalize(mu, dim=-1)
        self.kappa = kappa.clamp(min=1e-4)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability of observations x under vMF distribution."""
        # x: (..., 3) unit vectors (axis-angle direction)
        x_norm = F.normalize(x, dim=-1)
        kappa = self.kappa
        p = x.shape[-1]
        
        # log C_p(kappa) ≈ (p/2-1)*log(kappa) - (p/2)*log(2π) - kappa (for large kappa)
        log_norm = (
            (p / 2 - 1) * kappa.log()
            - (p / 2) * math.log(2 * math.pi)
            - kappa
            + kappa.exp().log()  # correction
        )
        
        # log p(x) = log C_p(κ) + κ * μᵀx
        log_p = log_norm + kappa * (self.mu * x_norm).sum(dim=-1, keepdim=True)
        return log_p

    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Sample from vMF distribution."""
        # Rejection sampling (simplified)
        shape = self.mu.shape[:-1]
        samples = []
        for _ in range(n_samples):
            # Perturb mean by Gaussian noise scaled by 1/sqrt(kappa)
            noise = torch.randn_like(self.mu) / self.kappa.sqrt().clamp(min=1e-4)
            sample = F.normalize(self.mu + noise, dim=-1)
            samples.append(sample)
        return torch.stack(samples, dim=-2)


class GeodesicMDN(nn.Module):
    """
    Mixture Density Network on SE(3).
    
    Predicts K mixture components, each with:
    - Translation: Gaussian in R³
    - Rotation: von Mises-Fisher on SO(3)
    """

    def __init__(self, input_dim: int, n_components: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.n_components = n_components

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Mixture weights
        self.pi_head = nn.Linear(hidden_dim, n_components)

        # Translation: mean + log-std for each component
        self.trans_mean_head = nn.Linear(hidden_dim, n_components * 3)
        self.trans_logstd_head = nn.Linear(hidden_dim, n_components * 3)

        # Rotation: vMF mean direction + log-concentration for each component
        self.rot_mean_head = nn.Linear(hidden_dim, n_components * 3)
        self.rot_logkappa_head = nn.Linear(hidden_dim, n_components)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, D) input features from VLA backbone
        Returns:
            dict with keys: pi, trans_mean, trans_std, rot_mean, rot_kappa
        """
        h = self.shared(features)
        B = features.shape[0]
        K = self.n_components

        pi = F.softmax(self.pi_head(h), dim=-1)  # (B, K)

        trans_mean = self.trans_mean_head(h).reshape(B, K, 3)
        trans_std = self.trans_logstd_head(h).reshape(B, K, 3).exp().clamp(min=1e-4)

        rot_mean = F.normalize(
            self.rot_mean_head(h).reshape(B, K, 3), dim=-1
        )
        rot_kappa = self.rot_logkappa_head(h).reshape(B, K, 1).exp().clamp(min=1e-4)

        return {
            "pi": pi,
            "trans_mean": trans_mean,
            "trans_std": trans_std,
            "rot_mean": rot_mean,
            "rot_kappa": rot_kappa,
        }

    def log_prob(
        self,
        params: Dict[str, torch.Tensor],
        target_trans: torch.Tensor,
        target_rot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probability of target action under mixture model.
        
        Args:
            params: output of forward()
            target_trans: (B, 3) target translation
            target_rot: (B, 3) target rotation (axis-angle)
        Returns:
            (B,) log-probabilities
        """
        pi = params["pi"]  # (B, K)
        trans_mean = params["trans_mean"]  # (B, K, 3)
        trans_std = params["trans_std"]  # (B, K, 3)
        rot_mean = params["rot_mean"]  # (B, K, 3)
        rot_kappa = params["rot_kappa"]  # (B, K, 1)

        # Translation log-prob (Gaussian)
        t = target_trans.unsqueeze(1)  # (B, 1, 3)
        log_p_trans = -0.5 * (
            ((t - trans_mean) / trans_std).pow(2)
            + 2 * trans_std.log()
            + math.log(2 * math.pi)
        ).sum(-1)  # (B, K)

        # Rotation log-prob (vMF approximation)
        r = F.normalize(target_rot, dim=-1).unsqueeze(1)  # (B, 1, 3)
        cos_sim = (r * rot_mean).sum(-1, keepdim=True)  # (B, K, 1)
        log_p_rot = rot_kappa * cos_sim  # (B, K, 1)

        # Mixture log-prob
        log_p = torch.log(pi + 1e-8) + log_p_trans + log_p_rot.squeeze(-1)
        return torch.logsumexp(log_p, dim=-1)  # (B,)

    def sample(self, params: Dict[str, torch.Tensor], n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from mixture model."""
        pi = params["pi"]
        B, K = pi.shape

        # Select component
        comp_idx = torch.multinomial(pi, n_samples, replacement=True)  # (B, n_samples)

        # Sample translation
        trans_mean = params["trans_mean"]
        trans_std = params["trans_std"]
        idx = comp_idx.unsqueeze(-1).expand(-1, -1, 3)
        mu = torch.gather(trans_mean, 1, idx)
        sig = torch.gather(trans_std, 1, idx)
        trans_samples = mu + sig * torch.randn_like(mu)

        # Sample rotation (vMF via perturbation)
        rot_mean = params["rot_mean"]
        rot_kappa = params["rot_kappa"]
        idx_r = comp_idx.unsqueeze(-1).expand(-1, -1, 3)
        mu_r = torch.gather(rot_mean, 1, idx_r)
        kap = torch.gather(rot_kappa, 1, comp_idx.unsqueeze(-1))
        noise = torch.randn_like(mu_r) / kap.sqrt().clamp(min=1e-4)
        rot_samples = F.normalize(mu_r + noise, dim=-1)

        return trans_samples, rot_samples


class GeodesicLoss(nn.Module):
    """
    Geodesic loss for SE(3) actions.
    
    L = α * d_SO3(R_pred, R_gt)² + ||t_pred - t_gt||²
    """

    def __init__(self, rotation_weight: float = 1.0):
        super().__init__()
        self.alpha = rotation_weight

    def forward(
        self,
        pred_trans: torch.Tensor,
        pred_rot: torch.Tensor,
        gt_trans: torch.Tensor,
        gt_rot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_trans: (B, 3) predicted translation
            pred_rot: (B, 3, 3) predicted rotation matrix
            gt_trans: (B, 3) ground truth translation
            gt_rot: (B, 3, 3) ground truth rotation matrix
        """
        d_rot = geodesic_distance_so3(pred_rot, gt_rot)  # (B,)
        d_trans = (pred_trans - gt_trans).norm(dim=-1)  # (B,)
        return (self.alpha * d_rot.pow(2) + d_trans.pow(2)).mean()


class ResidualRefinement(nn.Module):
    """
    Iterative residual refinement on SE(3).
    
    action_{i+1} = action_i + correction_i(features, action_i)
    Each step reduces error by ~15% (GeoAct paper).
    """

    def __init__(self, input_dim: int, n_steps: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.n_steps = n_steps

        self.correction_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim + 6, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 6),
            )
            for _ in range(n_steps)
        ])

    def forward(
        self,
        features: torch.Tensor,
        initial_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D) VLA backbone features
            initial_action: (B, 6) initial se(3) action [ω, v]
        Returns:
            (B, 6) refined action after n_steps corrections
        """
        action = initial_action
        for net in self.correction_nets:
            inp = torch.cat([features, action], dim=-1)
            correction = net(inp)
            action = action + correction
        return action


class GeoActHead(nn.Module):
    """
    Complete GeoAct action head: MDN + Geodesic Loss + Residual Refinement.
    
    Drop-in replacement for any VLA model's action head.
    """

    def __init__(
        self,
        input_dim: int,
        n_components: int = 4,
        n_refine_steps: int = 3,
        hidden_dim: int = 256,
        rotation_weight: float = 1.0,
    ):
        super().__init__()
        self.mdn = GeodesicMDN(input_dim, n_components, hidden_dim)
        self.refiner = ResidualRefinement(input_dim, n_refine_steps, hidden_dim // 2)
        self.loss_fn = GeodesicLoss(rotation_weight)

    def forward(
        self,
        features: torch.Tensor,
        target_trans: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, D) from VLA backbone
            target_trans: (B, 3) GT translation (for training)
            target_rot: (B, 3) GT rotation axis-angle (for training)
        """
        params = self.mdn(features)

        # Best component prediction (highest weight)
        best_k = params["pi"].argmax(dim=-1)  # (B,)
        B = features.shape[0]
        idx = best_k[:, None, None].expand(-1, 1, 3)

        pred_trans = torch.gather(params["trans_mean"], 1, idx).squeeze(1)
        pred_rot = torch.gather(params["rot_mean"], 1, idx).squeeze(1)

        # Residual refinement
        initial_action = torch.cat([pred_rot, pred_trans], dim=-1)  # (B, 6)
        refined_action = self.refiner(features, initial_action)

        output = {
            "params": params,
            "pred_trans": pred_trans,
            "pred_rot": pred_rot,
            "refined_action": refined_action,
            "refined_trans": refined_action[..., 3:],
            "refined_rot": refined_action[..., :3],
        }

        # Compute loss if targets provided
        if target_trans is not None and target_rot is not None:
            # MDN loss (negative log-likelihood)
            mdn_loss = -self.mdn.log_prob(params, target_trans, target_rot).mean()

            # Geodesic loss on refined action
            refined_rot_mat = exp_so3(refined_action[..., :3])
            gt_rot_mat = exp_so3(target_rot)
            geo_loss = self.loss_fn(
                refined_action[..., 3:], refined_rot_mat,
                target_trans, gt_rot_mat
            )

            output["loss"] = mdn_loss + geo_loss
            output["mdn_loss"] = mdn_loss
            output["geo_loss"] = geo_loss

        return output
