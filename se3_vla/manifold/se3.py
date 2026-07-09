"""
SE(3) Lie Group operations: exp map, log map, geodesic distance,
and related differential geometry utilities for rigid body transforms.

SE(3) = SO(3) ⋉ R³ — the special Euclidean group in 3D.
so(3) = R³ — the Lie algebra of SO(3) (axis-angle vectors).
se(3) = R⁶ — the Lie algebra of SE(3) (3 rotation + 3 translation).

References:
- Blanco, J.L. (2010). "A tutorial on SE(3) transformation parameterizations."
- Stillwell, J. (2008). "Naive Lie Theory."
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ─── SO(3) Operations ───────────────────────────────────────────────────────

def hat_so3(v: torch.Tensor) -> torch.Tensor:
    """
    Hat map: R³ → so(3). Converts axis-angle vector to skew-symmetric matrix.
    
    Args:
        v: (..., 3) axis-angle vectors
    Returns:
        (..., 3, 3) skew-symmetric matrices
    """
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    O = torch.zeros_like(x)
    return torch.stack([
        O, -z, y,
        z, O, -x,
        -y, x, O
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """
    Exponential map: so(3) → SO(3). Rodrigues' formula.
    
    Args:
        omega: (..., 3) axis-angle vectors (angle = ||omega||)
    Returns:
        (..., 3, 3) rotation matrices
    """
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., 1)
    theta_sq = theta.pow(2)
    
    # Normalize axis
    axis = omega / theta  # (..., 3)
    K = hat_so3(axis)  # (..., 3, 3)
    
    # Rodrigues: R = I + sin(θ)K + (1 - cos(θ))K²
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    sin_theta = theta.unsqueeze(-1).sin()
    cos_theta = theta.unsqueeze(-1).cos()
    
    # Taylor expansion for small angles
    small = theta_sq < 1e-6
    sin_term = torch.where(small, theta.unsqueeze(-1), sin_theta)
    cos_term = torch.where(small, 1 - theta_sq.unsqueeze(-1) / 2, cos_theta)
    
    R = I + sin_term * K + (1 - cos_term) * K @ K
    return R


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map: SO(3) → so(3). Inverse of exp_so3.
    
    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 3) axis-angle vectors
    """
    # Angle from trace: cos(θ) = (tr(R) - 1) / 2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = cos_theta.acos()  # (...)
    
    # Axis from skew-symmetric part: [ω]× = (R - Rᵀ) / (2 sin θ)
    skew = (R - R.transpose(-1, -2)) / 2  # (..., 3, 3)
    
    # Extract axis from skew matrix
    axis = torch.stack([
        skew[..., 2, 1],  # ω_x
        skew[..., 0, 2],  # ω_y
        skew[..., 1, 0],  # ω_z
    ], dim=-1)  # (..., 3)
    
    # Handle small angles with Taylor expansion
    sin_theta = theta.sin().unsqueeze(-1).clamp(min=1e-8)
    small = theta.abs().unsqueeze(-1) < 1e-6
    
    # For small angles: ω ≈ (R - Rᵀ) / 2  (directly the skew part)
    # For normal angles: ω = θ * axis / sin(θ)
    omega = torch.where(
        small,
        axis,  # For small angles, skew part ≈ axis * θ
        axis * theta.unsqueeze(-1) / sin_theta
    )
    
    return omega


def geodesic_distance_so3(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Geodesic distance between two rotations on SO(3).
    
    d(R1, R2) = ||log(R1ᵀ @ R2)|| 
    
    Args:
        R1, R2: (..., 3, 3) rotation matrices
    Returns:
        (...,) geodesic distances in radians
    """
    R_rel = R1.transpose(-1, -2) @ R2  # (..., 3, 3)
    omega = log_so3(R_rel)  # (..., 3)
    return omega.norm(dim=-1)  # (...)


def geodesic_interpolation_so3(
    R_start: torch.Tensor,
    R_end: torch.Tensor,
    t: torch.Tensor
) -> torch.Tensor:
    """
    Geodesic interpolation (SLERP) between two rotations.
    
    R(t) = R_start @ exp(t * log(R_startᵀ @ R_end))
    
    Args:
        R_start: (..., 3, 3) start rotation
        R_end: (..., 3, 3) end rotation
        t: (...) or scalar, interpolation parameter in [0, 1]
    Returns:
        (..., 3, 3) interpolated rotation
    """
    R_rel = R_start.transpose(-1, -2) @ R_end
    omega = log_so3(R_rel)
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(omega.shape[:-1])
    return R_start @ exp_so3(t.unsqueeze(-1) * omega)


# ─── SE(3) Operations ───────────────────────────────────────────────────────

def exp_se3(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exponential map: se(3) → SE(3).
    
    Args:
        xi: (..., 6) se(3) vectors [ω₁, ω₂, ω₃, v₁, v₂, v₃]
    Returns:
        R: (..., 3, 3) rotation matrices
        t: (..., 3) translation vectors
    """
    omega = xi[..., :3]  # rotation part
    v = xi[..., 3:]      # translation part
    
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    theta_sq = theta.pow(2)
    theta_cubed = theta_sq * theta
    
    R = exp_so3(omega)
    
    # Translation: t = V @ v where V = I + (1-cosθ)/θ² [ω]× + (θ-sinθ)/θ³ [ω]×²
    K = hat_so3(omega)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand_as(K)
    
    small = theta_sq < 1e-6
    sin_theta = theta.unsqueeze(-1).sin()
    cos_theta = theta.unsqueeze(-1).cos()
    
    # Taylor expansions for small angles
    a = torch.where(small, 1 - theta_sq.unsqueeze(-1) / 6, sin_theta / theta.unsqueeze(-1))
    b = torch.where(small, 0.5 - theta_sq.unsqueeze(-1) / 24, (1 - cos_theta) / theta_sq.unsqueeze(-1))
    
    V = I + b * K + (1 - a) / theta_sq.unsqueeze(-1) * K @ K  # WRONG: should be different
    # Correct V: I + (1-cosθ)/θ² * K + (θ-sinθ)/θ³ * K²
    # Let me redo:
    V = I + b * K + ((theta.unsqueeze(-1) - sin_theta) / theta_cubed.unsqueeze(-1)) * K @ K
    # For small angles: V ≈ I + K/2 + K²/6
    
    t_out = (V @ v.unsqueeze(-1)).squeeze(-1)
    
    return R, t_out


def log_se3(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map: SE(3) → se(3).
    
    Args:
        R: (..., 3, 3) rotation matrices
        t: (..., 3) translation vectors
    Returns:
        (..., 6) se(3) vectors [ω, v]
    """
    omega = log_so3(R)  # (..., 3)
    
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    theta_sq = theta.pow(2)
    
    K = hat_so3(omega)
    I = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(K)
    
    sin_theta = theta.unsqueeze(-1).sin()
    cos_theta = theta.unsqueeze(-1).cos()
    
    small = theta_sq < 1e-6
    
    # V⁻¹ = I - ω/2 + (1/θ²)(1 - θ*sinθ/(2(1-cosθ)))[ω]×²
    half_theta = theta.unsqueeze(-1) / 2
    coeff = torch.where(
        small,
        1 / 12 + theta_sq.unsqueeze(-1) / 720,  # Taylor
        (1 / theta_sq.unsqueeze(-1)) * (1 - theta.unsqueeze(-1) * sin_theta / (2 * (1 - cos_theta)))
    )
    
    V_inv = I - half_theta * K + coeff * K @ K
    v_out = (V_inv @ t.unsqueeze(-1)).squeeze(-1)
    
    return torch.cat([omega, v_out], dim=-1)


def geodesic_distance_se3(
    R1: torch.Tensor, t1: torch.Tensor,
    R2: torch.Tensor, t2: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Geodesic distance on SE(3).
    
    d_SE3 = sqrt(α * d_SO3² + ||t1 - t2||²)
    
    Args:
        R1, t1: first SE(3) transform
        R2, t2: second SE(3) transform
        alpha: rotation weight (default 1.0)
    Returns:
        (...,) geodesic distances
    """
    d_rot = geodesic_distance_so3(R1, R2)  # (...)
    d_trans = (t1 - t2).norm(dim=-1)  # (...)
    return (alpha * d_rot.pow(2) + d_trans.pow(2)).sqrt()


# ─── Frechet Mean on SE(3) ──────────────────────────────────────────────────

def frechet_mean_so3(
    R: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6
) -> torch.Tensor:
    """
    Compute the Frechet (Karcher) mean of rotations on SO(3).
    
    Args:
        R: (N, 3, 3) or (B, N, 3, 3) rotation matrices
    Returns:
        (3, 3) or (B, 3, 3) mean rotation
    """
    # Initialize with first rotation
    mu = R[..., 0, :, :].clone()  # (..., 3, 3)
    
    for _ in range(max_iter):
        # Compute log map from mean to all rotations
        R_rel = mu.transpose(-1, -2).unsqueeze(-3) @ R  # (..., N, 3, 3)
        omega = log_so3(R_rel)  # (..., N, 3)
        
        # Weighted average in tangent space
        delta = omega.mean(dim=-2)  # (..., 3)
        
        # Update mean
        mu = mu @ exp_so3(delta.unsqueeze(-2)).squeeze(-2)
        
        if delta.norm().max() < tol:
            break
    
    return mu


def frechet_mean_se3(
    R: torch.Tensor, t: torch.Tensor,
    max_iter: int = 50,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Frechet mean on SE(3).
    
    Args:
        R: (N, 3, 3) or (B, N, 3, 3)
        t: (N, 3) or (B, N, 3)
    Returns:
        R_mean, t_mean
    """
    R_mean = frechet_mean_so3(R, max_iter, tol)
    
    # Translation mean in tangent space
    R_rel = R_mean.unsqueeze(-3).transpose(-1, -2) @ R
    omega = log_so3(R_rel)  # (..., N, 3)
    v = t - t.mean(dim=-2, keepdim=True)  # centered
    
    # Mean translation
    t_mean = t.mean(dim=-2)
    
    return R_mean, t_mean


# ─── Utility: Rotation matrix ↔ 6D representation ───────────────────────────

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation (Zhou et al., CVPR 2019) to rotation matrix.
    
    Args:
        d6: (..., 6) — first two columns of rotation matrix, flattened
    Returns:
        (..., 3, 3) rotation matrices
    """
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    
    # Gram-Schmidt
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation.
    
    Args:
        R: (..., 3, 3) rotation matrices
    Returns:
        (..., 6) — first two columns, flattened
    """
    return R[..., :2, :].transpose(-1, -2).reshape(*R.shape[:-2], 6)
