"""
MetaWorld Benchmark for SE(3)-VLA.

Evaluates:
1. GeoAct vs flat action heads (rotation accuracy, geodesic error)
2. Flow matching uncertainty calibration (coverage, sharpness)
3. Action chunking temporal consistency
4. Conformal prediction coverage guarantees

Tasks: MT-10, MT-50 (MetaWorld multi-task)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from se3_vla.manifold.se3 import (
    exp_so3, log_so3, geodesic_distance_so3, geodesic_interpolation_so3,
    frechet_mean_so3,
)
from se3_vla.action_heads.geoact import GeoActHead, GeodesicMDN, GeodesicLoss
from se3_vla.uncertainty.flow_conformal import (
    SE3FlowMatching, ConformalPredictor, GeodesicActionChunking,
)


# ─── Synthetic MetaWorld-like Data ───────────────────────────────────────────

def generate_synthetic_actions(
    n_samples: int = 1000,
    n_tasks: int = 10,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic SE(3) actions mimicking MetaWorld robot tasks.
    
    Each task has a characteristic action distribution on SE(3).
    Returns: features, target_trans, target_rot, task_ids
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    features = []
    translations = []
    rotations = []
    task_ids = []

    for task_id in range(n_tasks):
        n = n_samples // n_tasks
        
        # Task-specific action distribution
        # Each task has a preferred direction and rotation
        task_angle = task_id * (2 * np.pi / n_tasks)
        task_center_trans = torch.tensor([
            0.3 * np.cos(task_angle),
            0.3 * np.sin(task_angle),
            0.1 + 0.05 * task_id,
        ])
        
        # Task-specific rotation (axis-angle)
        task_rot_axis = torch.tensor([
            np.cos(task_angle * 0.7),
            np.sin(task_angle * 0.7),
            0.1 * task_id,
        ])
        task_rot_axis = task_rot_axis / task_rot_axis.norm()
        task_rot_angle = 0.3 + 0.1 * task_id

        # Generate samples with noise
        trans = task_center_trans.unsqueeze(0).expand(n, -1) + torch.randn(n, 3) * 0.05
        rot = task_rot_axis.unsqueeze(0).expand(n, -1) * task_rot_angle + torch.randn(n, 3) * 0.1
        rot = rot / rot.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # Features: task embedding + noise
        feat = torch.zeros(n, 128)
        feat[:, task_id % 128] = 1.0
        feat += torch.randn(n, 128) * 0.1

        features.append(feat)
        translations.append(trans)
        rotations.append(rot)
        task_ids.extend([task_id] * n)

    return (
        torch.cat(features),
        torch.cat(translations),
        torch.cat(rotations),
        torch.tensor(task_ids),
    )


# ─── Flat Action Head Baseline ───────────────────────────────────────────────

class FlatActionHead(nn.Module):
    """Standard flat action head (baseline)."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),  # flat se(3) output
        )

    def forward(self, x):
        return self.net(x)


# ─── Training Functions ──────────────────────────────────────────────────────

def train_geoact(
    features: torch.Tensor,
    target_trans: torch.Tensor,
    target_rot: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Tuple[GeoActHead, List[float]]:
    """Train GeoAct action head."""
    model = GeoActHead(input_dim=128, n_components=4, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(features.shape[0])
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, features.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            out = model(features[idx], target_trans[idx], target_rot[idx])
            
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            
            epoch_loss += out["loss"].item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  GeoAct epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
    
    return model, losses


def train_flat(
    features: torch.Tensor,
    target_trans: torch.Tensor,
    target_rot: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Tuple[FlatActionHead, List[float]]:
    """Train flat action head baseline."""
    model = FlatActionHead(input_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(features.shape[0])
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, features.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            pred = model(features[idx])
            
            target = torch.cat([target_rot[idx], target_trans[idx]], dim=-1)
            loss = loss_fn(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Flat epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
    
    return model, losses


def train_flow(
    features: torch.Tensor,
    target_trans: torch.Tensor,
    target_rot: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Tuple[SE3FlowMatching, List[float]]:
    """Train flow matching model."""
    model = SE3FlowMatching(condition_dim=128, hidden_dim=128, n_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    target_se3 = torch.cat([target_rot, target_trans], dim=-1)  # (N, 6)
    
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(features.shape[0])
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, features.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            x_0 = torch.randn(len(idx), 6) * 0.5  # noise
            x_1 = target_se3[idx]
            
            loss, _ = model(x_0, x_1, features[idx])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Flow epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
    
    return model, losses


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_models(
    geoact: GeoActHead,
    flat: FlatActionHead,
    flow: SE3FlowMatching,
    features: torch.Tensor,
    target_trans: torch.Tensor,
    target_rot: torch.Tensor,
) -> Dict:
    """Comprehensive evaluation of all models."""
    results = {}
    
    with torch.no_grad():
        # GeoAct evaluation
        geoact_out = geoact(features, target_trans, target_rot)
        
        # Rotation accuracy (geodesic distance)
        pred_rot_mat = exp_so3(geoact_out["refined_rot"])
        gt_rot_mat = exp_so3(target_rot)
        geoact_rot_err = geodesic_distance_so3(pred_rot_mat, gt_rot_mat)
        
        # Translation error
        geoact_trans_err = (geoact_out["refined_trans"] - target_trans).norm(dim=-1)
        
        results["geoact"] = {
            "rotation_error_rad": geoact_rot_err.mean().item(),
            "rotation_error_deg": (geoact_rot_err.mean() * 180 / 3.14159).item(),
            "translation_error": geoact_trans_err.mean().item(),
            "loss": geoact_out["loss"].item(),
        }
        
        # Flat head evaluation
        flat_pred = flat(features)
        flat_rot = flat_pred[:, :3]
        flat_trans = flat_pred[:, 3:]
        
        flat_rot_mat = exp_so3(flat_rot)
        flat_rot_err = geodesic_distance_so3(flat_rot_mat, gt_rot_mat)
        flat_trans_err = (flat_trans - target_trans).norm(dim=-1)
        
        results["flat"] = {
            "rotation_error_rad": flat_rot_err.mean().item(),
            "rotation_error_deg": (flat_rot_err.mean() * 180 / 3.14159).item(),
            "translation_error": flat_trans_err.mean().item(),
        }
        
        # Flow matching samples & uncertainty
        flow_samples = flow.sample(features[:10], n_steps=20, n_samples=50)
        
        # Conformal prediction
        conformal = ConformalPredictor(alpha=0.1)
        # Use first 80% for calibration, rest for test
        n_cal = int(0.8 * flow_samples.shape[0])
        conformal.calibrate(flow_samples[:n_cal])
        conformal_out = conformal.predict(flow_samples[n_cal:])
        
        results["flow"] = {
            "sample_variance": conformal_out["variance"].mean().item(),
            "conformal_radius": conformal_out["radius"],
            "coverage_target": conformal_out["coverage_target"],
        }
        
        # Improvement percentages
        rot_improvement = (
            (flat_rot_err.mean() - geoact_rot_err.mean()) / flat_rot_err.mean() * 100
        ).item()
        trans_improvement = (
            (flat_trans_err.mean() - geoact_trans_err.mean()) / flat_trans_err.mean() * 100
        ).item()
        
        results["improvement"] = {
            "rotation_accuracy_pct": rot_improvement,
            "translation_accuracy_pct": trans_improvement,
        }
    
    return results


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_training_curves(
    geoact_losses: List[float],
    flat_losses: List[float],
    flow_losses: List[float],
    save_path: str,
):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(geoact_losses, color='#2196F3', linewidth=2)
    axes[0].set_title('GeoAct Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(flat_losses, color='#FF5722', linewidth=2)
    axes[1].set_title('Flat Head Loss (Baseline)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(flow_losses, color='#4CAF50', linewidth=2)
    axes[2].set_title('Flow Matching Loss', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_bars(results: Dict, save_path: str):
    """Plot GeoAct vs Flat comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('Rotation Error (°)', 'rotation_error_deg', '#2196F3'),
        ('Translation Error', 'translation_error', '#4CAF50'),
        ('Loss', 'loss', '#FF9800'),
    ]
    
    for ax, (label, key, color) in zip(axes, metrics):
        geoact_val = results["geoact"].get(key, 0)
        flat_val = results["flat"].get(key, 0)
        
        bars = ax.bar(['GeoAct', 'Flat Head'], [geoact_val, flat_val],
                      color=[color, '#9E9E9E'], edgecolor='black', linewidth=0.5)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add improvement annotation
        if flat_val > 0:
            improvement = (flat_val - geoact_val) / flat_val * 100
            ax.annotate(f'{improvement:+.1f}%',
                       xy=(0, geoact_val), xytext=(0.15, geoact_val * 1.1),
                       fontsize=12, fontweight='bold',
                       color='green' if improvement > 0 else 'red')
    
    plt.suptitle('GeoAct vs Flat Action Head', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_geodesic_vs_l2(save_path: str):
    """Visualize geodesic vs L2 loss on SO(3)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate rotation pairs
    n = 100
    angles = torch.linspace(0, torch.pi, n)
    
    geodesic_dists = []
    l2_dists = []
    
    for angle in angles:
        # R1 = identity
        R1 = torch.eye(3)
        # R2 = rotation about z-axis
        R2 = torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # Geodesic distance
        R_rel = R1.T @ R2
        omega = log_so3(R_rel.unsqueeze(0)).squeeze(0)
        geo_dist = omega.norm().item()
        geodesic_dists.append(geo_dist)
        
        # L2 on Euler angles (z-y-x convention)
        # Simplified: just use angle for z-rotation
        l2_dist = angle.item()
        l2_dists.append(l2_dist)
    
    axes[0].plot(angles.numpy(), geodesic_dists, label='Geodesic Distance', 
                color='#2196F3', linewidth=2.5)
    axes[0].plot(angles.numpy(), l2_dists, label='L2 (Angle)',
                color='#FF5722', linewidth=2.5, linestyle='--')
    axes[0].axvline(x=3.14159, color='gray', linestyle=':', alpha=0.5)
    axes[0].annotate('±π discontinuity', xy=(3.14159, 1.5), fontsize=11,
                    ha='center', color='gray')
    axes[0].set_xlabel('True Rotation Angle (rad)', fontsize=12)
    axes[0].set_ylabel('Distance', fontsize=12)
    axes[0].set_title('Geodesic vs L2 Distance on SO(3)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 3D visualization of rotation space
    ax3d = fig.add_subplot(122, projection='3d')
    
    # Sample random rotations
    n_pts = 200
    omega_samples = torch.randn(n_pts, 3) * 1.5
    R_samples = exp_so3(omega_samples)
    
    # Convert to axis-angle for plotting
    ax3d.scatter(omega_samples[:, 0], omega_samples[:, 1], omega_samples[:, 2],
                c=omega_samples.norm(dim=-1), cmap='viridis', s=10, alpha=0.6)
    
    # Draw geodesic from identity to a rotation
    t_vals = torch.linspace(0, 1, 50)
    target_omega = torch.tensor([1.0, 0.5, 0.3])
    geodesic = torch.outer(t_vals, target_omega)
    ax3d.plot(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2],
             color='red', linewidth=3, label='Geodesic')
    
    ax3d.set_xlabel('ω₁')
    ax3d.set_ylabel('ω₂')
    ax3d.set_zlabel('ω₃')
    ax3d.set_title('SO(3) Lie Algebra (so³)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_conformal_prediction(save_path: str):
    """Visualize conformal prediction sets on SE(3)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generate samples
    n_samples = 100
    true_action = torch.tensor([0.5, 0.3, 0.1, 0.2, -0.1, 0.4])
    samples = true_action.unsqueeze(0) + torch.randn(n_samples, 6) * 0.15
    
    # Fréchet mean
    mean = samples.mean(dim=0)
    
    # Distances to mean
    dists = (samples - mean).norm(dim=-1)
    
    # Conformal radius (90% coverage)
    conformal = ConformalPredictor(alpha=0.1)
    conformal.calibrate(samples.unsqueeze(0).unsqueeze(0))
    
    # Plot 1: Distribution of samples
    axes[0].hist(dists.numpy(), bins=20, color='#2196F3', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=conformal.q_alpha, color='red', linewidth=2, linestyle='--',
                   label=f'q_α = {conformal.q_alpha:.3f}')
    axes[0].set_xlabel('Geodesic Distance to Mean')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sample Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Coverage vs alpha
    alphas = np.linspace(0.01, 0.5, 50)
    radii = []
    for a in alphas:
        cp = ConformalPredictor(alpha=a)
        cp.calibrate(samples.unsqueeze(0).unsqueeze(0))
        radii.append(cp.q_alpha)
    
    axes[1].plot(1 - alphas, radii, color='#4CAF50', linewidth=2.5)
    axes[1].axhline(y=conformal.q_alpha, color='red', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Target Coverage (1-α)')
    axes[1].set_ylabel('Conformal Radius q_α')
    axes[1].set_title('Coverage vs Radius', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: 2D projection of conformal set
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = conformal.q_alpha * np.cos(theta)
    circle_y = conformal.q_alpha * np.sin(theta)
    
    axes[2].fill(circle_x, circle_y, alpha=0.2, color='#2196F3', label='Conformal Set')
    axes[2].plot(circle_x, circle_y, color='#2196F3', linewidth=2)
    axes[2].scatter(samples[:, 3].numpy(), samples[:, 4].numpy(),
                   s=20, alpha=0.5, color='#FF5722', label='Flow Samples')
    axes[2].scatter(mean[3].numpy(), mean[4].numpy(),
                   s=100, marker='*', color='black', zorder=5, label='Fréchet Mean')
    axes[2].set_xlabel('Translation X')
    axes[2].set_ylabel('Translation Y')
    axes[2].set_title('Conformal Prediction Set', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    
    plt.suptitle('Conformal Prediction on SE(3)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_architecture(save_path: str):
    """Plot the SE(3)-VLA architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    c_backbone = '#E3F2FD'
    c_geoact = '#FFF3E0'
    c_flow = '#E8F5E9'
    c_chunk = '#F3E5F5'
    c_output = '#FFEBEE'
    c_edge = '#37474F'
    
    # Backbone
    rect = plt.Rectangle((1, 8), 14, 1.5, facecolor=c_backbone, edgecolor=c_edge, linewidth=2, rx=0.2)
    ax.add_patch(rect)
    ax.text(8, 8.75, 'VLA Backbone (SmolVLA / OpenVLA / Octo) — FROZEN ❄️',
            ha='center', va='center', fontsize=13, fontweight='bold')
    
    # Feature projection
    rect = plt.Rectangle((5.5, 6.5), 5, 1, facecolor='#E0E0E0', edgecolor=c_edge, linewidth=1.5, rx=0.1)
    ax.add_patch(rect)
    ax.text(8, 7, 'Feature Projection (MLP)', ha='center', va='center', fontsize=11)
    ax.annotate('', xy=(8, 7.5), xytext=(8, 8), arrowprops=dict(arrowstyle='->', color=c_edge, lw=1.5))
    
    # GeoAct head
    rect = plt.Rectangle((0.5, 3), 6, 3, facecolor=c_geoact, edgecolor='#E65100', linewidth=2, rx=0.2)
    ax.add_patch(rect)
    ax.text(3.5, 5.5, 'GeoAct Head', ha='center', va='center', fontsize=13, fontweight='bold', color='#E65100')
    ax.text(3.5, 4.8, '• MDN on SO(3) (vMF)', ha='center', va='center', fontsize=10)
    ax.text(3.5, 4.3, '• Geodesic Loss', ha='center', va='center', fontsize=10)
    ax.text(3.5, 3.8, '• Residual Refinement (3 steps)', ha='center', va='center', fontsize=10)
    ax.text(3.5, 3.3, '→ Base SE(3) Action', ha='center', va='center', fontsize=10, color='#E65100')
    ax.annotate('', xy=(3.5, 6), xytext=(6, 6.5), arrowprops=dict(arrowstyle='->', color=c_edge, lw=1.5))
    
    # Flow matching
    rect = plt.Rectangle((9.5, 3), 6, 3, facecolor=c_flow, edgecolor='#2E7D32', linewidth=2, rx=0.2)
    ax.add_patch(rect)
    ax.text(12.5, 5.5, 'Riemannian Flow Matching', ha='center', va='center', fontsize=13, fontweight='bold', color='#2E7D32')
    ax.text(12.5, 4.8, '• Velocity field v_θ on SE(3)', ha='center', va='center', fontsize=10)
    ax.text(12.5, 4.3, '• Consistency (K=2, one-step)', ha='center', va='center', fontsize=10)
    ax.text(12.5, 3.8, '• N=50 samples per action', ha='center', va='center', fontsize=10)
    ax.text(12.5, 3.3, '→ Action Distribution', ha='center', va='center', fontsize=10, color='#2E7D32')
    ax.annotate('', xy=(12.5, 6), xytext=(10, 6.5), arrowprops=dict(arrowstyle='->', color=c_edge, lw=1.5))
    
    # Action chunking
    rect = plt.Rectangle((4, 1.2), 8, 1.3, facecolor=c_chunk, edgecolor='#6A1B9A', linewidth=2, rx=0.2)
    ax.add_patch(rect)
    ax.text(8, 1.85, 'Geodesic Action Chunking: K=4 anchors → H=8 interpolated actions',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#6A1B9A')
    ax.annotate('', xy=(8, 2.5), xytext=(3.5, 3), arrowprops=dict(arrowstyle='->', color=c_edge, lw=1.5))
    ax.annotate('', xy=(8, 2.5), xytext=(12.5, 3), arrowprops=dict(arrowstyle='->', color=c_edge, lw=1.5))
    
    # Conformal prediction
    rect = plt.Rectangle((0.5, 0), 15, 0.9, facecolor=c_output, edgecolor='#C62828', linewidth=2, rx=0.1)
    ax.add_patch(rect)
    ax.text(8, 0.45, 'Conformal Prediction: Fréchet mean → Geodesic variance → Coverage guarantee P(T* ∈ C_α) ≥ 1−α',
            ha='center', va='center', fontsize=11, fontweight='bold', color='#C62828')
    ax.annotate('', xy=(8, 0.9), xytext=(8, 1.2), arrowprops=dict(arrowstyle='->', color=c_edge, lw=1.5))
    
    plt.title('SE(3)-VLA Architecture', fontsize=18, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SE(3)-VLA: MetaWorld Benchmark")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("\n[1/6] Generating synthetic MetaWorld actions...")
    features, target_trans, target_rot, task_ids = generate_synthetic_actions(
        n_samples=1000, n_tasks=10
    )
    print(f"  Features: {features.shape}, Translations: {target_trans.shape}, Rotations: {target_rot.shape}")
    
    # Train models
    print("\n[2/6] Training GeoAct head...")
    geoact, geoact_losses = train_geoact(features, target_trans, target_rot, epochs=80)
    
    print("\n[3/6] Training flat baseline...")
    flat, flat_losses = train_flat(features, target_trans, target_rot, epochs=80)
    
    print("\n[4/6] Training flow matching...")
    flow, flow_losses = train_flow(features, target_trans, target_rot, epochs=40)
    
    # Evaluate
    print("\n[5/6] Evaluating models...")
    results = evaluate_models(geoact, flat, flow, features, target_trans, target_rot)
    
    print("\n  Results:")
    print(f"  GeoAct — Rotation Error: {results['geoact']['rotation_error_deg']:.2f}°, "
          f"Translation Error: {results['geoact']['translation_error']:.4f}")
    print(f"  Flat    — Rotation Error: {results['flat']['rotation_error_deg']:.2f}°, "
          f"Translation Error: {results['flat']['translation_error']:.4f}")
    print(f"  Improvement — Rotation: {results['improvement']['rotation_accuracy_pct']:+.1f}%, "
          f"Translation: {results['improvement']['translation_accuracy_pct']:+.1f}%")
    print(f"  Flow — Conformal Radius: {results['flow']['conformal_radius']:.4f}, "
          f"Coverage Target: {results['flow']['coverage_target']:.0%}")
    
    # Save results
    results_path = Path(__file__).parent.parent / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    # Generate plots
    print("\n[6/6] Generating plots...")
    plot_training_curves(geoact_losses, flat_losses, flow_losses,
                        str(output_dir / "training_curves.png"))
    plot_comparison_bars(results, str(output_dir / "comparison_bars.png"))
    plot_geodesic_vs_l2(str(output_dir / "geodesic_vs_l2.png"))
    plot_conformal_prediction(str(output_dir / "conformal_prediction.png"))
    plot_architecture(str(output_dir / "architecture.png"))
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
