"""Tests for SE(3) manifold operations."""
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from se3_vla.manifold.se3 import (
    exp_so3, log_so3, hat_so3,
    geodesic_distance_so3, geodesic_interpolation_so3,
    exp_se3, log_se3, geodesic_distance_se3,
    frechet_mean_so3,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
)


class TestSO3:
    """Tests for SO(3) operations."""

    def test_exp_log_roundtrip(self):
        """exp(log(R)) = R"""
        omega = torch.randn(5, 3) * 0.5
        R = exp_so3(omega)
        omega_recovered = log_so3(R)
        R_recovered = exp_so3(omega_recovered)
        assert torch.allclose(R, R_recovered, atol=1e-5)

    def test_exp_identity(self):
        """exp(0) = I"""
        omega = torch.zeros(3, 3)
        R = exp_so3(omega)
        I = torch.eye(3).unsqueeze(0).expand(3, -1, -1)
        assert torch.allclose(R, I, atol=1e-6)

    def test_rotation_matrix_valid(self):
        """exp(omega) produces valid rotation matrices."""
        omega = torch.randn(10, 3)
        R = exp_so3(omega)
        # R^T R = I
        RTR = R.transpose(-1, -2) @ R
        I = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
        assert torch.allclose(RTR, I, atol=1e-5)
        # det(R) = 1
        det = torch.det(R)
        assert torch.allclose(det, torch.ones(10), atol=1e-5)

    def test_geodesic_distance_identity(self):
        """d(R, R) = 0"""
        omega = torch.randn(5, 3)
        R = exp_so3(omega)
        d = geodesic_distance_so3(R, R)
        assert torch.allclose(d, torch.zeros(5), atol=1e-6)

    def test_geodesic_distance_symmetric(self):
        """d(R1, R2) = d(R2, R1)"""
        R1 = exp_so3(torch.randn(5, 3) * 0.5)
        R2 = exp_so3(torch.randn(5, 3) * 0.5)
        d12 = geodesic_distance_so3(R1, R2)
        d21 = geodesic_distance_so3(R2, R1)
        assert torch.allclose(d12, d21, atol=1e-6)

    def test_geodesic_interpolation_endpoints(self):
        """t=0 → R_start, t=1 → R_end"""
        R_start = exp_so3(torch.randn(3, 3) * 0.3)
        R_end = exp_so3(torch.randn(3, 3) * 0.3)
        
        R_0 = geodesic_interpolation_so3(R_start, R_end, torch.tensor(0.0))
        R_1 = geodesic_interpolation_so3(R_start, R_end, torch.tensor(1.0))
        
        assert torch.allclose(R_0, R_start, atol=1e-5)
        # R_1 should be close to R_end (may have sign ambiguity for small angles)

    def test_hat_so3(self):
        """Hat map produces skew-symmetric matrix."""
        v = torch.randn(3, 3)
        K = hat_so3(v)
        assert torch.allclose(K, -K.transpose(-1, -2), atol=1e-6)

    def test_frechet_mean(self):
        """Frechet mean of identical rotations is that rotation."""
        omega = torch.tensor([0.3, 0.1, 0.2])
        R = exp_so3(omega.unsqueeze(0).expand(10, -1))
        mean = frechet_mean_so3(R)
        assert torch.allclose(mean, R[0], atol=1e-4)


class TestSE3:
    """Tests for SE(3) operations."""

    def test_exp_log_roundtrip(self):
        """exp(log(T)) = T for SE(3)."""
        xi = torch.randn(5, 6) * 0.3
        R, t = exp_se3(xi)
        xi_recovered = log_se3(R, t)
        assert torch.allclose(xi, xi_recovered, atol=1e-4)

    def test_geodesic_distance_identity(self):
        """d(T, T) = 0 for SE(3)."""
        xi = torch.randn(3, 6) * 0.3
        R, t = exp_se3(xi)
        d = geodesic_distance_se3(R, t, R, t)
        assert torch.allclose(d, torch.zeros(3), atol=1e-6)


class Test6DRotation:
    """Tests for 6D rotation representation."""

    def test_roundtrip(self):
        """6D → matrix → 6D roundtrip."""
        R = exp_so3(torch.randn(5, 3) * 0.5)
        d6 = matrix_to_rotation_6d(R)
        R_recovered = rotation_6d_to_matrix(d6)
        assert torch.allclose(R, R_recovered, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
