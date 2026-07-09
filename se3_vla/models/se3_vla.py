"""
SE(3)-VLA Model: Wraps any VLA backbone with SE(3)-native action prediction.

Supported backbones:
- SmolVLA (450M)
- OpenVLA (7B)
- Octo (93M)
- RT-1-X
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from ..uncertainty.flow_conformal import SE3VLAHead


class SE3VLAModel(nn.Module):
    """
    SE(3)-Native VLA Model.
    
    Architecture:
    ┌─────────────────────────────────────────┐
    │  VLA Backbone (frozen or fine-tuned)    │
    │  Vision Encoder + Language Encoder      │
    │  → hidden state h ∈ R^D                 │
    ├─────────────────────────────────────────┤
    │  SE(3)-VLA Head (trainable)             │
    │  ┌─────────────┐ ┌──────────────────┐  │
    │  │ GeoAct MDN  │ │ Flow Matching    │  │
    │  │ + Geodesic  │ │ + Conformal      │  │
    │  │   Loss      │ │   Prediction     │  │
    │  │ + Residual  │ │ + Action Chunking│  │
    │  │   Refine    │ │                  │  │
    │  └──────┬──────┘ └────────┬─────────┘  │
    │         └────────┬────────┘             │
    │           SE(3) Action + Uncertainty    │
    └─────────────────────────────────────────┘
    """

    def __init__(
        self,
        backbone_dim: int = 768,
        n_components: int = 4,
        n_anchors: int = 4,
        horizon: int = 8,
        n_flow_samples: int = 50,
        alpha: float = 0.1,
        hidden_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Placeholder for backbone (user provides their own)
        self.backbone_dim = backbone_dim
        self.freeze_backbone = freeze_backbone

        # SE(3)-VLA head
        self.head = SE3VLAHead(
            input_dim=backbone_dim,
            n_components=n_components,
            n_anchors=n_anchors,
            horizon=horizon,
            n_flow_samples=n_flow_samples,
            alpha=alpha,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        features: torch.Tensor,
        target_trans: Optional[torch.Tensor] = None,
        target_rot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (B, D) from VLA backbone
            target_trans: (B, H, 3) GT translations
            target_rot: (B, H, 3) GT rotations
        """
        return self.head(features, target_trans, target_rot)

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict actions with uncertainty."""
        return self.head.predict_with_uncertainty(features)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per component."""
        geoact_params = sum(p.numel() for p in self.head.geoact.parameters())
        flow_params = sum(p.numel() for p in self.head.flow.parameters())
        chunk_params = sum(p.numel() for p in self.head.chunker.parameters())
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "geoact": geoact_params,
            "flow_matching": flow_params,
            "action_chunking": chunk_params,
            "total_trainable": total,
        }
