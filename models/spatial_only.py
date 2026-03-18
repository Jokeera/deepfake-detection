"""
Spatial-only модель (ablation A2 / baseline B1).

Только пространственная ветвь:
Spatial Branch -> Classification Head

Используется как контрольный baseline для оценки вклада temporal branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.spatial_branch import SpatialBranch
from models.dual_path import ClassificationHead


class SpatialOnlyModel(nn.Module):
    """
    Spatial-only baseline:
    видеоклип -> spatial branch -> head -> logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.spatial_branch = SpatialBranch(cfg)
        self.head = ClassificationHead(
            input_dim=cfg.projection_dim,
            dropout=cfg.head_dropout,
        )

    def forward(self, spatial_input: torch.Tensor, temporal_input: torch.Tensor = None):
        """
        Args:
            spatial_input:  [B, T, 3, H, W]
            temporal_input: не используется, оставлен для совместимости интерфейса

        Returns:
            logits: [B]
            alpha:  None
        """
        if spatial_input.ndim != 5:
            raise ValueError(
                f"spatial_input должен иметь размерность [B, T, C, H, W], "
                f"получено: {tuple(spatial_input.shape)}"
            )

        h_s = self.spatial_branch(spatial_input)   # [B, projection_dim]
        logits = self.head(h_s)                    # [B]
        return logits, None

    def freeze_spatial_backbone(self) -> None:
        self.spatial_branch.freeze_backbone()

    def unfreeze_spatial_backbone(self) -> None:
        self.spatial_branch.unfreeze_backbone(
            unfreeze_last_n_blocks=self.cfg.unfreeze_last_n_blocks
        )