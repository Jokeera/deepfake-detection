"""
Temporal-only модель (ablation A3 / temporal baseline).

Только временная ветвь:
Temporal Branch -> Classification Head

Используется для проверки, насколько temporal признаки
информативны сами по себе без spatial branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.temporal_branch import TemporalBranch
from models.dual_path import ClassificationHead


class TemporalOnlyModel(nn.Module):
    """
    Temporal-only baseline:
    diff-клипы -> temporal branch -> head -> logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.temporal_branch = TemporalBranch(cfg)
        self.head = ClassificationHead(
            input_dim=cfg.projection_dim,
            dropout=cfg.head_dropout,
        )

    def forward(self, spatial_input: torch.Tensor = None, temporal_input: torch.Tensor = None):
        """
        Args:
            spatial_input: не используется, оставлен для совместимости интерфейса
            temporal_input: [B, T-1, 3, H, W]

        Returns:
            logits: [B]
            alpha:  None
        """
        if temporal_input is None:
            raise ValueError("Для TemporalOnlyModel требуется temporal_input.")

        if temporal_input.ndim != 5:
            raise ValueError(
                f"temporal_input должен иметь размерность [B, T-1, C, H, W], "
                f"получено: {tuple(temporal_input.shape)}"
            )

        h_t = self.temporal_branch(temporal_input)  # [B, projection_dim]
        logits = self.head(h_t)                     # [B]
        return logits, None

    def freeze_spatial_backbone(self) -> None:
        # Spatial backbone отсутствует — оставляем метод для совместимости интерфейса.
        return None

    def unfreeze_spatial_backbone(self) -> None:
        # Spatial backbone отсутствует — оставляем метод для совместимости интерфейса.
        return None