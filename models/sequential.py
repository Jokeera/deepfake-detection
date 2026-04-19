"""
Sequential модель: CNN -> BiLSTM (ablation A4 / baseline B3).

Идея:
- сначала извлекаются пространственные признаки каждого кадра;
- затем они подаются как последовательность в BiLSTM;
- итоговое временное представление используется для классификации.

Это baseline для сравнения с parallel dual-path архитектурой.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.spatial_branch import SpatialBranch
from models.dual_path import ClassificationHead


class SequentialModel(nn.Module):
    """
    Sequential baseline:
    per-frame spatial features -> BiLSTM -> projection -> head -> logits
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.spatial_branch = SpatialBranch(cfg)

        self.lstm = nn.LSTM(
            input_size=self.spatial_branch.feat_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.transformer_dropout if cfg.lstm_layers > 1 else 0.0,
        )

        self.projection = nn.Sequential(
            nn.Linear(cfg.lstm_hidden * 2, cfg.projection_dim),
            nn.ReLU(inplace=True),
        )

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

        frame_features = self.spatial_branch.get_frame_features(spatial_input)  # [B, T, feat_dim]

        _, (h_n, _) = self.lstm(frame_features)

        # Для bidirectional LSTM:
        # h_n shape = [num_layers * 2, B, hidden]
        h_forward = h_n[-2, :, :]   # [B, hidden]
        h_backward = h_n[-1, :, :]  # [B, hidden]
        h_combined = torch.cat([h_forward, h_backward], dim=-1)  # [B, hidden*2]

        projected = self.projection(h_combined)  # [B, projection_dim]
        logits = self.head(projected)            # [B]

        return logits, None

    def freeze_spatial_backbone(self) -> None:
        self.spatial_branch.freeze_backbone()

    def unfreeze_spatial_backbone(self) -> None:
        self.spatial_branch.unfreeze_backbone(
            unfreeze_last_n_blocks=self.cfg.unfreeze_last_n_blocks
        )