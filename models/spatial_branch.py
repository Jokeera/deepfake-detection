"""
Пространственная ветвь (Spatial Branch).

Идея:
- каждый кадр видеоклипа обрабатывается независимо CNN-backbone'ом;
- затем покадровые признаки агрегируются по времени;
- далее выполняется проекция в общее пространство fusion.

Вход:
    x: [B, T, 3, H, W]

Выход:
    h_s: [B, projection_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class SpatialBranch(nn.Module):
    """
    Spatial Branch для извлечения пространственных признаков из видеоклипа.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.spatial_backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Аккуратно определяем размерность признаков backbone.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.spatial_size, cfg.spatial_size)
            feat_dim = self.backbone(dummy).shape[-1]
        self.backbone.train(was_training)

        self.feat_dim = feat_dim

        self.projection = nn.Sequential(
            nn.Linear(self.feat_dim, cfg.projection_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, 3, H, W]

        Returns:
            Tensor [B, projection_dim]
        """
        frame_features = self.get_frame_features(x)   # [B, T, feat_dim]
        aggregated = frame_features.mean(dim=1)       # [B, feat_dim]
        projected = self.projection(aggregated)       # [B, projection_dim]
        return projected

    def get_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Извлекает признаки каждого кадра без temporal aggregation.

        Args:
            x: Tensor [B, T, 3, H, W]

        Returns:
            Tensor [B, T, feat_dim]
        """
        if x.ndim != 5:
            raise ValueError(
                f"Ожидается вход размерности [B, T, C, H, W], получено: {tuple(x.shape)}"
            )

        b, t, c, h, w = x.shape
        # Приводим к contiguous-памяти перед подачей в backbone, чтобы
        # избежать ошибок .view(...) на MPS/CUDA при нестандартных страйдах.
        x = x.reshape(b * t, c, h, w).contiguous()  # [B*T, 3, H, W]
        features = self.backbone(x)                 # [B*T, feat_dim]
        features = features.reshape(b, t, -1)  # [B, T, feat_dim]
        return features

    def freeze_backbone(self) -> None:
        """Полностью замораживает backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 4) -> None:
        """
        Размораживает последние N блоков backbone.

        Для EfficientNet через timm обычно доступны self.backbone.blocks,
        а также head-слои bn2 / conv_head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "blocks") and unfreeze_last_n_blocks > 0:
            blocks = list(self.backbone.blocks)
            for block in blocks[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True