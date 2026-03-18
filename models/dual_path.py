"""
Полная Dual-Path модель:
Spatial Branch + Temporal Branch + Fusion + Classification Head.

Это основной метод ВКР:
совместный пространственно-временной анализ манипулированных видеопоследовательностей.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spatial_branch import SpatialBranch
from models.temporal_branch import TemporalBranch


class AdaptiveWeightedFusion(nn.Module):
    """
    Обучаемое взвешивание двух представлений.

    alpha = softmax(W [h_s ; h_t] + b)
    h_fused = alpha_1 * h_s + alpha_2 * h_t
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.weight_layer = nn.Linear(input_dim * 2, 2)

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor):
        if h_s.shape != h_t.shape:
            raise ValueError(
                f"Размерности spatial и temporal признаков не совпадают: "
                f"{tuple(h_s.shape)} vs {tuple(h_t.shape)}"
            )

        concat = torch.cat([h_s, h_t], dim=-1)
        alpha = F.softmax(self.weight_layer(concat), dim=-1)
        h_fused = alpha[:, 0:1] * h_s + alpha[:, 1:2] * h_t
        return h_fused, alpha


class ConcatFusion(nn.Module):
    """
    Конкатенация + линейная проекция.
    Используется как ablation-вариант fusion.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor):
        if h_s.shape != h_t.shape:
            raise ValueError(
                f"Размерности spatial и temporal признаков не совпадают: "
                f"{tuple(h_s.shape)} vs {tuple(h_t.shape)}"
            )

        concat = torch.cat([h_s, h_t], dim=-1)
        h_fused = self.projection(concat)

        alpha = torch.full((h_s.shape[0], 2), 0.5, device=h_s.device, dtype=h_s.dtype)
        return h_fused, alpha


class GatedFusion(nn.Module):
    """
    Gated Fusion:
        g = sigmoid(W [h_s ; h_t])
        h = g * h_s + (1 - g) * h_t
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_s: torch.Tensor, h_t: torch.Tensor):
        if h_s.shape != h_t.shape:
            raise ValueError(
                f"Размерности spatial и temporal признаков не совпадают: "
                f"{tuple(h_s.shape)} vs {tuple(h_t.shape)}"
            )

        concat = torch.cat([h_s, h_t], dim=-1)
        g = self.gate(concat)
        h_fused = g * h_s + (1.0 - g) * h_t

        alpha_s = g.mean(dim=-1, keepdim=True)
        alpha_t = 1.0 - alpha_s
        alpha = torch.cat([alpha_s, alpha_t], dim=-1)
        return h_fused, alpha


class ClassificationHead(nn.Module):
    """
    Классификационная голова.
    Возвращает logits [B] под BCEWithLogitsLoss.
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class DualPathModel(nn.Module):
    """
    Полная Dual-Path модель:
    Spatial Branch || Temporal Branch -> Fusion -> Classification Head
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.spatial_branch = SpatialBranch(cfg)
        self.temporal_branch = TemporalBranch(cfg)

        if cfg.fusion_type == "adaptive":
            self.fusion = AdaptiveWeightedFusion(cfg.projection_dim)
        elif cfg.fusion_type == "concat":
            self.fusion = ConcatFusion(cfg.projection_dim)
        elif cfg.fusion_type == "gate":
            self.fusion = GatedFusion(cfg.projection_dim)
        else:
            raise ValueError(f"Неизвестный тип fusion: {cfg.fusion_type}")

        self.head = ClassificationHead(
            input_dim=cfg.projection_dim,
            dropout=cfg.head_dropout,
        )

    def forward(self, spatial_input: torch.Tensor, temporal_input: torch.Tensor):
        """
        Args:
            spatial_input:  [B, T, 3, H, W]
            temporal_input: [B, T-1, 3, H, W]

        Returns:
            logits: [B]
            alpha:  [B, 2]
        """
        if spatial_input.ndim != 5:
            raise ValueError(
                f"spatial_input должен иметь размерность [B, T, C, H, W], "
                f"получено: {tuple(spatial_input.shape)}"
            )

        if temporal_input.ndim != 5:
            raise ValueError(
                f"temporal_input должен иметь размерность [B, T-1, C, H, W], "
                f"получено: {tuple(temporal_input.shape)}"
            )

        if spatial_input.shape[0] != temporal_input.shape[0]:
            raise ValueError(
                f"Batch size не совпадает: "
                f"spatial B={spatial_input.shape[0]} vs temporal B={temporal_input.shape[0]}"
            )

        # NB: spatial и temporal входы могут иметь разные H,W
        # (spatial_size=224, temporal_size=128) — ветви обрабатывают независимо.

        expected_temporal_len = spatial_input.shape[1] - 1
        if temporal_input.shape[1] != expected_temporal_len:
            raise ValueError(
                f"Несогласованная длина клипа: spatial T={spatial_input.shape[1]}, "
                f"ожидалось temporal T-1={expected_temporal_len}, "
                f"получено temporal T={temporal_input.shape[1]}"
            )

        h_s = self.spatial_branch(spatial_input)
        h_t = self.temporal_branch(temporal_input)

        h_fused, alpha = self.fusion(h_s, h_t)
        logits = self.head(h_fused)

        return logits, alpha

    def freeze_spatial_backbone(self) -> None:
        """Замораживает spatial backbone для warmup-фазы."""
        self.spatial_branch.freeze_backbone()

    def unfreeze_spatial_backbone(self) -> None:
        """Размораживает последние блоки spatial backbone для fine-tuning."""
        self.spatial_branch.unfreeze_backbone(
            unfreeze_last_n_blocks=self.cfg.unfreeze_last_n_blocks
        )