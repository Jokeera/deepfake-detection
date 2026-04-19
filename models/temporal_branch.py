"""
Временная ветвь (Temporal Branch).

Идея:
- берём последовательность разностей соседних кадров;
- извлекаем признаки для каждого diff-кадра через CNN-backbone;
- проектируем признаки в пространство Transformer;
- агрегируем временной контекст через Transformer Encoder;
- используем [CLS]-token как итоговое временное представление.

Вход:
    x: [B, T-1, 3, H, W]

Выход:
    h_t: [B, projection_dim]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class TemporalBranch(nn.Module):
    """
    Temporal Branch для анализа межкадровой динамики.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # CNN feature extractor для diff-кадров
        self.feature_extractor = timm.create_model(
            cfg.temporal_backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Аккуратно определяем размерность выхода backbone
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, cfg.temporal_size, cfg.temporal_size)
            feat_dim = self.feature_extractor(dummy).shape[-1]
        self.feature_extractor.train(was_training)

        self.feat_dim = feat_dim

        # Проекция признаков diff-кадров в пространство Transformer
        self.diff_projection = nn.Sequential(
            nn.Linear(self.feat_dim, cfg.transformer_dim),
            nn.LayerNorm(cfg.transformer_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.transformer_dropout),
        )

        # CLS-token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, cfg.transformer_dim) * 0.02
        )

        # Максимальная длина последовательности:
        # (T-1) diff-кадров + 1 CLS = T
        max_seq_len = cfg.num_frames
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, cfg.transformer_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_dim,
            nhead=cfg.transformer_heads,
            dim_feedforward=cfg.transformer_ff_dim,
            dropout=cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_layers,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(cfg.transformer_dim, cfg.projection_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T-1, 3, H, W]

        Returns:
            Tensor [B, projection_dim]
        """
        if x.ndim != 5:
            raise ValueError(
                f"Ожидается вход размерности [B, T-1, C, H, W], получено: {tuple(x.shape)}"
            )

        b, t_minus_1, c, h, w = x.shape

        # 1. CNN features для каждого diff-кадра
        # Contiguous перед подачей в backbone, чтобы избежать ошибок .view(...)
        # внутри timm на не‑контiguous входах.
        x = x.reshape(b * t_minus_1, c, h, w).contiguous()  # [B*(T-1), 3, H, W]
        features = self.feature_extractor(x)                # [B*(T-1), feat_dim]
        features = features.reshape(b, t_minus_1, -1)     # [B, T-1, feat_dim]

        # 2. Проекция в Transformer space
        projected = self.diff_projection(features)        # [B, T-1, transformer_dim]

        # 3. Добавляем CLS-token
        cls_tokens = self.cls_token.expand(b, -1, -1)     # [B, 1, transformer_dim]
        sequence = torch.cat([cls_tokens, projected], dim=1)  # [B, T, transformer_dim]

        # 4. Positional encoding
        seq_len = sequence.shape[1]
        if seq_len > self.pos_encoding.shape[1]:
            raise ValueError(
                f"Длина последовательности {seq_len} превышает "
                f"максимум positional encoding {self.pos_encoding.shape[1]}"
            )

        sequence = sequence + self.pos_encoding[:, :seq_len, :]

        # 5. Transformer
        encoded = self.transformer(sequence)              # [B, T, transformer_dim]

        # 6. Берём CLS
        cls_output = encoded[:, 0, :]                     # [B, transformer_dim]

        # 7. Проекция в fusion space
        output = self.output_projection(cls_output)       # [B, projection_dim]
        return output

    def get_attention_weights(self, x: torch.Tensor):
        """
        Заглушка для будущей визуализации attention.

        Сейчас attention weights явно не извлекаются.
        Если понадобится для анализа ошибок, нужно реализовать через hooks
        или кастомный Transformer слой.

        Returns:
            None
        """
        return None