"""
Единая конфигурация проекта deepfake_detection.

Назначение:
- хранить пути, гиперпараметры и настройки экспериментов в одном месте;
- обеспечивать воспроизводимый запуск обучения, оценки и инференса;
- быть переносимой между macOS / Linux / Windows;
- отражать именно video-level постановку задачи для ВКР:
  на вход подаются клипы из последовательностей кадров, а не одиночные изображения.

Практические замечания для ВКР:
1. Этот конфиг — стабильный baseline для финальных экспериментов и MVP.
2. По результатам EDA fine-tuning spatial backbone на MPS оставлен отключённым
   (unfreeze_last_n_blocks = 0), так как ранее вызывал зависание процесса.
3. Параметры class imbalance добавлены как ОПЦИИ, но не включены по умолчанию:
   video-level DFDC после очистки близок к сбалансированному, поэтому
   агрессивную компенсацию дисбаланса не стоит включать без проверки train.py.
4. Основная метрика выбора лучшей модели — ROC-AUC.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional


ModelType = Literal["full", "spatial", "temporal", "sequential"]
FusionType = Literal["adaptive", "concat", "gate"]
DeviceMode = Literal["auto", "cpu", "cuda", "mps"]
SplitMode = Literal["random", "fixed", "official"]
RunMode = Literal["train", "eval", "infer"]


@dataclass
class Config:
    # =========================================================================
    # PATHS
    # =========================================================================
    # Корень проекта. По умолчанию: текущая рабочая директория.
    project_root: str = os.getcwd()

    # Путь к ПРЕДОБРАБОТАННОМУ датасету (относительно корня проекта).
    # Основной датасет для train/val/test — DFDC02 (сбалансирован).
    # DFD01 используется как cross-dataset для оценки переноса.
    # Ожидаемая структура: dataset_root/real/<id>/frame.jpg, fake/<id>/frame.jpg
    dataset_root: str = "data/preprocessed_data/preprocessed_DFDC02_16"

    # Путь к RAW-датасету с исходными видеофайлами.
    # Используется только для preprocessing / audit.
    raw_video_dataset_root: str = "./data"

    # Каталог хранения предобработанных кадров/клипов.
    preprocessed_dataset_root: str = "./preprocessed_frames"

    # Корневая директория экспериментов.
    output_dir: str = "./experiments"

    # Каталог для фиксированных split-файлов.
    splits_dir: str = "./splits"

    # Каталог для сервисных артефактов инференса / экспортов.
    artifacts_dir: str = "./artifacts"

    # =========================================================================
    # RUN CONTROL
    # =========================================================================
    run_mode: RunMode = "train"

    # Имя набора данных для логов / отчётов / воспроизводимости.
    # Примеры: "dfd01", "dfdc02", "ffpp", "celebdfv2"
    dataset_name: str = "dfdc02"

    # Режим формирования split.
    # fixed — воспроизводимость: один и тот же split (split_seed42.json).
    # random — пересобирать split при каждом запуске.
    split_mode: SplitMode = "fixed"

    # Имя split-файла (актуально для fixed / official).
    split_filename: str = "split_seed42.json"

    # Сохранять ли split после первого построения.
    # Полезно для воспроизводимости ВКР.
    save_split: bool = True

    # =========================================================================
    # DEVICE / CROSS-PLATFORM
    # =========================================================================
    # auto: cuda -> mps -> cpu
    # cpu : только CPU
    # cuda: только NVIDIA CUDA
    # mps : только Apple Silicon MPS
    device: DeviceMode = "auto"

    # Mixed precision:
    # False по умолчанию ради переносимости и предсказуемости.
    # В train/eval коде можно включать автоматически для CUDA.
    use_amp: bool = False

    # Число worker'ов по умолчанию ставим безопасным для macOS / Windows / Linux.
    num_workers: int = 0

    # pin_memory полезен в основном для CUDA.
    # Для cross-platform лучше не включать по умолчанию.
    pin_memory: bool = False

    # Для воспроизводимости.
    seed: int = 42

    # =========================================================================
    # DATA
    # =========================================================================
    # Число кадров в одном клипе.
    # Единый параметр: preprocess_videos.py читает это значение автоматически.
    # Увеличение (32, 64) даёт больше временного контекста, но требует больше памяти.
    # Уменьшение (8) ускоряет обучение, но теряет межкадровую информацию.
    num_frames: int = 16

    # Минимальное число кадров в видео для включения в выборку.
    # Должно быть >= num_frames. Видео с меньшим числом кадров будут отфильтрованы.
    min_frames_per_video: int = 16

    # Размер входа для spatial branch.
    spatial_size: int = 224

    # Размер входа для temporal branch.
    temporal_size: int = 128

    # Доли разбиения.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Важно:
    # split должен строиться ПО ВИДЕО, а не по кадрам.
    # Это соответствует video-level постановке ВКР.
    split_by_video: bool = True

    # Если dataset.py поддерживает стратификацию по label на уровне видео,
    # имеет смысл держать её включённой.
    stratify_by_label: bool = True

    # =========================================================================
    # MODEL
    # =========================================================================
    model_type: ModelType = "full"

    # Spatial branch
    spatial_backbone: str = "efficientnet_b4"

    # Temporal branch
    temporal_backbone: str = "efficientnet_b0"

    # Общее пространство для fusion
    projection_dim: int = 512

    # Temporal Transformer
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_dim: int = 256
    transformer_ff_dim: int = 1024
    transformer_dropout: float = 0.1

    # Sequential baseline (CNN -> BiLSTM)
    lstm_hidden: int = 256
    lstm_layers: int = 2

    # Classification head
    head_dropout: float = 0.3

    # Fusion for full model
    fusion_type: FusionType = "adaptive"

    # =========================================================================
    # TRAINING
    # =========================================================================
    batch_size: int = 8
    max_epochs: int = 30

    # LR для backbone и head
    lr_backbone: float = 1e-4
    lr_head: float = 3e-4

    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    # Early stopping по val AUC
    patience: int = 7

    # Warmup: сколько эпох spatial backbone заморожен.
    # Оставляем для стабильного старта.
    warmup_epochs: int = 5

    # Сколько последних блоков spatial backbone размораживать после warmup.
    # 2 блока — компромисс: достаточно для адаптации, стабильно на MPS.
    # Если на MPS возникают зависания, можно вернуть к 0.
    unfreeze_last_n_blocks: int = 2

    # Главная метрика выбора лучшей модели
    primary_metric: str = "auc"

    # Дополнительные безопасные флаги под class imbalance.
    # По умолчанию НЕ включаем:
    # - video-level DFDC после очистки близок к сбалансированному;
    # - включать компенсацию дисбаланса нужно только после проверки train.py.
    use_class_weights: bool = False
    use_weighted_sampler: bool = False

    # Явный pos_weight для BCEWithLogitsLoss, если понадобится.
    # None = вычисляется в train.py либо не используется.
    pos_weight: Optional[float] = None

    # =========================================================================
    # AUGMENTATION
    # =========================================================================
    # Эти параметры описывают намерение аугментации.
    # Реализация в dataset/train должна учитывать clip-consistent логику,
    # если аугментация применяется ко всем кадрам клипа одинаково.

    augment_hflip: bool = True
    augment_brightness: float = 0.1
    augment_contrast: float = 0.1

    # JPEG augmentation
    augment_jpeg_prob: float = 0.3
    augment_jpeg_quality_min: int = 70
    augment_jpeg_quality_max: int = 95

    # Если True, одна и та же JPEG-аугментация должна применяться
    # ко всем кадрам одного клипа.
    clip_consistent_jpeg: bool = True

    # =========================================================================
    # EVALUATION / INFERENCE
    # =========================================================================
    # Порог классификации по умолчанию.
    # Это стартовое значение. Финальный рабочий порог можно уточнить после eval.
    decision_threshold: float = 0.5

    # Сохранять ли детальные предсказания на уровне видео
    save_predictions: bool = True

    # Сохранять ли fusion weights / alpha для full-модели
    save_fusion_weights: bool = True

    # Сохранять ли ROC/PR-артефакты, если это поддержано в evaluate.py
    save_eval_curves: bool = True

    # =========================================================================
    # VALIDATION
    # =========================================================================
    def validate(self) -> None:
        """Проверка базовой корректности конфигурации."""
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Сумма train/val/test ratio должна быть 1.0, сейчас: {ratio_sum:.6f}"
            )

        if self.num_frames < 2:
            raise ValueError("num_frames должен быть >= 2 для video-level анализа.")

        if self.min_frames_per_video < self.num_frames:
            raise ValueError(
                "min_frames_per_video должен быть >= num_frames, "
                "иначе часть видео не сможет сформировать клип."
            )

        if self.spatial_size <= 0 or self.temporal_size <= 0:
            raise ValueError("spatial_size и temporal_size должны быть > 0.")

        if self.batch_size <= 0:
            raise ValueError("batch_size должен быть > 0.")

        if self.max_epochs <= 0:
            raise ValueError("max_epochs должен быть > 0.")

        if self.patience <= 0:
            raise ValueError("patience должен быть > 0.")

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs не может быть < 0.")

        if self.unfreeze_last_n_blocks < 0:
            raise ValueError("unfreeze_last_n_blocks не может быть < 0.")

        if self.lr_backbone <= 0 or self.lr_head <= 0:
            raise ValueError("lr_backbone и lr_head должны быть > 0.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay не может быть < 0.")

        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm должен быть > 0.")

        if not (0.0 <= self.head_dropout < 1.0):
            raise ValueError("head_dropout должен быть в диапазоне [0, 1).")

        if not (0.0 <= self.transformer_dropout < 1.0):
            raise ValueError("transformer_dropout должен быть в диапазоне [0, 1).")

        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError("decision_threshold должен быть в диапазоне [0, 1].")

        if not (0.0 <= self.augment_brightness <= 1.0):
            raise ValueError("augment_brightness должен быть в диапазоне [0, 1].")

        if not (0.0 <= self.augment_contrast <= 1.0):
            raise ValueError("augment_contrast должен быть в диапазоне [0, 1].")

        if not (0.0 <= self.augment_jpeg_prob <= 1.0):
            raise ValueError("augment_jpeg_prob должен быть в диапазоне [0, 1].")

        if self.augment_jpeg_quality_min <= 0 or self.augment_jpeg_quality_max > 100:
            raise ValueError("JPEG quality должна быть в диапазоне 1..100.")

        if self.augment_jpeg_quality_min > self.augment_jpeg_quality_max:
            raise ValueError(
                "augment_jpeg_quality_min не может быть больше augment_jpeg_quality_max."
            )

        if self.pos_weight is not None and self.pos_weight <= 0:
            raise ValueError("pos_weight должен быть > 0, если задан.")

        if self.use_class_weights and self.use_weighted_sampler:
            raise ValueError(
                "Не включайте одновременно use_class_weights и use_weighted_sampler "
                "без явного осознанного решения в train.py."
            )

    # =========================================================================
    # PATH HELPERS
    # =========================================================================
    def experiment_name(self) -> str:
        """
        Имя эксперимента для логирования и структуры артефактов.
        Пример:
            dfdc02_full_seed42_bs8_T16_adaptive
        """
        parts = [
            self.dataset_name,
            self.model_type,
            f"seed{self.seed}",
            f"bs{self.batch_size}",
            f"T{self.num_frames}",
        ]

        if self.model_type == "full":
            parts.append(self.fusion_type)

        return "_".join(parts)

    def experiment_dir(self) -> str:
        """Директория конкретного эксперимента."""
        path = os.path.join(self.output_dir, self.experiment_name())
        os.makedirs(path, exist_ok=True)
        return path

    def checkpoint_path(self) -> str:
        """Путь к лучшему чекпоинту."""
        return os.path.join(self.experiment_dir(), "best_model.pt")

    def metrics_path(self) -> str:
        """Путь к основным метрикам эксперимента."""
        return os.path.join(self.experiment_dir(), "metrics.json")

    def predictions_path(self) -> str:
        """Путь к детальным предсказаниям."""
        return os.path.join(self.experiment_dir(), "predictions.csv")

    def fusion_weights_path(self) -> str:
        """Путь к сохранению fusion weights / alpha."""
        return os.path.join(self.experiment_dir(), "fusion_weights.json")

    def log_path(self) -> str:
        """Путь к логу обучения/оценки."""
        return os.path.join(self.experiment_dir(), "training.log")

    def split_path(self) -> str:
        """Путь к файлу сохранённого split."""
        os.makedirs(self.splits_dir, exist_ok=True)
        return os.path.join(self.splits_dir, self.split_filename)

    def artifacts_path(self) -> str:
        """Каталог общих артефактов."""
        os.makedirs(self.artifacts_dir, exist_ok=True)
        return self.artifacts_dir