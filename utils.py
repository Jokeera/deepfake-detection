"""
Вспомогательные функции:
- воспроизводимость,
- вычисление метрик,
- логирование,
- сохранение/загрузка JSON,
- таймер.

Версия аккуратнее для macOS / Linux / Windows.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
)


# =============================================================================
# ВОСПРОИЗВОДИМОСТЬ
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Фиксирует random seed для воспроизводимости.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# МЕТРИКИ
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Вычисляет метрики бинарной классификации.

    Args:
        y_true:   истинные метки {0, 1}, shape [N]
        y_proba:  вероятности положительного класса, shape [N]
        threshold: порог бинаризации

    Returns:
        dict:
            auc
            accuracy
            f1
            eer
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if y_true.ndim != 1 or y_proba.ndim != 1:
        raise ValueError(
            f"Ожидаются одномерные массивы, получено: "
            f"y_true={y_true.shape}, y_proba={y_proba.shape}"
        )

    if len(y_true) != len(y_proba):
        raise ValueError(
            f"Размеры y_true и y_proba не совпадают: "
            f"{len(y_true)} vs {len(y_proba)}"
        )

    if len(y_true) == 0:
        raise ValueError("Пустые массивы для compute_metrics().")

    y_pred = (y_proba >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = 0.0

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        ap = float(average_precision_score(y_true, y_proba))
    except ValueError:
        ap = 0.0

    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    except Exception:
        eer = 1.0

    return {
        "auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "ap": round(ap, 4),
        "f1": round(f1, 4),
        "eer": round(eer, 4),
    }


# =============================================================================
# ЛОГИРОВАНИЕ
# =============================================================================

def setup_logger(log_path: str, name: str = "deepfake") -> logging.Logger:
    """
    Настраивает логгер: файл + консоль.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Удаляем старые handlers, чтобы не было дублей
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# JSON IO
# =============================================================================

def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """
    Сохраняет словарь метрик в JSON.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics(path: str) -> Dict[str, Any]:
    """
    Загружает словарь метрик из JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# ТАЙМЕР
# =============================================================================

class Timer:
    """
    Контекстный менеджер для замера времени.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.start: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start is None:
            self.elapsed = 0.0
        else:
            self.elapsed = time.time() - self.start

    def __str__(self) -> str:
        prefix = f"{self.name}: " if self.name else ""
        return f"{prefix}{self.elapsed:.2f}s"