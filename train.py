"""
Обучение модели deepfake detection.
Полный цикл: warmup -> fine-tuning (опционально) -> early stopping ->
лучший чекпоинт -> финальная оценка на test.

Поддержка:
- macOS / Linux / Windows
- CPU / CUDA / MPS
- full / spatial / temporal / sequential

Финальные принципы для ВКР:
1. Главная метрика выбора лучшей модели задаётся через cfg.primary_metric.
2. Fine-tuning spatial backbone может быть полностью отключён
   (unfreeze_last_n_blocks = 0) ради стабильности среды.
3. Поддерживается pos_weight для BCEWithLogitsLoss, если он задан в config.
4. После выбора лучшего чекпоинта проводится честная оценка на test.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc as sk_auc,
    precision_recall_curve, average_precision_score, precision_score, recall_score,
    f1_score as sk_f1_score,
)

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

try:
    from torch.amp import GradScaler
    _GRADSCALER_MODE = "torch_amp"
except ImportError:
    from torch.cuda.amp import GradScaler
    _GRADSCALER_MODE = "cuda_amp"

from config import Config
from dataset import create_dataloaders
from models import build_model
from utils import compute_metrics, save_metrics, set_seed, setup_logger


def get_device(device_arg: str = "auto") -> torch.device:
    """Выбирает устройство: auto / cuda / mps / cpu."""
    device_arg = device_arg.lower()

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен device='cuda', но CUDA недоступна.")
        return torch.device("cuda")

    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Запрошен device='mps', но MPS недоступен.")
        return torch.device("mps")

    if device_arg != "auto":
        raise ValueError(f"Неизвестный device: {device_arg}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_device_type(device: torch.device) -> str:
    """
    Для torch.amp.autocast нужен device_type строкой.
    AMP включаем только на CUDA.
    """
    return "cuda" if device.type == "cuda" else "cpu"


def split_optimizer_params(model: nn.Module) -> Tuple[list, list]:
    """
    Разделяет параметры на:
    - spatial backbone
    - остальные
    """
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "spatial_branch.backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    return backbone_params, other_params


def build_criterion(cfg: Config, device: torch.device) -> nn.Module:
    """
    Создаёт criterion с поддержкой pos_weight.
    """
    if cfg.pos_weight is not None:
        pos_weight = torch.tensor([cfg.pos_weight], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return nn.BCEWithLogitsLoss()


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    cfg: Config,
    logger=None,
) -> float:
    """Одна эпоха обучения."""
    model.train()

    total_loss = 0.0
    num_batches = 0
    total_batches = len(loader)
    log_every = max(1, total_batches // 10)  # ~10 сообщений за эпоху

    amp_enabled = cfg.use_amp and device.type == "cuda"
    amp_device_type = get_amp_device_type(device)

    for batch in loader:
        spatial = batch["spatial"].to(device)
        temporal = batch["temporal"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits, _ = model(spatial, temporal)
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

        if logger and num_batches % log_every == 0:
            avg_loss = total_loss / num_batches
            logger.info(
                f"  [batch {num_batches}/{total_batches}] "
                f"avg_loss={avg_loss:.4f}"
            )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    return_predictions: bool = False,
) -> Dict[str, float]:
    """Валидация или тест."""
    model.eval()

    total_loss = 0.0
    num_batches = 0
    all_labels: List[float] = []
    all_proba: List[float] = []
    all_video_ids: List[str] = []

    amp_enabled = cfg.use_amp and device.type == "cuda"
    amp_device_type = get_amp_device_type(device)

    for batch in loader:
        spatial = batch["spatial"].to(device)
        temporal = batch["temporal"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits, _ = model(spatial, temporal)
            loss = criterion(logits, labels)

        # Guard against AMP fp16 overflow producing nan
        if not torch.isfinite(loss):
            continue
        proba = torch.sigmoid(logits)
        proba = torch.clamp(proba, 0.0, 1.0)

        total_loss += float(loss.item())
        num_batches += 1

        batch_labels = labels.detach().cpu().numpy().tolist()
        batch_proba = proba.detach().cpu().numpy().tolist()

        all_labels.extend(batch_labels)
        all_proba.extend(batch_proba)

        if return_predictions:
            all_video_ids.extend(batch["video_id"])

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(
        y_true=np.array(all_labels),
        y_proba=np.array(all_proba),
        threshold=cfg.decision_threshold,
    )
    metrics["loss"] = round(avg_loss, 4)

    if return_predictions:
        metrics["_predictions"] = {
            "video_id": all_video_ids,
            "y_true": all_labels,
            "y_proba": all_proba,
        }

    return metrics


def get_primary_metric_value(metrics: Dict[str, float], cfg: Config) -> float:
    """
    Возвращает значение главной метрики из config.
    """
    metric_name = cfg.primary_metric.lower()

    if metric_name not in metrics:
        raise KeyError(
            f"primary_metric='{cfg.primary_metric}' отсутствует в metrics. "
            f"Доступные ключи: {list(metrics.keys())}"
        )

    return float(metrics[metric_name])


def save_predictions_csv(predictions: Dict[str, List], path: str) -> None:
    """
    Сохраняет детальные предсказания на уровне видео.
    """
    rows = zip(
        predictions["video_id"],
        predictions["y_true"],
        predictions["y_proba"],
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "y_true", "y_proba"])
        writer.writerows(rows)


def predictions_csv_path(cfg: Config) -> str:
    """
    Отдельный путь для детальных test-предсказаний.
    """
    base = cfg.predictions_path()
    if base.lower().endswith(".csv"):
        return base
    return f"{base}.csv"


def train(cfg: Config):
    """Полный цикл обучения."""
    cfg.validate()
    set_seed(cfg.seed)

    device = get_device(cfg.device)
    logger = setup_logger(cfg.log_path())

    logger.info("=" * 70)
    logger.info(f"Эксперимент: {cfg.experiment_name()}")
    logger.info(f"Dataset: {cfg.dataset_name}")
    logger.info(f"Model: {cfg.model_type}")
    logger.info(f"Fusion: {cfg.fusion_type}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Epochs: {cfg.max_epochs}")
    logger.info(f"Primary metric: {cfg.primary_metric}")
    logger.info("=" * 70)

    logger.info("Загрузка данных...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    logger.info(
        f"Train: {len(train_loader.dataset)} videos | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    logger.info("Создание модели...")
    model = build_model(cfg).to(device)

    if hasattr(model, "freeze_spatial_backbone"):
        model.freeze_spatial_backbone()
        logger.info(
            f"Warmup: spatial backbone заморожен "
            f"(эпохи 1..{cfg.warmup_epochs})"
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Параметры: всего={total_params:,}, обучаемых={trainable_params:,}"
    )

    backbone_params, other_params = split_optimizer_params(model)

    logger.info(
        f"Optimizer groups: backbone={len(backbone_params)} tensors, "
        f"other={len(other_params)} tensors"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": other_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )

    # Warmup: линейный рост LR от 0 до target за warmup_epochs,
    # затем CosineAnnealing на оставшиеся эпохи.
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(cfg.warmup_epochs, 1),
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(cfg.max_epochs - cfg.warmup_epochs, 1),
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )

    criterion = build_criterion(cfg, device)

    amp_enabled = cfg.use_amp and device.type == "cuda"
    if _GRADSCALER_MODE == "torch_amp":
        scaler = GradScaler(device="cuda", enabled=amp_enabled)
    else:
        scaler = GradScaler(enabled=amp_enabled)

    best_primary_metric = -1.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_eer": [],
    }

    for epoch in range(1, cfg.max_epochs + 1):
        epoch_start = time.time()

        if (
            epoch == cfg.warmup_epochs + 1
            and cfg.unfreeze_last_n_blocks > 0
            and hasattr(model, "unfreeze_spatial_backbone")
        ):
            model.unfreeze_spatial_backbone()

            # Добавляем размороженные параметры backbone в оптимизатор
            newly_unfrozen = [
                p for name, p in model.named_parameters()
                if "spatial_branch.backbone" in name and p.requires_grad
                and not any(p is existing for group in optimizer.param_groups
                            for existing in group["params"])
            ]
            if newly_unfrozen:
                optimizer.add_param_group({
                    "params": newly_unfrozen,
                    "lr": cfg.lr_backbone,
                })
                logger.info(
                    f"Добавлено {len(newly_unfrozen)} размороженных "
                    f"параметров backbone в оптимизатор"
                )

            logger.info(
                f"Fine-tuning: разморожены последние "
                f"{cfg.unfreeze_last_n_blocks} блока spatial backbone"
            )
        elif epoch == cfg.warmup_epochs + 1 and cfg.unfreeze_last_n_blocks == 0:
            logger.info("Fine-tuning пропущен: unfreeze_last_n_blocks = 0")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            cfg=cfg,
            logger=logger,
        )

        val_metrics = evaluate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        logger.info(
            f"Epoch {epoch:02d}/{cfg.max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"LR backbone: {current_lr_backbone:.6f} | "
            f"LR head: {current_lr_head:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(val_metrics["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_eer"].append(val_metrics["eer"])

        current_primary_metric = get_primary_metric_value(val_metrics, cfg)
        is_best = current_primary_metric > best_primary_metric

        # Append to history.csv (incremental, survives crashes)
        history_csv_path = os.path.join(cfg.experiment_dir(), "history.csv")
        write_header = not os.path.exists(history_csv_path)
        with open(history_csv_path, "a", newline="", encoding="utf-8") as hf:
            w = csv.writer(hf)
            if write_header:
                w.writerow([
                    "epoch", "train_loss", "val_loss", "val_auc",
                    "val_acc", "val_f1", "val_eer",
                    "lr_backbone", "lr_head", "epoch_time_sec", "is_best",
                ])
            w.writerow([
                epoch, round(train_loss, 4), val_metrics["loss"],
                val_metrics["auc"], val_metrics["accuracy"],
                val_metrics["f1"], val_metrics["eer"],
                f"{current_lr_backbone:.8f}", f"{current_lr_head:.8f}",
                round(epoch_time, 1), int(is_best),
            ])

        if current_primary_metric > best_primary_metric:
            best_primary_metric = current_primary_metric
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "primary_metric_name": cfg.primary_metric,
                    "primary_metric_value": best_primary_metric,
                    "val_metrics": val_metrics,
                    "config": vars(cfg),
                },
                cfg.checkpoint_path(),
            )
            logger.info(
                f"Лучшая модель сохранена: "
                f"{cfg.primary_metric} = {best_primary_metric:.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    f"Early stopping на эпохе {epoch}. "
                    f"Лучший {cfg.primary_metric} = {best_primary_metric:.4f} "
                    f"(epoch {best_epoch})"
                )
                break

    logger.info("=" * 70)
    logger.info("ТЕСТИРОВАНИЕ ЛУЧШЕГО ЧЕКПОИНТА")

    checkpoint = torch.load(
        cfg.checkpoint_path(),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    logger.info(
        f"Загружен чекпоинт эпохи {checkpoint['epoch']} "
        f"({checkpoint['primary_metric_name']} = "
        f"{checkpoint['primary_metric_value']:.4f})"
    )

    test_metrics = evaluate_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        cfg=cfg,
        return_predictions=cfg.save_predictions,
    )

    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test Acc: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 : {test_metrics['f1']:.4f}")
    logger.info(f"Test EER: {test_metrics['eer']:.4f}")

    if cfg.save_predictions and "_predictions" in test_metrics:
        preds = test_metrics.pop("_predictions")
        save_predictions_csv(preds, predictions_csv_path(cfg))
        logger.info(f"Предсказания сохранены: {predictions_csv_path(cfg)}")

        # Confusion matrix и ROC curve
        try:
            y_true = np.array(preds["y_true"])
            y_proba = np.array(preds["y_proba"])
            y_pred = (y_proba >= cfg.decision_threshold).astype(int)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
            disp.plot(ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Confusion Matrix — {cfg.experiment_name()}")
            fig.tight_layout()
            cm_path = os.path.join(cfg.experiment_dir(), "confusion_matrix_test.png")
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            logger.info(f"Confusion matrix: {cm_path}")

            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc_val = sk_auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC={roc_auc_val:.4f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {cfg.experiment_name()}")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            roc_path = os.path.join(cfg.experiment_dir(), "roc_curve_test.png")
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            logger.info(f"ROC curve: {roc_path}")

            # Precision-Recall curve
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec_arr, prec_arr, "g-", linewidth=2, label=f"PR (AP={ap:.4f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve — {cfg.experiment_name()}")
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            pr_path = os.path.join(cfg.experiment_dir(), "pr_curve_test.png")
            fig.savefig(pr_path, dpi=150)
            plt.close(fig)
            logger.info(f"PR curve: {pr_path}")

            # Threshold sweep: F1, Precision, Recall vs threshold
            thresholds = np.linspace(0.05, 0.95, 50)
            f1s, precs, recs = [], [], []
            for t in thresholds:
                yp = (y_proba >= t).astype(int)
                f1s.append(sk_f1_score(y_true, yp, zero_division=0))
                precs.append(precision_score(y_true, yp, zero_division=0))
                recs.append(recall_score(y_true, yp, zero_division=0))
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(thresholds, f1s, "b-", linewidth=2, label="F1")
            ax.plot(thresholds, precs, "r--", linewidth=1.5, label="Precision")
            ax.plot(thresholds, recs, "g--", linewidth=1.5, label="Recall")
            best_t_idx = int(np.argmax(f1s))
            ax.axvline(thresholds[best_t_idx], color="gray", linestyle=":",
                       label=f"Best F1={f1s[best_t_idx]:.3f} @ t={thresholds[best_t_idx]:.2f}")
            ax.set_xlabel("Threshold")
            ax.set_ylabel("Score")
            ax.set_title(f"Threshold Sweep — {cfg.experiment_name()}")
            ax.legend(loc="center left")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            ts_path = os.path.join(cfg.experiment_dir(), "threshold_sweep_test.png")
            fig.savefig(ts_path, dpi=150)
            plt.close(fig)
            logger.info(f"Threshold sweep: {ts_path}")

            # Score distribution: real vs fake
            fig, ax = plt.subplots(figsize=(7, 5))
            real_scores = y_proba[y_true == 0]
            fake_scores = y_proba[y_true == 1]
            ax.hist(real_scores, bins=30, alpha=0.6, color="green", label=f"Real (n={len(real_scores)})")
            ax.hist(fake_scores, bins=30, alpha=0.6, color="red", label=f"Fake (n={len(fake_scores)})")
            ax.axvline(cfg.decision_threshold, color="black", linestyle="--",
                       label=f"Threshold={cfg.decision_threshold}")
            ax.set_xlabel("Predicted Probability (Fake)")
            ax.set_ylabel("Count")
            ax.set_title(f"Score Distribution — {cfg.experiment_name()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            sd_path = os.path.join(cfg.experiment_dir(), "score_distribution_test.png")
            fig.savefig(sd_path, dpi=150)
            plt.close(fig)
            logger.info(f"Score distribution: {sd_path}")
        except Exception as e:
            logger.warning(f"Не удалось построить графики: {e}")

    all_metrics = {
        "experiment": cfg.experiment_name(),
        "dataset_name": cfg.dataset_name,
        "model_type": cfg.model_type,
        "fusion_type": cfg.fusion_type,
        "seed": cfg.seed,
        "device": str(device),
        "best_epoch": checkpoint["epoch"],
        "best_primary_metric_name": checkpoint["primary_metric_name"],
        "best_primary_metric_value": checkpoint["primary_metric_value"],
        "best_val_metrics": checkpoint["val_metrics"],
        "test": test_metrics,
        "history": history,
    }

    save_metrics(all_metrics, cfg.metrics_path())
    logger.info(f"Метрики сохранены: {cfg.metrics_path()}")
    logger.info("=" * 70)

    return all_metrics



def parse_args():
    parser = argparse.ArgumentParser(description="Обучение модели deepfake detection")

    parser.add_argument(
        "--model_type",
        type=str,
        default="full",
        choices=["full", "spatial", "temporal", "sequential"],
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="adaptive",
        choices=["adaptive", "concat", "gate"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Путь к preprocessed dataset",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="dfdc02",
        help="Имя датасета для логов/артефактов",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Отключить mixed precision",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Включить pin_memory для DataLoader",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config()
    cfg.model_type = args.model_type
    cfg.fusion_type = args.fusion_type
    cfg.seed = args.seed
    cfg.batch_size = args.batch_size
    cfg.max_epochs = args.max_epochs
    cfg.num_frames = args.num_frames
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    cfg.use_amp = not args.no_amp
    cfg.num_workers = args.num_workers
    cfg.pin_memory = args.pin_memory
    cfg.dataset_name = args.dataset_name

    if args.dataset_root is not None:
        cfg.dataset_root = args.dataset_root

    train(cfg)