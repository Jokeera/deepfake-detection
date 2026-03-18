"""
Оценка обученной модели на val / test / all.

Поддержка:
- macOS / Linux / Windows
- CPU / CUDA / MPS
- full / spatial / temporal / sequential
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from config import Config
from dataset import DeepfakeVideoDataset, build_video_index, load_split, split_index
from models import build_model
from utils import compute_metrics, save_metrics, set_seed


def get_device(device_arg: str = "auto") -> torch.device:
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
    return "cuda" if device.type == "cuda" else "cpu"


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_labels = []
    all_proba = []
    all_video_ids = []
    all_fusion_weights = []

    total_loss = 0.0
    num_batches = 0

    amp_enabled = cfg.use_amp and device.type == "cuda"
    amp_device_type = get_amp_device_type(device)

    for batch in loader:
        spatial = batch["spatial"].to(device)
        temporal = batch["temporal"].to(device)
        labels = batch["label"].to(device)
        video_ids = batch["video_id"]

        with autocast(device_type=amp_device_type, enabled=amp_enabled):
            logits, alpha = model(spatial, temporal)
            loss = criterion(logits, labels)

        proba = torch.sigmoid(logits).detach().cpu().numpy()

        total_loss += float(loss.item())
        num_batches += 1

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_proba.extend(proba.tolist())
        all_video_ids.extend(video_ids)

        if alpha is not None:
            alpha_np = alpha.detach().cpu().numpy()
            for i, vid in enumerate(video_ids):
                all_fusion_weights.append(
                    {
                        "video_id": vid,
                        "alpha_spatial": float(alpha_np[i, 0]),
                        "alpha_temporal": float(alpha_np[i, 1]),
                    }
                )

    y_true = np.array(all_labels)
    y_proba = np.array(all_proba)

    metrics = compute_metrics(
        y_true=y_true,
        y_proba=y_proba,
        threshold=cfg.decision_threshold,
    )
    metrics["loss"] = round(total_loss / max(num_batches, 1), 4)

    predictions = []
    for vid, label, prob in zip(all_video_ids, all_labels, all_proba):
        predictions.append(
            {
                "video_id": vid,
                "label": int(label),
                "proba": round(float(prob), 4),
                "correct": int((float(prob) >= cfg.decision_threshold) == int(label)),
            }
        )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "fusion_weights": all_fusion_weights,
        "num_videos": len(all_labels),
    }


def build_eval_loader(cfg: Config, split_name: str) -> DataLoader:
    index = build_video_index(cfg.dataset_root)

    if cfg.split_mode == "fixed":
        train_idx, val_idx, test_idx = load_split(cfg, index)
    else:
        train_idx, val_idx, test_idx = split_index(
            index=index,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )

    if split_name == "test":
        eval_idx = test_idx
    elif split_name == "val":
        eval_idx = val_idx
    elif split_name == "all":
        eval_idx = index
    else:
        raise ValueError(f"Неизвестный split: {split_name}")

    eval_ds = DeepfakeVideoDataset(eval_idx, cfg, is_train=False)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    return eval_loader


def measure_inference_ms(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    sample = next(iter(loader))
    spatial_sample = sample["spatial"][:1].to(device)
    temporal_sample = sample["temporal"][:1].to(device)

    model.eval()

    with torch.no_grad():
        for _ in range(3):
            model(spatial_sample, temporal_sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    n_runs = 20

    with torch.no_grad():
        for _ in range(n_runs):
            model(spatial_sample, temporal_sample)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t1 = time.time()
    return (t1 - t0) / n_runs * 1000.0


def save_confusion_matrix(y_true, y_pred, output_path: str) -> None:
    """Сохраняет confusion matrix как PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {output_path}")


def save_roc_curve(y_true, y_proba, output_path: str) -> None:
    """Сохраняет ROC-кривую как PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка модели deepfake detection")

    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к best_model.pt")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Путь к preprocessed dataset (override checkpoint config)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Имя датасета для логов/результатов",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test", "all"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения eval json",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Отключить mixed precision",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Переопределить число worker'ов",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Включить pin_memory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    saved_cfg = checkpoint.get("config", {})
    cfg = Config()

    for key, value in saved_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if args.dataset_root is not None:
        cfg.dataset_root = args.dataset_root

    if args.dataset_name is not None:
        cfg.dataset_name = args.dataset_name

    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    cfg.pin_memory = args.pin_memory
    cfg.device = args.device
    cfg.use_amp = not args.no_amp

    cfg.validate()
    set_seed(cfg.seed)

    eval_loader = build_eval_loader(cfg, args.split)
    print(f"Оценка на split='{args.split}': {len(eval_loader.dataset)} видео")

    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)

    print(
        f"Модель загружена: {cfg.model_type}, "
        f"dataset={cfg.dataset_name}, "
        f"epoch={checkpoint.get('epoch', '?')}, "
        f"device={device}"
    )

    results = evaluate_model(
        model=model,
        loader=eval_loader,
        device=device,
        cfg=cfg,
    )

    m = results["metrics"]
    print("\n" + "=" * 60)
    print(f"Результаты ({args.split}, {results['num_videos']} видео):")
    print(f"  AUC:      {m['auc']:.4f}")
    print(f"  Accuracy: {m['accuracy']:.4f}")
    print(f"  F1:       {m['f1']:.4f}")
    print(f"  EER:      {m['eer']:.4f}")
    print(f"  Loss:     {m['loss']:.4f}")

    if results["num_videos"] > 0:
        ms_per_video = measure_inference_ms(model, eval_loader, device)
        print(f"  Inference: {ms_per_video:.1f} ms/video")
        results["metrics"]["inference_ms"] = round(ms_per_video, 1)

    preds = results["predictions"]
    fn = [p for p in preds if p["label"] == 1 and p["proba"] < cfg.decision_threshold]
    fp = [p for p in preds if p["label"] == 0 and p["proba"] >= cfg.decision_threshold]
    print(f"\nОшибки: {len(fn)} FN, {len(fp)} FP")

    # Сохраняем confusion matrix и ROC curve
    y_true = np.array([p["label"] for p in preds])
    y_proba = np.array([p["proba"] for p in preds])
    y_pred = (y_proba >= cfg.decision_threshold).astype(int)

    exp_dir = os.path.dirname(args.checkpoint)
    save_confusion_matrix(
        y_true, y_pred,
        os.path.join(exp_dir, f"confusion_matrix_{args.split}.png"),
    )
    save_roc_curve(
        y_true, y_proba,
        os.path.join(exp_dir, f"roc_curve_{args.split}.png"),
    )

    if results["fusion_weights"]:
        alphas_s = [fw["alpha_spatial"] for fw in results["fusion_weights"]]
        alphas_t = [fw["alpha_temporal"] for fw in results["fusion_weights"]]
        print(
            f"Fusion weights: "
            f"spatial={np.mean(alphas_s):.3f}±{np.std(alphas_s):.3f}, "
            f"temporal={np.mean(alphas_t):.3f}±{np.std(alphas_t):.3f}"
        )

    print("=" * 60)

    if args.output is None:
        exp_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(exp_dir, f"eval_{args.split}.json")

    save_metrics(results, args.output)
    print(f"\nРезультаты сохранены: {args.output}")


if __name__ == "__main__":
    main()