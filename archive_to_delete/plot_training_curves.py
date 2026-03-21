"""
Визуализация результатов экспериментов из metrics.json.

Строит графики, основанные ТОЛЬКО на реальных данных обучения:
1. Train/Val Loss по эпохам (каждая модель)
2. Val AUC по эпохам (все модели на одном графике)
3. Ablation study — сравнение Test-метрик 4 архитектур
4. Overfit analysis — gap между train loss и val loss
5. Val EER по эпохам (все модели)
6. Convergence — best epoch + training duration

Использование:
    python plot_training_curves.py --experiments_dir ./experiments
    python plot_training_curves.py --experiments_dir kaggle_output/experiments/experiments
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Настройки визуализации ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Цвета для 4 моделей (согласованы по всем графикам)
MODEL_COLORS = {
    "full": "#2196F3",      # синий
    "spatial": "#4CAF50",   # зелёный
    "temporal": "#FF9800",  # оранжевый
    "sequential": "#9C27B0",  # фиолетовый
}

MODEL_LABELS = {
    "full": "A1: Full (dual-path)",
    "spatial": "A2: Spatial-only",
    "temporal": "A3: Temporal-only",
    "sequential": "A4: Sequential (BiLSTM)",
}


def load_experiment_metrics(experiments_dir: str) -> List[Dict]:
    """Загружает metrics.json из каждого подкаталога экспериментов."""
    results = []
    for entry in sorted(os.listdir(experiments_dir)):
        metrics_path = os.path.join(experiments_dir, entry, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_dir_name"] = entry
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Пропуск {metrics_path}: {e}")
            continue
    return results


def _clean_series(values: List[float]) -> np.ndarray:
    """Заменяет NaN и аномальные значения (0.0 в AUC) на np.nan для корректного отображения."""
    arr = np.array(values, dtype=float)
    arr[np.isnan(arr)] = np.nan
    return arr


def _get_model_type(data: Dict) -> str:
    return data.get("model_type", "unknown")


def _get_label(data: Dict) -> str:
    mt = _get_model_type(data)
    return MODEL_LABELS.get(mt, data.get("experiment", data.get("_dir_name", "?")))


def _get_color(data: Dict) -> str:
    mt = _get_model_type(data)
    return MODEL_COLORS.get(mt, "#607D8B")


# ── Граф 1: Train/Val Loss ────────────────────────────────────────────
def plot_loss_curves(results: List[Dict], output_dir: str) -> None:
    """Train/Val loss по эпохам для каждого эксперимента (subplots)."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)

    for i, data in enumerate(results):
        ax = axes[0][i]
        history = data.get("history", {})
        train_loss = _clean_series(history.get("train_loss", []))
        val_loss = _clean_series(history.get("val_loss", []))
        epochs = np.arange(1, len(train_loss) + 1)

        color = _get_color(data)
        if len(train_loss) > 0:
            ax.plot(epochs, train_loss, "o-", color=color, alpha=0.7,
                    label="Train Loss", markersize=3, linewidth=1.5)
        if len(val_loss) > 0:
            ax.plot(epochs, val_loss, "s--", color=color, alpha=0.9,
                    label="Val Loss", markersize=3, linewidth=1.5)

        best_epoch = data.get("best_epoch")
        if best_epoch is not None and best_epoch <= len(val_loss):
            ax.axvline(x=best_epoch, color="red", linestyle=":", alpha=0.5,
                       label=f"Best epoch={best_epoch}")

        ax.set_title(_get_label(data), fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)

    fig.suptitle("Кривые обучения: Train / Validation Loss", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "01_loss_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Граф 2: Val AUC ───────────────────────────────────────────────────
def plot_val_auc_curves(results: List[Dict], output_dir: str) -> None:
    """Val AUC по эпохам — все модели на одном графике."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for data in results:
        history = data.get("history", {})
        val_auc = _clean_series(history.get("val_auc", []))
        # Заменяем аномальные 0.0 на NaN (AUC=0 — артефакт NaN-эпохи)
        val_auc[val_auc == 0.0] = np.nan

        if len(val_auc) == 0:
            continue
        epochs = np.arange(1, len(val_auc) + 1)
        label = _get_label(data)
        color = _get_color(data)

        ax.plot(epochs, val_auc, "o-", label=label, color=color,
                markersize=4, linewidth=2)

        best_epoch = data.get("best_epoch")
        best_val = data.get("best_primary_metric_value")
        if best_epoch and best_val:
            ax.scatter([best_epoch], [best_val], color=color,
                       s=120, zorder=5, edgecolors="black", linewidths=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC по эпохам (все модели)")
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    path = os.path.join(output_dir, "02_val_auc_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Граф 3: Ablation Study — Test Metrics ─────────────────────────────
def plot_ablation_study(results: List[Dict], output_dir: str) -> None:
    """Сравнительная столбчатая диаграмма Test-метрик (ablation study)."""
    names = []
    aucs, accs, f1s, eers = [], [], [], []

    for data in results:
        test = data.get("test", {})
        if not test:
            continue
        names.append(_get_label(data))
        aucs.append(test.get("auc", 0))
        accs.append(test.get("accuracy", 0))
        f1s.append(test.get("f1", 0))
        eers.append(test.get("eer", 0))

    if not names:
        return

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_auc = ax.bar(x - 1.5 * width, aucs, width, label="AUC", color="#2196F3")
    bars_acc = ax.bar(x - 0.5 * width, accs, width, label="Accuracy", color="#4CAF50")
    bars_f1 = ax.bar(x + 0.5 * width, f1s, width, label="F1-score", color="#FF9800")
    bars_eer = ax.bar(x + 1.5 * width, eers, width, label="EER", color="#F44336")

    # Подписи значений
    for bars in [bars_auc, bars_acc, bars_f1, bars_eer]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Сравнение Test-метрик 4 архитектур")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    path = os.path.join(output_dir, "03_ablation_study.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Граф 4: Overfit Analysis ──────────────────────────────────────────
def plot_overfit_analysis(results: List[Dict], output_dir: str) -> None:
    """Разница (val_loss - train_loss) по эпохам — индикатор переобучения."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for data in results:
        history = data.get("history", {})
        train_loss = _clean_series(history.get("train_loss", []))
        val_loss = _clean_series(history.get("val_loss", []))

        n = min(len(train_loss), len(val_loss))
        if n == 0:
            continue

        gap = val_loss[:n] - train_loss[:n]
        epochs = np.arange(1, n + 1)
        label = _get_label(data)
        color = _get_color(data)

        ax.plot(epochs, gap, "o-", label=label, color=color,
                markersize=3, linewidth=1.5)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.8)
    ax.fill_between([0, 35], 0, 0.3, alpha=0.05, color="red",
                    label="Зона переобучения (gap > 0)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss − Train Loss")
    ax.set_title("Overfit Analysis: разрыв Val/Train Loss по эпохам")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(output_dir, "04_overfit_analysis.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Граф 5: Val EER ───────────────────────────────────────────────────
def plot_val_eer_curves(results: List[Dict], output_dir: str) -> None:
    """Val EER по эпохам — все модели."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for data in results:
        history = data.get("history", {})
        val_eer = _clean_series(history.get("val_eer", []))
        # EER=1.0 — артефакт NaN-эпохи
        val_eer[val_eer >= 1.0] = np.nan

        if len(val_eer) == 0:
            continue
        epochs = np.arange(1, len(val_eer) + 1)
        label = _get_label(data)
        color = _get_color(data)

        ax.plot(epochs, val_eer, "o-", label=label, color=color,
                markersize=4, linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation EER")
    ax.set_title("Validation EER по эпохам (все модели)")
    ax.legend(loc="upper right")
    ax.invert_yaxis()  # Меньше EER = лучше — вверху
    fig.tight_layout()
    path = os.path.join(output_dir, "05_val_eer_curves.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Граф 6: Convergence & Best Epoch ──────────────────────────────────
def plot_convergence(results: List[Dict], output_dir: str) -> None:
    """Best epoch и количество эпох до сходимости для каждой модели."""
    names = []
    best_epochs = []
    total_epochs = []
    colors = []

    for data in results:
        test = data.get("test", {})
        if not test:
            continue
        names.append(_get_label(data))
        best_epochs.append(data.get("best_epoch", 0))
        history = data.get("history", {})
        total_epochs.append(len(history.get("train_loss", [])))
        colors.append(_get_color(data))

    if not names:
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, total_epochs, width, label="Всего эпох", alpha=0.4, color=colors)
    ax.bar(x + width / 2, best_epochs, width, label="Best epoch", color=colors)

    for i, (te, be) in enumerate(zip(total_epochs, best_epochs)):
        ax.annotate(f"{be}", xy=(x[i] + width / 2, be),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Epoch")
    ax.set_title("Сходимость моделей: best epoch vs общее число эпох")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "06_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Summary Table (текстовый файл) ────────────────────────────────────
def generate_summary(results: List[Dict], output_dir: str) -> None:
    """Генерирует актуальную сводную таблицу из metrics.json."""
    lines = []
    lines.append("=" * 100)
    lines.append("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (сгенерировано из metrics.json)")
    lines.append("=" * 100)
    header = (
        f"{'Модель':<30} {'Test AUC':>10} {'Test Acc':>10} "
        f"{'Test F1':>10} {'Test EER':>10} {'Best Ep':>8} {'Эпох':>6}"
    )
    lines.append(header)
    lines.append("-" * 100)

    for data in results:
        test = data.get("test", {})
        if not test:
            continue
        label = _get_label(data)
        auc = test.get("auc", 0)
        acc = test.get("accuracy", 0)
        f1 = test.get("f1", 0)
        eer = test.get("eer", 0)
        best_ep = data.get("best_epoch", "?")
        total_ep = len(data.get("history", {}).get("train_loss", []))
        lines.append(
            f"{label:<30} {auc:>10.4f} {acc:>10.4f} "
            f"{f1:>10.4f} {eer:>10.4f} {str(best_ep):>8} {total_ep:>6}"
        )

    lines.append("=" * 100)

    # Определяем лучшую модель
    best = max(results, key=lambda d: d.get("test", {}).get("auc", 0))
    best_label = _get_label(best)
    best_auc = best.get("test", {}).get("auc", 0)
    lines.append(f"\nЛучшая модель по AUC: {best_label} (AUC = {best_auc:.4f})")

    table_text = "\n".join(lines)

    path = os.path.join(output_dir, "results_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(table_text)
    print(f"  Saved: {path}")
    print("\n" + table_text)


def main():
    parser = argparse.ArgumentParser(
        description="Визуализация кривых обучения из metrics.json"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="./experiments",
        help="Директория с подкаталогами экспериментов (каждый содержит metrics.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Директория для сохранения графиков (по умолчанию: experiments_dir/plots)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiments_dir, "plots")

    os.makedirs(args.output_dir, exist_ok=True)

    results = load_experiment_metrics(args.experiments_dir)
    if not results:
        print(f"Не найдены metrics.json в {args.experiments_dir}")
        return

    print(f"Найдено {len(results)} экспериментов:")
    for r in results:
        mt = r.get("model_type", "?")
        test_auc = r.get("test", {}).get("auc", "N/A")
        print(f"  - {_get_label(r)}  |  Test AUC: {test_auc}")

    print(f"\nГенерация графиков в: {args.output_dir}")
    plot_loss_curves(results, args.output_dir)
    plot_val_auc_curves(results, args.output_dir)
    plot_ablation_study(results, args.output_dir)
    plot_overfit_analysis(results, args.output_dir)
    plot_val_eer_curves(results, args.output_dir)
    plot_convergence(results, args.output_dir)
    generate_summary(results, args.output_dir)

    print(f"\nВсе графики сохранены в: {args.output_dir}")


if __name__ == "__main__":
    main()
