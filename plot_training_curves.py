"""
Визуализация кривых обучения из metrics.json.

Строит графики:
- Train / Val Loss по эпохам
- Val AUC по эпохам
- Сравнительная таблица Test-метрик всех экспериментов

Использование:
    python plot_training_curves.py --experiments_dir ./experiments
    python plot_training_curves.py --experiments_dir ./experiments --output_dir ./experiments/plots
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


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
        except (json.JSONDecodeError, OSError):
            continue
    return results


def plot_loss_curves(results: List[Dict], output_dir: str) -> None:
    """Train/Val loss по эпохам для каждого эксперимента."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4), squeeze=False)

    for i, data in enumerate(results):
        ax = axes[0][i]
        history = data.get("history", {})
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        epochs = list(range(1, len(train_loss) + 1))

        if train_loss:
            ax.plot(epochs, train_loss, "o-", label="Train Loss", markersize=3)
        if val_loss:
            ax.plot(epochs, val_loss, "s-", label="Val Loss", markersize=3)

        name = data.get("experiment", data.get("_dir_name", f"exp_{i}"))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training / Validation Loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/loss_curves.png")


def plot_auc_curves(results: List[Dict], output_dir: str) -> None:
    """Val AUC по эпохам (все эксперименты на одном графике)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for data in results:
        history = data.get("history", {})
        val_auc = history.get("val_auc", [])
        if not val_auc:
            continue
        epochs = list(range(1, len(val_auc) + 1))
        name = data.get("experiment", data.get("_dir_name", "?"))
        ax.plot(epochs, val_auc, "o-", label=name, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUC")
    ax.set_title("Validation AUC per Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "val_auc_curves.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/val_auc_curves.png")


def plot_test_comparison(results: List[Dict], output_dir: str) -> None:
    """Сравнительная барная диаграмма Test-метрик."""
    names = []
    aucs = []
    accs = []
    f1s = []

    for data in results:
        test = data.get("test", {})
        if not test:
            continue
        name = data.get("experiment", data.get("_dir_name", "?"))
        names.append(name.replace("dfdc02_", "").replace("_seed42_bs8_T16", ""))
        aucs.append(test.get("auc", 0))
        accs.append(test.get("accuracy", 0))
        f1s.append(test.get("f1", 0))

    if not names:
        return

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))
    ax.bar(x - width, aucs, width, label="AUC")
    ax.bar(x, accs, width, label="Accuracy")
    ax.bar(x + width, f1s, width, label="F1")

    ax.set_ylabel("Score")
    ax.set_title("Test Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "test_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir}/test_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Визуализация кривых обучения")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="./experiments",
        help="Директория с результатами экспериментов",
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

    print(f"Найдено {len(results)} экспериментов")

    plot_loss_curves(results, args.output_dir)
    plot_auc_curves(results, args.output_dir)
    plot_test_comparison(results, args.output_dir)

    print(f"\nВсе графики сохранены в: {args.output_dir}")


if __name__ == "__main__":
    main()
