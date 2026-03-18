#!/usr/bin/env python3
"""
Устойчивый запуск экспериментов с автоматическим resume.

Ключевые особенности:
1. Проверяет, завершён ли эксперимент (есть metrics.json с test результатами).
2. Если эксперимент уже завершён — пропускает его.
3. Если эксперимент частично выполнен (есть checkpoint) — перезапускает train
   (train.py сам перезаписывает checkpoint, так что это безопасно).
4. Каждый эксперимент запускается в отдельном subprocess — если один падает,
   остальные продолжают.
5. После всех экспериментов генерирует графики и сводную таблицу.

Использование:
    python run_resilient.py
    python run_resilient.py --max_epochs 15
    python run_resilient.py --only A1 A3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "A1": {
        "name": "A1_full_model",
        "model_type": "full",
        "fusion_type": "adaptive",
        "description": "Полная модель (dual-path + adaptive fusion)",
    },
    "A2": {
        "name": "A2_spatial_only",
        "model_type": "spatial",
        "fusion_type": "adaptive",
        "description": "Spatial-only (ablation)",
    },
    "A3": {
        "name": "A3_temporal_only",
        "model_type": "temporal",
        "fusion_type": "adaptive",
        "description": "Temporal-only (ablation)",
    },
    "A4": {
        "name": "A4_sequential",
        "model_type": "sequential",
        "fusion_type": "adaptive",
        "description": "Sequential CNN→BiLSTM (ablation)",
    },
}


def get_experiment_dir(model_type: str, fusion_type: str, dataset_name: str = "dfdc02") -> str:
    """Вычисляет имя директории эксперимента как в Config.experiment_name()."""
    parts = [dataset_name, model_type, "seed42", "bs8", "T16"]
    if model_type == "full":
        parts.append(fusion_type)
    return os.path.join(PROJECT_DIR, "experiments", "_".join(parts))


def is_experiment_complete(exp_dir: str) -> bool:
    """Проверяет, завершён ли эксперимент (есть metrics.json с test результатами)."""
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return False
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
        # Проверяем что есть test метрики и history с несколькими эпохами
        test = data.get("test", {})
        history = data.get("history", {})
        train_loss = history.get("train_loss", [])
        has_test = "auc" in test and test["auc"] > 0
        has_training = len(train_loss) >= 2  # минимум 2 эпохи
        return has_test and has_training
    except (json.JSONDecodeError, OSError, KeyError):
        return False


def run_single_experiment(
    exp_key: str,
    exp_config: dict,
    dataset_root: str,
    max_epochs: int,
    max_retries: int = 3,
) -> bool:
    """Запускает один эксперимент как subprocess с ретраями."""
    exp_dir = get_experiment_dir(exp_config["model_type"], exp_config["fusion_type"])

    if is_experiment_complete(exp_dir):
        print(f"\n{'='*60}")
        print(f"✅ {exp_key} ({exp_config['name']}) — уже завершён, ПРОПУСК")
        print(f"{'='*60}")
        return True

    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"🚀 {exp_key} ({exp_config['name']}) — попытка {attempt}/{max_retries}")
        print(f"   {exp_config['description']}")
        print(f"   model={exp_config['model_type']}, fusion={exp_config['fusion_type']}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, os.path.join(PROJECT_DIR, "train.py"),
            "--model_type", exp_config["model_type"],
            "--fusion_type", exp_config["fusion_type"],
            "--dataset_root", dataset_root,
            "--max_epochs", str(max_epochs),
            "--batch_size", "8",
            "--seed", "42",
            "--device", "auto",
            "--num_workers", "0",
        ]

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_DIR,
                timeout=None,  # без таймаута — ждём сколько нужно
            )
            elapsed = time.time() - start_time
            elapsed_min = elapsed / 60

            if result.returncode == 0:
                print(f"\n✅ {exp_key} завершён за {elapsed_min:.1f} мин")
                return True
            else:
                print(f"\n❌ {exp_key} упал с кодом {result.returncode} "
                      f"(попытка {attempt}/{max_retries}, {elapsed_min:.1f} мин)")

        except KeyboardInterrupt:
            print(f"\n⏹  Прервано пользователем")
            sys.exit(1)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ {exp_key} ошибка: {e} (попытка {attempt}/{max_retries})")

        if attempt < max_retries:
            wait = 10 * attempt
            print(f"   Ожидание {wait}с перед следующей попыткой...")
            time.sleep(wait)

    print(f"\n💀 {exp_key} — все {max_retries} попыток исчерпаны")
    return False


def generate_plots():
    """Генерирует графики после экспериментов."""
    print(f"\n{'='*60}")
    print("📊 Генерация графиков...")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, "plot_training_curves.py"),
        "--experiments_dir", os.path.join(PROJECT_DIR, "experiments"),
    ]
    subprocess.run(cmd, cwd=PROJECT_DIR)


def generate_summary(results: dict):
    """Выводит финальную сводку."""
    print(f"\n{'='*70}")
    print("📋 ИТОГОВАЯ СВОДКА")
    print(f"{'='*70}")
    print(f"{'Эксперимент':<25} {'Статус':<12} {'Test AUC':>10} {'Test Acc':>10} {'Test F1':>10}")
    print("-" * 70)

    for key in ["A1", "A2", "A3", "A4"]:
        exp = EXPERIMENTS[key]
        exp_dir = get_experiment_dir(exp["model_type"], exp["fusion_type"])
        status = results.get(key, "не запущен")

        if status == "success" and is_experiment_complete(exp_dir):
            try:
                with open(os.path.join(exp_dir, "metrics.json")) as f:
                    data = json.load(f)
                test = data.get("test", {})
                auc = test.get("auc", 0)
                acc = test.get("accuracy", 0)
                f1 = test.get("f1", 0)
                print(f"{exp['name']:<25} {'✅ OK':<12} {auc:>10.4f} {acc:>10.4f} {f1:>10.4f}")
            except Exception:
                print(f"{exp['name']:<25} {'⚠️ partial':<12} {'—':>10} {'—':>10} {'—':>10}")
        else:
            print(f"{exp['name']:<25} {'❌ FAIL':<12} {'—':>10} {'—':>10} {'—':>10}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Устойчивый запуск экспериментов с auto-resume")
    parser.add_argument("--dataset_root", type=str,
                        default="data/preprocessed_data/preprocessed_DFDC02_16")
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Число повторных попыток при падении")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Запустить только указанные (A1 A2 A3 A4)")
    args = parser.parse_args()

    experiments_to_run = args.only if args.only else ["A1", "A2", "A3", "A4"]

    print(f"\n{'#'*70}")
    print(f"# RESILIENT EXPERIMENT RUNNER")
    print(f"# Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"# Dataset: {args.dataset_root}")
    print(f"# Max epochs: {args.max_epochs}")
    print(f"# Max retries: {args.max_retries}")
    print(f"# Эксперименты: {', '.join(experiments_to_run)}")
    print(f"{'#'*70}")

    # Проверяем какие уже завершены
    for key in experiments_to_run:
        exp = EXPERIMENTS[key]
        exp_dir = get_experiment_dir(exp["model_type"], exp["fusion_type"])
        if is_experiment_complete(exp_dir):
            print(f"  {key}: ✅ уже завершён")
        else:
            print(f"  {key}: ⏳ требует обучения")

    results = {}
    total_start = time.time()

    for key in experiments_to_run:
        if key not in EXPERIMENTS:
            print(f"⚠️  Неизвестный эксперимент: {key}, пропуск")
            continue

        success = run_single_experiment(
            exp_key=key,
            exp_config=EXPERIMENTS[key],
            dataset_root=args.dataset_root,
            max_epochs=args.max_epochs,
            max_retries=args.max_retries,
        )
        results[key] = "success" if success else "failed"

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n⏱  Общее время: {total_elapsed:.1f} мин")

    # Генерируем графики
    generate_plots()

    # Финальная сводка
    generate_summary(results)

    # Проверяем что всё ОК
    all_ok = all(results.get(k) == "success" for k in experiments_to_run)
    if all_ok:
        print("\n🎉 Все эксперименты завершены успешно!")
    else:
        failed = [k for k in experiments_to_run if results.get(k) != "success"]
        print(f"\n⚠️  Не завершены: {', '.join(failed)}")
        print(f"   Перезапустите: python run_resilient.py --only {' '.join(failed)}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
