"""
Запуск серии экспериментов для Главы 4 ВКР.

Поддержка:
- mandatory / full
- macOS / Linux / Windows
- CPU / CUDA / MPS
- optional cross-dataset evaluation

Финальные принципы:
1. Каждый эксперимент запускается в изолированной конфигурации.
2. После каждого шага результаты сериализуются в all_results.json.
3. Ведётся сводная таблица и manifest запуска серии.
4. Cross-dataset evaluation выполняется только по успешным чекпоинтам.
"""

from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import DeepfakeVideoDataset, build_video_index
from evaluate import evaluate_model, get_device
from models import build_model
from train import train
from utils import save_metrics


MANDATORY_EXPERIMENTS = [
    {
        "name": "A1_full_model",
        "description": "Полная модель (dual-path + adaptive fusion)",
        "model_type": "full",
        "fusion_type": "adaptive",
    },
    {
        "name": "A2_spatial_only",
        "description": "Spatial-only (ablation: удалён temporal branch)",
        "model_type": "spatial",
        "fusion_type": "adaptive",
    },
    {
        "name": "A3_temporal_only",
        "description": "Temporal-only (ablation: удалён spatial branch)",
        "model_type": "temporal",
        "fusion_type": "adaptive",
    },
    {
        "name": "A4_sequential",
        "description": "Sequential CNN->BiLSTM (ablation: parallel -> sequential)",
        "model_type": "sequential",
        "fusion_type": "adaptive",
    },
]

OPTIONAL_EXPERIMENTS = [
    {
        "name": "A5_fusion_concat",
        "description": "Fusion: concat вместо adaptive",
        "model_type": "full",
        "fusion_type": "concat",
    },
    {
        "name": "A6_fusion_gate",
        "description": "Fusion: gate вместо adaptive",
        "model_type": "full",
        "fusion_type": "gate",
    },
]


def make_cfg(base_cfg: Config, exp_config: Dict) -> Config:
    cfg = deepcopy(base_cfg)
    cfg.model_type = exp_config["model_type"]
    cfg.fusion_type = exp_config["fusion_type"]
    cfg.validate()
    return cfg


def ensure_dir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_run_manifest(path: str, payload: Dict) -> None:
    ensure_dir_for_file(path)
    save_metrics(payload, path)


def run_experiment(exp_config: Dict, base_cfg: Config) -> Dict:
    cfg = make_cfg(base_cfg, exp_config)

    print("\n" + "#" * 72)
    print(f"# ЭКСПЕРИМЕНТ: {exp_config['name']}")
    print(f"# {exp_config['description']}")
    print(f"# model_type={cfg.model_type}, fusion_type={cfg.fusion_type}")
    print("#" * 72 + "\n")

    started_at = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    try:
        metrics = train(cfg)
        duration_sec = round(time.time() - t0, 2)

        metrics["experiment_name"] = exp_config["name"]
        metrics["description"] = exp_config["description"]
        metrics["status"] = "success"
        metrics["checkpoint_path"] = cfg.checkpoint_path()
        metrics["experiment_dir"] = cfg.experiment_dir()
        metrics["started_at"] = started_at
        metrics["finished_at"] = datetime.now().isoformat(timespec="seconds")
        metrics["duration_sec"] = duration_sec

    except Exception as e:
        duration_sec = round(time.time() - t0, 2)
        print(f"[ОШИБКА] {exp_config['name']}: {e}")
        metrics = {
            "experiment_name": exp_config["name"],
            "description": exp_config["description"],
            "status": "failed",
            "error": str(e),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": duration_sec,
        }

    return metrics


def generate_results_table(all_results: List[Dict]) -> str:
    lines = []
    lines.append("=" * 118)
    lines.append("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    lines.append("=" * 118)

    has_cross = any("cross_dataset_auc" in r for r in all_results)
    if has_cross:
        header = (
            f"{'Эксперимент':<25} {'Status':<10} {'Test AUC':>12} {'Cross AUC':>10} "
            f"{'Test Acc':>10} {'Test F1':>10} {'Epoch':>8} {'Overfit?':>10}"
        )
    else:
        header = (
            f"{'Эксперимент':<25} {'Status':<10} {'Test AUC':>12} "
            f"{'Test Acc':>10} {'Test F1':>10} {'Epoch':>8} {'Overfit?':>10}"
        )

    lines.append(header)
    lines.append("-" * len(header))

    full_auc = None
    for r in all_results:
        if r.get("status") == "success" and r.get("experiment_name") == "A1_full_model":
            full_auc = r.get("test", {}).get("auc")
            break

    for r in all_results:
        name = str(r.get("experiment_name", "?"))[:25]
        status = str(r.get("status", "unknown"))[:10]

        if status != "success":
            if has_cross:
                lines.append(
                    f"{name:<25} {status:<10} {'FAILED':>12} {'—':>10} {'—':>10} {'—':>10} {'—':>8} {'—':>10}"
                )
            else:
                lines.append(
                    f"{name:<25} {status:<10} {'FAILED':>12} {'—':>10} {'—':>10} {'—':>8} {'—':>10}"
                )
            continue

        test = r.get("test", {})
        auc = float(test.get("auc", 0.0))
        acc = float(test.get("accuracy", 0.0))
        f1 = float(test.get("f1", 0.0))
        epoch = r.get("best_epoch", "?")

        best_val = r.get("best_val_metrics") or {}
        val_auc = best_val.get("auc") if isinstance(best_val, dict) else None

        overfit_flag = "?"
        if val_auc is not None:
            try:
                gap = float(val_auc) - float(auc)
                overfit_flag = "WARN" if gap > 0.03 else "OK"
            except Exception:
                overfit_flag = "?"

        delta = ""
        if full_auc is not None and r.get("experiment_name") != "A1_full_model":
            delta = f" ({auc - full_auc:+.4f})"

        if has_cross:
            cross_auc = r.get("cross_dataset_auc")
            cross_str = f"{cross_auc:.4f}" if cross_auc is not None else "—"
            lines.append(
                f"{name:<25} {status:<10} {auc:>10.4f}{delta} {cross_str:>10} "
                f"{acc:>10.4f} {f1:>10.4f} {str(epoch):>8} {overfit_flag:>10}"
            )
        else:
            lines.append(
                f"{name:<25} {status:<10} {auc:>10.4f}{delta} "
                f"{acc:>10.4f} {f1:>10.4f} {str(epoch):>8} {overfit_flag:>10}"
            )

    lines.append("=" * 118)
    return "\n".join(lines)


def run_cross_dataset_eval(
    all_results: List[Dict],
    experiments: List[Dict],
    base_cfg: Config,
    cross_dataset_root: str,
    cross_dataset_name: str,
    device_arg: str,
) -> List[Dict]:
    print("\n" + "=" * 72)
    print("CROSS-DATASET EVALUATION")
    print(f"Второй датасет: {cross_dataset_root}")
    print(f"Имя второго датасета: {cross_dataset_name}")
    print("=" * 72 + "\n")

    device = get_device(device_arg)
    cross_index = build_video_index(cross_dataset_root)
    if len(cross_index) == 0:
        raise RuntimeError("Cross-dataset index пуст. Проверьте путь и структуру датасета.")

    for result in all_results:
        if result.get("status") != "success":
            continue

        exp_name = result["experiment_name"]
        exp_config = next(e for e in experiments if e["name"] == exp_name)

        checkpoint_path = result.get("checkpoint_path")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[ПРОПУСК] Чекпоинт не найден: {checkpoint_path}")
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        # Restore config from checkpoint for architecture consistency
        cfg = Config()
        for k, v in checkpoint.get("config", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.dataset_root = cross_dataset_root
        cfg.dataset_name = cross_dataset_name
        cfg.device = str(device)

        model = build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        cross_ds = DeepfakeVideoDataset(cross_index, cfg, is_train=False)
        cross_loader = DataLoader(
            cross_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
        )

        cross_results = evaluate_model(
            model=model,
            loader=cross_loader,
            device=device,
            cfg=cfg,
        )

        cross_auc = float(cross_results["metrics"]["auc"])
        result["cross_dataset_name"] = cross_dataset_name
        result["cross_dataset_auc"] = cross_auc

        cross_path = os.path.join(result["experiment_dir"], "cross_dataset_eval.json")
        save_metrics(cross_results, cross_path)

        print(f"{exp_name}: Cross-dataset AUC = {cross_auc:.4f}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="Запуск серии экспериментов для ВКР")

    parser.add_argument("--dataset_root", type=str, required=True, help="Путь к preprocessed dataset")
    parser.add_argument("--dataset_name", type=str, default=None, help="Имя датасета")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--level", type=str, default="mandatory", choices=["mandatory", "full"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--cross_dataset_root", type=str, default=None)
    parser.add_argument("--cross_dataset_name", type=str, default="cross_dataset")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    base_cfg = Config()
    base_cfg.dataset_root = args.dataset_root

    if args.dataset_name is not None:
        base_cfg.dataset_name = args.dataset_name
    if args.output_dir is not None:
        base_cfg.output_dir = args.output_dir
    if args.seed is not None:
        base_cfg.seed = args.seed
    if args.batch_size is not None:
        base_cfg.batch_size = args.batch_size
    if args.max_epochs is not None:
        base_cfg.max_epochs = args.max_epochs
    if args.num_workers is not None:
        base_cfg.num_workers = args.num_workers
    if args.device is not None:
        base_cfg.device = args.device
    if args.no_amp:
        base_cfg.use_amp = False
    if args.pin_memory:
        base_cfg.pin_memory = True

    os.makedirs(base_cfg.output_dir, exist_ok=True)
    base_cfg.validate()

    experiments = list(MANDATORY_EXPERIMENTS)
    if args.level == "full":
        experiments += OPTIONAL_EXPERIMENTS

    print("\n" + "=" * 72)
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ ДЛЯ ВКР")
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Датасет: {base_cfg.dataset_root}")
    print(f"Имя датасета: {base_cfg.dataset_name}")
    print(f"Уровень: {args.level} ({len(experiments)} экспериментов)")
    print(f"Устройство: {base_cfg.device}")
    print("=" * 72 + "\n")

    manifest_path = os.path.join(base_cfg.output_dir, "run_manifest.json")
    summary_path = os.path.join(base_cfg.output_dir, "all_results.json")
    table_path = os.path.join(base_cfg.output_dir, "results_table.txt")

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": base_cfg.dataset_root,
        "dataset_name": base_cfg.dataset_name,
        "output_dir": base_cfg.output_dir,
        "level": args.level,
        "seed": base_cfg.seed,
        "batch_size": base_cfg.batch_size,
        "max_epochs": base_cfg.max_epochs,
        "num_workers": base_cfg.num_workers,
        "device": base_cfg.device,
        "use_amp": base_cfg.use_amp,
        "pin_memory": base_cfg.pin_memory,
        "cross_dataset_root": args.cross_dataset_root,
        "cross_dataset_name": args.cross_dataset_name if args.cross_dataset_root else None,
        "experiments": [e["name"] for e in experiments],
    }
    save_run_manifest(manifest_path, manifest)

    all_results: List[Dict] = []

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {exp['name']}")
        result = run_experiment(exp, base_cfg)
        all_results.append(result)
        save_metrics(all_results, summary_path)

    if args.cross_dataset_root:
        all_results = run_cross_dataset_eval(
            all_results=all_results,
            experiments=experiments,
            base_cfg=base_cfg,
            cross_dataset_root=args.cross_dataset_root,
            cross_dataset_name=args.cross_dataset_name,
            device_arg=base_cfg.device,
        )
        save_metrics(all_results, summary_path)

    table = generate_results_table(all_results)
    print("\n" + table)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table)

    save_metrics(all_results, summary_path)

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_run_manifest(manifest_path, manifest)

    print(f"\nManifest запуска: {manifest_path}")
    print(f"Таблица сохранена: {table_path}")
    print(f"Все результаты: {summary_path}")


if __name__ == "__main__":
    main()