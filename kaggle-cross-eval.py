"""
Cross-dataset evaluation: модели обучены на DFDC02, тестируются на DFD01.

Запуск на Kaggle GPU (или локально):
  python kaggle-cross-eval.py \
    --checkpoints_dir ./experiments \
    --cross_dataset data/preprocessed_data/preprocessed_DFD01_16 \
    --output_dir ./cross_eval_results \
    --device auto

Требует: обученные best_model.pt (из run_experiments.py) и preprocessed DFD01.
НЕ МОДИФИЦИРУЕТ предыдущие эксперименты — только читает чекпоинты.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import DeepfakeVideoDataset, build_video_index
from models import build_model
from utils import compute_metrics


def find_checkpoints(checkpoints_dir: str) -> List[Dict]:
    """Find all best_model.pt with their metrics.json."""
    root = Path(checkpoints_dir)
    found = []
    for pt in root.rglob("best_model.pt"):
        exp_dir = pt.parent.name
        metrics_path = pt.parent / "metrics.json"

        # Read original test AUC for reference
        orig_auc = None
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            orig_auc = m.get("test", {}).get("auc")

        found.append({
            "path": str(pt),
            "exp_dir": exp_dir,
            "model_type": m.get("model_type", "?") if metrics_path.exists() else "?",
            "orig_test_auc": orig_auc,
        })

    # Sort by model type for consistent ordering
    order = {"full": 0, "spatial": 1, "temporal": 2, "sequential": 3}
    found.sort(key=lambda x: order.get(x["model_type"], 99))
    return found


def evaluate_cross_dataset(
    checkpoint_info: Dict,
    cross_index: List[Dict],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 2,
) -> Dict:
    """Evaluate one checkpoint on cross-dataset."""
    ckpt = torch.load(checkpoint_info["path"], map_location=device, weights_only=False)

    cfg = Config()
    for k, v in ckpt["config"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = str(device)

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = DeepfakeVideoDataset(cross_index, cfg, is_train=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            spatial = batch["spatial"].to(device)
            temporal = batch["temporal"].to(device)
            labels = batch["label"].numpy()

            logits, _ = model(spatial, temporal)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    metrics = compute_metrics(probs_arr, labels_arr)

    return {
        "exp_dir": checkpoint_info["exp_dir"],
        "model_type": checkpoint_info["model_type"],
        "orig_test_auc": checkpoint_info["orig_test_auc"],
        "cross_dataset_metrics": metrics,
        "num_samples": len(all_probs),
        "num_real": int((labels_arr == 0).sum()),
        "num_fake": int((labels_arr == 1).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--checkpoints_dir", type=str, required=True,
                        help="Dir with experiment subdirs containing best_model.pt")
    parser.add_argument("--cross_dataset", type=str, required=True,
                        help="Path to preprocessed cross-dataset (real/fake)")
    parser.add_argument("--output_dir", type=str, default="./cross_eval_results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoints_dir)
    print(f"Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        print(f"  {c['exp_dir']} (type={c['model_type']}, orig AUC={c['orig_test_auc']})")

    # Load cross-dataset
    cross_index = build_video_index(args.cross_dataset)
    num_real = sum(1 for v in cross_index if v["label"] == 0)
    num_fake = sum(1 for v in cross_index if v["label"] == 1)
    print(f"\nCross-dataset: {len(cross_index)} videos (real={num_real}, fake={num_fake})")

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each checkpoint
    all_results = []
    for i, ckpt_info in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] {ckpt_info['exp_dir']}")
        t0 = time.time()

        result = evaluate_cross_dataset(
            ckpt_info, cross_index, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        elapsed = time.time() - t0
        result["eval_time_sec"] = round(elapsed, 1)

        m = result["cross_dataset_metrics"]
        print(f"  Cross AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  EER={m['eer']:.4f}  ({elapsed:.0f}s)")
        print(f"  In-domain AUC={result['orig_test_auc']:.4f} → "
              f"Cross AUC={m['auc']:.4f} (Δ={m['auc'] - result['orig_test_auc']:+.4f})")

        all_results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print(f"Trained on: DFDC02 | Tested on: DFD01 ({len(cross_index)} videos)")
    print("=" * 80)
    print(f"{'Model':<30} {'In-domain AUC':>14} {'Cross AUC':>12} {'Δ AUC':>10} {'Cross Acc':>12}")
    print("-" * 80)
    for r in all_results:
        m = r["cross_dataset_metrics"]
        delta = m["auc"] - r["orig_test_auc"]
        print(f"{r['exp_dir']:<30} {r['orig_test_auc']:>14.4f} {m['auc']:>12.4f} "
              f"{delta:>+10.4f} {m['accuracy']:>12.4f}")
    print("=" * 80)

    # Save results
    output_path = os.path.join(args.output_dir, "cross_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
