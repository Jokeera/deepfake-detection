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
import sys
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

        # model_type: prefer metrics.json, fallback to checkpoint config
        model_type = "?"
        if metrics_path.exists():
            model_type = m.get("model_type", "?")
        if model_type == "?":
            try:
                ckpt_meta = torch.load(str(pt), map_location="cpu", weights_only=False)
                model_type = ckpt_meta.get("config", {}).get("model_type", "?")
                del ckpt_meta
            except Exception:
                pass

        found.append({
            "path": str(pt),
            "exp_dir": exp_dir,
            "model_type": model_type,
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
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            spatial = batch["spatial"].to(device)
            temporal = batch["temporal"].to(device)
            labels = batch["label"].numpy()

            logits, _ = model(spatial, temporal)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                done = len(all_probs)
                total = len(cross_index)
                sys.stdout.flush(); print(f"    [{done}/{total}] {done*100//total}%", flush=True)

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)
    metrics = compute_metrics(labels_arr, probs_arr)

    # Per-class metrics for imbalanced datasets
    preds = (probs_arr >= 0.5).astype(int)
    real_mask = labels_arr == 0
    fake_mask = labels_arr == 1
    real_acc = float(np.mean(preds[real_mask] == 0)) if real_mask.any() else 0.0
    fake_acc = float(np.mean(preds[fake_mask] == 1)) if fake_mask.any() else 0.0
    balanced_acc = (real_acc + fake_acc) / 2.0

    metrics["balanced_accuracy"] = round(balanced_acc, 4)
    metrics["real_accuracy"] = round(real_acc, 4)
    metrics["fake_accuracy"] = round(fake_acc, 4)

    return {
        "exp_dir": checkpoint_info["exp_dir"],
        "model_type": checkpoint_info["model_type"],
        "orig_test_auc": checkpoint_info["orig_test_auc"],
        "cross_dataset_metrics": metrics,
        "num_samples": len(all_probs),
        "num_real": int(real_mask.sum()),
        "num_fake": int(fake_mask.sum()),
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
    parser.add_argument("--num_workers", type=int, default=0)
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
    sys.stdout.flush(); print(f"Device: {device}")

    # Find checkpoints
    checkpoints = find_checkpoints(args.checkpoints_dir)
    sys.stdout.flush(); print(f"Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        sys.stdout.flush(); print(f"  {c['exp_dir']} (type={c['model_type']}, orig AUC={c['orig_test_auc']})")

    # Determine min frames from first checkpoint config
    min_frames = 16
    if checkpoints:
        first_ckpt = torch.load(checkpoints[0]["path"], map_location="cpu", weights_only=False)
        min_frames = first_ckpt.get("config", {}).get("num_frames", 16)
        del first_ckpt
    sys.stdout.flush(); print(f"Min frames filter: {min_frames} (from checkpoint config)")

    # Load cross-dataset and filter short videos
    cross_index_raw = build_video_index(args.cross_dataset)
    cross_index = [v for v in cross_index_raw if len(v.get("frames", [])) >= min_frames]
    skipped = len(cross_index_raw) - len(cross_index)
    num_real = sum(1 for v in cross_index if v["label"] == 0)
    num_fake = sum(1 for v in cross_index if v["label"] == 1)
    sys.stdout.flush(); print(f"\nCross-dataset: {len(cross_index)} videos (real={num_real}, fake={num_fake})")
    if skipped:
        sys.stdout.flush(); print(f"  Filtered out {skipped} videos with <{min_frames} frames")
    if num_real > 0 and num_fake > 0:
        ratio = num_fake / num_real
        sys.stdout.flush(); print(f"  Imbalance ratio (fake/real): {ratio:.1f}:1")
        if ratio > 3:
            sys.stdout.flush(); print(f"  NOTE: Dataset is imbalanced. Using AUC + balanced accuracy.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each checkpoint
    all_results = []
    for i, ckpt_info in enumerate(checkpoints, 1):
        sys.stdout.flush(); print(f"\n[{i}/{len(checkpoints)}] {ckpt_info['exp_dir']}")
        t0 = time.time()

        result = evaluate_cross_dataset(
            ckpt_info, cross_index, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        elapsed = time.time() - t0
        result["eval_time_sec"] = round(elapsed, 1)

        m = result["cross_dataset_metrics"]
        sys.stdout.flush(); print(f"  Cross AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
              f"F1={m['f1']:.4f}  EER={m['eer']:.4f}  ({elapsed:.0f}s)")
        sys.stdout.flush(); print(f"  In-domain AUC={result['orig_test_auc']:.4f} → "
              f"Cross AUC={m['auc']:.4f} (Δ={m['auc'] - result['orig_test_auc']:+.4f})")

        all_results.append(result)

    # Summary table
    print("\n" + "=" * 100)
    print("CROSS-DATASET EVALUATION SUMMARY")
    sys.stdout.flush(); print(f"Trained on: DFDC02 | Tested on: DFD01 ({len(cross_index)} videos, "
          f"real={num_real}, fake={num_fake})")
    print("=" * 100)
    sys.stdout.flush(); print(f"{'Model':<30} {'DFDC02 AUC':>11} {'DFD01 AUC':>10} {'Δ AUC':>8} "
          f"{'AP':>7} {'Bal.Acc':>8} {'Real Acc':>9} {'Fake Acc':>9} {'EER':>7}")
    print("-" * 110)
    for r in all_results:
        m = r["cross_dataset_metrics"]
        delta = m["auc"] - r["orig_test_auc"]
        sys.stdout.flush(); print(f"{r['exp_dir']:<30} {r['orig_test_auc']:>11.4f} {m['auc']:>10.4f} "
              f"{delta:>+8.4f} {m.get('ap', 0):>7.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['real_accuracy']:>9.4f} {m['fake_accuracy']:>9.4f} {m['eer']:>7.4f}")
    print("=" * 110)

    # Save results
    output_path = os.path.join(args.output_dir, "cross_eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    sys.stdout.flush(); print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
