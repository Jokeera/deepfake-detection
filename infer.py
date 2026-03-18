"""
Single-sample inference for video-level deepfake detection.

Поддерживает два режима входа:
1. Папка с уже предобработанными кадрами:
   path/to/sequence/0000.jpg ...
2. Сырой видеофайл:
   path/to/video.mp4

Назначение:
- загрузить best_model.pt;
- подготовить clip в формате, совместимом с train/evaluate;
- выдать probability + label;
- сохранить result.json для MVP / демо.

Финальные принципы:
1. Inference остаётся video-level, а не image-only.
2. Для папок с кадрами используется та же clip-логика, что и в dataset pipeline.
3. Для raw video делается встроенный face-centric preprocessing через MTCNN.
4. Код остаётся переносимым: CPU / CUDA / MPS.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import Config
from models import build_model

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# =============================================================================
# DEVICE
# =============================================================================

def get_device(device_arg: str = "auto") -> torch.device:
    device_arg = device_arg.lower()

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Запрошен device='cuda', но CUDA недоступен.")
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


# =============================================================================
# PATH / INPUT TYPE
# =============================================================================

def infer_input_type(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Путь не существует: {path}")

    if path.is_dir():
        return "frames_dir"

    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        raise ValueError(
            "infer.py ожидает либо папку с кадрами, либо видеофайл. "
            "Одиночное изображение не подходит для video-level inference."
        )

    raise ValueError(f"Неподдерживаемый тип входа: {path}")


def list_frame_paths(frames_dir: Path) -> List[Path]:
    frames = [
        p for p in sorted(frames_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not frames:
        raise RuntimeError(f"В папке нет изображений: {frames_dir}")
    return frames


# =============================================================================
# SAMPLING / TENSORS
# =============================================================================

def sample_indices(total_frames: int, t: int) -> List[int]:
    if total_frames <= 0:
        raise RuntimeError("Невозможно выбрать кадры: total_frames <= 0")

    if total_frames >= t:
        return [int(k * total_frames / t) for k in range(t)]

    indices = list(range(total_frames))
    while len(indices) < t:
        indices.append(indices[len(indices) % total_frames])
    return indices[:t]


def spatial_normalizer() -> transforms.Normalize:
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def frames_to_tensors(
    frames_rgb: Sequence[np.ndarray],
    cfg: Config,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spatial_frames: List[np.ndarray] = []
    temporal_frames: List[np.ndarray] = []

    for frame_rgb in frames_rgb:
        spatial_img = cv2.resize(
            frame_rgb,
            (cfg.spatial_size, cfg.spatial_size),
            interpolation=cv2.INTER_AREA,
        )
        temporal_img = cv2.resize(
            frame_rgb,
            (cfg.temporal_size, cfg.temporal_size),
            interpolation=cv2.INTER_AREA,
        )
        spatial_frames.append(spatial_img)
        temporal_frames.append(temporal_img)

    normalize = spatial_normalizer()

    spatial_tensors: List[torch.Tensor] = []
    for frame in spatial_frames:
        t = torch.from_numpy(frame).float() / 255.0
        t = t.permute(2, 0, 1)
        t = normalize(t)
        spatial_tensors.append(t)
    spatial_tensor = torch.stack(spatial_tensors, dim=0)

    diff_frames: List[np.ndarray] = []
    for i in range(len(temporal_frames) - 1):
        diff = temporal_frames[i + 1].astype(np.float32) - temporal_frames[i].astype(np.float32)
        diff_frames.append(diff)

    if len(diff_frames) == 0:
        raise RuntimeError("Недостаточно кадров для temporal diffs.")

    stacked = np.stack(diff_frames, axis=0).astype(np.float32)
    mean = float(stacked.mean())
    std = float(stacked.std())
    if std < 1e-8:
        std = 1.0
    stacked = (stacked - mean) / std
    temporal_tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2)

    return spatial_tensor, temporal_tensor


# =============================================================================
# FACE-CENTRIC VIDEO PREP
# =============================================================================

def resize_for_detector(frame_rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = frame_rgb.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return frame_rgb, 1.0

    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def choose_largest_box(
    boxes: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    min_conf: float,
) -> Optional[np.ndarray]:
    if boxes is None or probs is None or len(boxes) == 0:
        return None

    valid = []
    for box, prob in zip(boxes, probs):
        if prob is None:
            continue
        if float(prob) >= min_conf:
            x1, y1, x2, y2 = box
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            valid.append((area, box))

    if not valid:
        return None

    valid.sort(key=lambda x: x[0], reverse=True)
    return np.asarray(valid[0][1], dtype=np.float32)


def expand_box(
    box: np.ndarray,
    image_w: int,
    image_h: int,
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(np.float32)

    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin_ratio
    my = bh * margin_ratio

    nx1 = max(0, int(round(x1 - mx)))
    ny1 = max(0, int(round(y1 - my)))
    nx2 = min(image_w, int(round(x2 + mx)))
    ny2 = min(image_h, int(round(y2 + my)))

    return nx1, ny1, nx2, ny2


def detect_face_crop(
    detector,
    frame_rgb: np.ndarray,
    min_conf: float = 0.90,
    margin_ratio: float = 0.20,
    detector_max_side: int = 960,
) -> Optional[np.ndarray]:
    orig_h, orig_w = frame_rgb.shape[:2]

    det_input, scale = resize_for_detector(frame_rgb, detector_max_side)
    pil_img = Image.fromarray(det_input)

    boxes, probs = detector.detect(pil_img)
    best_box = choose_largest_box(boxes, probs, min_conf)
    if best_box is None:
        return None

    if scale != 1.0:
        best_box = best_box / scale

    x1, y1, x2, y2 = expand_box(
        best_box,
        image_w=orig_w,
        image_h=orig_h,
        margin_ratio=margin_ratio,
    )

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


def extract_face_frames_from_video(
    video_path: Path,
    cfg: Config,
    device: torch.device,
    min_face_confidence: float = 0.90,
    face_margin_ratio: float = 0.20,
    detector_max_side: int = 960,
    retry_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
) -> List[np.ndarray]:
    if MTCNN is None:
        raise ImportError(
            "Для инференса по raw video нужен facenet-pytorch. "
            "Установи: pip install facenet-pytorch"
        )

    detector = MTCNN(
        image_size=cfg.spatial_size,
        margin=0,
        min_face_size=20,
        keep_all=True,
        post_process=False,
        device=device,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Не удалось определить число кадров: {video_path}")

    target_indices = sample_indices(frame_count, cfg.num_frames)
    saved_faces: List[np.ndarray] = []

    for idx in target_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        face_crop = None

        if ok and frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            face_crop = detect_face_crop(
                detector=detector,
                frame_rgb=frame_rgb,
                min_conf=min_face_confidence,
                margin_ratio=face_margin_ratio,
                detector_max_side=detector_max_side,
            )

        if face_crop is None:
            for offset in retry_offsets:
                alt_idx = idx + offset
                if alt_idx < 0 or alt_idx >= frame_count:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, alt_idx)
                ok_alt, alt_bgr = cap.read()
                if not ok_alt or alt_bgr is None:
                    continue

                alt_rgb = cv2.cvtColor(alt_bgr, cv2.COLOR_BGR2RGB)
                face_crop = detect_face_crop(
                    detector=detector,
                    frame_rgb=alt_rgb,
                    min_conf=min_face_confidence,
                    margin_ratio=face_margin_ratio,
                    detector_max_side=detector_max_side,
                )
                if face_crop is not None:
                    break

        if face_crop is not None:
            face_crop = cv2.resize(
                face_crop,
                (cfg.spatial_size, cfg.spatial_size),
                interpolation=cv2.INTER_AREA,
            )
            saved_faces.append(face_crop)

    cap.release()

    if len(saved_faces) == 0:
        raise RuntimeError("Не удалось извлечь ни одного face crop из видео.")

    original_count = len(saved_faces)
    while len(saved_faces) < cfg.num_frames:
        saved_faces.append(saved_faces[len(saved_faces) % original_count])

    return saved_faces[:cfg.num_frames]


# =============================================================================
# INPUT PREP
# =============================================================================

def prepare_frames_from_dir(frames_dir: Path, cfg: Config) -> List[np.ndarray]:
    frame_paths = list_frame_paths(frames_dir)
    indices = sample_indices(len(frame_paths), cfg.num_frames)

    frames_rgb: List[np.ndarray] = []
    for idx in indices:
        img = cv2.imread(str(frame_paths[idx]))
        if img is None:
            raise RuntimeError(f"Не удалось прочитать кадр: {frame_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_rgb.append(img)

    return frames_rgb


def prepare_input_tensors(
    input_path: Path,
    cfg: Config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    input_type = infer_input_type(input_path)

    if input_type == "frames_dir":
        frames_rgb = prepare_frames_from_dir(input_path, cfg)
        prep_info = {
            "input_type": "frames_dir",
            "source_path": str(input_path),
            "num_source_frames": len(list_frame_paths(input_path)),
            "num_used_frames": len(frames_rgb),
            "face_preprocessing": "preprocessed_frames_assumed",
        }
    else:
        frames_rgb = extract_face_frames_from_video(
            video_path=input_path,
            cfg=cfg,
            device=device,
        )
        prep_info = {
            "input_type": "video",
            "source_path": str(input_path),
            "num_used_frames": len(frames_rgb),
            "face_preprocessing": "mtcnn_internal",
        }

    spatial_tensor, temporal_tensor = frames_to_tensors(frames_rgb, cfg)

    spatial_tensor = spatial_tensor.unsqueeze(0).to(device)   # [1, T, 3, H, W]
    temporal_tensor = temporal_tensor.unsqueeze(0).to(device) # [1, T-1, 3, h, w]

    return spatial_tensor, temporal_tensor, prep_info


# =============================================================================
# MODEL IO
# =============================================================================

def load_checkpoint_and_cfg(
    checkpoint_path: Path,
    device: torch.device,
    override_device: str,
    use_amp: bool,
) -> Tuple[Dict, Config, torch.nn.Module]:
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )

    saved_cfg = checkpoint.get("config", {})
    cfg = Config()

    for key, value in saved_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    cfg.device = override_device
    cfg.use_amp = use_amp
    cfg.validate()

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return checkpoint, cfg, model


# =============================================================================
# INFERENCE
# =============================================================================

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    spatial_tensor: torch.Tensor,
    temporal_tensor: torch.Tensor,
    cfg: Config,
) -> Dict:
    logits, alpha = model(spatial_tensor, temporal_tensor)
    proba = float(torch.sigmoid(logits).item())
    pred = int(proba >= cfg.decision_threshold)
    label_name = "fake" if pred == 1 else "real"

    result = {
        "probability_fake": round(proba, 6),
        "decision_threshold": float(cfg.decision_threshold),
        "pred_label_int": pred,
        "pred_label": label_name,
    }

    if alpha is not None:
        alpha_np = alpha.detach().cpu().numpy()[0]
        result["fusion_weights"] = {
            "alpha_spatial": round(float(alpha_np[0]), 6),
            "alpha_temporal": round(float(alpha_np[1]), 6),
        }

    return result


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-sample inference for deepfake detection")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Путь к best_model.pt",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Путь к папке с кадрами ИЛИ к видеофайлу",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Куда сохранить result json",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Отключить AMP",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint не найден: {checkpoint_path}")

    device = get_device(args.device)

    checkpoint, cfg, model = load_checkpoint_and_cfg(
        checkpoint_path=checkpoint_path,
        device=device,
        override_device=args.device,
        use_amp=not args.no_amp,
    )

    spatial_tensor, temporal_tensor, prep_info = prepare_input_tensors(
        input_path=input_path,
        cfg=cfg,
        device=device,
    )

    result = run_inference(
        model=model,
        spatial_tensor=spatial_tensor,
        temporal_tensor=temporal_tensor,
        cfg=cfg,
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "dataset_name": cfg.dataset_name,
        "model_type": cfg.model_type,
        "fusion_type": cfg.fusion_type,
        "device": str(device),
        "input": prep_info,
        "result": result,
    }

    print("\n=== INFERENCE RESULT ===")
    print(f"Input type       : {prep_info['input_type']}")
    print(f"Model            : {cfg.model_type}")
    print(f"Probability fake : {result['probability_fake']:.6f}")
    print(f"Threshold        : {result['decision_threshold']:.4f}")
    print(f"Prediction       : {result['pred_label']}")

    if "fusion_weights" in result:
        fw = result["fusion_weights"]
        print(
            f"Fusion weights   : spatial={fw['alpha_spatial']:.4f}, "
            f"temporal={fw['alpha_temporal']:.4f}"
        )

    if args.output is None:
        out_dir = checkpoint_path.parent
        safe_name = input_path.stem.replace(" ", "_")
        args.output = str(out_dir / f"infer_{safe_name}.json")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved            : {args.output}")


if __name__ == "__main__":
    main()