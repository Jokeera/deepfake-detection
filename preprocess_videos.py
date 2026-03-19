"""
Face-centric preprocessing for deepfake videos.

Назначение:
- пройти по raw video dataset;
- извлечь из каждого видео последовательность face crops;
- сохранить их в структуру:
    output_root/
      real/<video_id>/0000.jpg ...
      fake/<video_id>/0000.jpg ...
- сохранить manifest.csv и summary.json.

Финальные принципы:
1. Видеоцентричная обработка: один video_id -> одна папка кадров.
2. Безопасные уникальные video_id, чтобы не было коллизий по stem().
3. Более мягкие дефолты после EDA:
   - min_detection_ratio по умолчанию 0.55, а не 0.60;
   - min_saved_faces по умолчанию согласован с 16-кадровым клипом.
4. Добавлен режим strict_temporal, но по умолчанию он выключен:
   не надо слишком агрессивно выкидывать видео при текущей temporal readiness.
5. Подробный summary для дипломной работы и аудита качества preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class PreprocessConfig:
    input_root: str
    output_root: str

    # Сколько кадров максимум брать из одного видео.
    max_frames: int = 16

    # Итоговый размер face crop.
    output_size: int = 224

    # Порог confidence для MTCNN.
    min_face_confidence: float = 0.90

    # Мягче, чем было раньше, чтобы не терять лишние видео.
    min_detection_ratio: float = 0.55

    # Для модели с num_frames=16 разумно держать дефолт 16.
    # Если видео сохраняет меньше — dataset.py потом всё равно может отфильтровать.
    min_saved_faces: int = 16

    # Отступ вокруг лица.
    face_margin_ratio: float = 0.20

    # Ограничение длинной стороны кадра перед detector input.
    detector_max_side: int = 960

    # auto / cpu / cuda / mps
    device: str = "auto"

    jpeg_quality: int = 95

    # Повторные попытки вокруг целевого кадра.
    retry_offsets: Tuple[int, ...] = (-2, -1, 1, 2)

    # Сохранять ли manifest и summary
    save_manifest: bool = True

    # Для задач, где важнее temporal-consistency, чем recall.
    # По умолчанию выключен.
    strict_temporal: bool = False

    # Если strict_temporal=True, можно потребовать минимальную длину
    # непрерывной последовательности найденных лиц.
    min_contiguous_faces: int = 8

    def validate(self) -> None:
        if self.max_frames <= 0:
            raise ValueError("max_frames должен быть > 0.")
        if self.output_size <= 0:
            raise ValueError("output_size должен быть > 0.")
        if not (0.0 <= self.min_face_confidence <= 1.0):
            raise ValueError("min_face_confidence должен быть в диапазоне [0, 1].")
        if not (0.0 <= self.min_detection_ratio <= 1.0):
            raise ValueError("min_detection_ratio должен быть в диапазоне [0, 1].")
        if self.min_saved_faces <= 0:
            raise ValueError("min_saved_faces должен быть > 0.")
        if self.min_saved_faces > self.max_frames:
            raise ValueError("min_saved_faces не может быть больше max_frames.")
        if self.face_margin_ratio < 0.0:
            raise ValueError("face_margin_ratio не может быть < 0.")
        if self.detector_max_side <= 0:
            raise ValueError("detector_max_side должен быть > 0.")
        if self.jpeg_quality < 1 or self.jpeg_quality > 100:
            raise ValueError("jpeg_quality должен быть в диапазоне 1..100.")
        if self.min_contiguous_faces <= 0:
            raise ValueError("min_contiguous_faces должен быть > 0.")


# =============================================================================
# DEVICE
# =============================================================================

def select_device(device_mode: str) -> torch.device:
    mode = device_mode.lower()

    if mode == "cpu":
        return torch.device("cpu")

    if mode == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Запрошен device='cuda', но CUDA недоступен.")

    if mode == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Запрошен device='mps', но MPS недоступен.")

    if mode != "auto":
        raise ValueError(f"Неизвестный режим device: {device_mode}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# FILE DISCOVERY / LABELS / IDS
# =============================================================================

def find_videos(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"input_root не существует: {root}")

    videos: List[Path] = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    videos = sorted(videos)
    if not videos:
        raise RuntimeError(f"В {root} не найдено ни одного видеофайла.")
    return videos


def infer_label_from_path(video_path: Path) -> str:
    """
    Строго определяет label по пути.
    """
    parts = [p.lower() for p in video_path.parts]

    real_keywords = {"real", "original", "pristine", "actors"}
    fake_keywords = {"fake", "manipulated", "altered", "deepfake", "df"}

    # Substring-поиск по каждой части пути отдельно (не по всему пути)
    has_real = any(kw in part for part in parts for kw in real_keywords)
    has_fake = any(kw in part for part in parts for kw in fake_keywords)

    if has_real and not has_fake:
        return "real"
    if has_fake and not has_real:
        return "fake"

    raise ValueError(
        f"Не удалось однозначно определить label для видео:\n{video_path}\n"
        "Ожидались признаки real/original/pristine/actors "
        "или fake/manipulated/altered/deepfake в пути."
    )


def make_safe_video_id(video_path: Path, input_root: Path) -> str:
    """
    Делает устойчивый уникальный video_id:
    - основан на относительном пути,
    - безопасен для файловой системы,
    - защищён коротким hash suffix.
    """
    rel = video_path.relative_to(input_root).with_suffix("")
    rel_str = str(rel).replace("\\", "/")
    safe = rel_str.replace("/", "__").replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe)

    digest = hashlib.md5(rel_str.encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{digest}"


# =============================================================================
# VIDEO SAMPLING / IO
# =============================================================================

def get_uniform_frame_indices(frame_count: int, max_frames: int) -> List[int]:
    if frame_count <= 0:
        return []

    num = min(frame_count, max_frames)
    if num == 1:
        return [0]

    indices = np.linspace(0, frame_count - 1, num=num, dtype=int)
    return indices.tolist()


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


def safe_read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb


# =============================================================================
# FACE DETECTION / CROPPING
# =============================================================================


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


@dataclass
class FaceDetectionResult:
    crop: np.ndarray
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (expanded)
    n_faces_in_frame: int


def detect_face_crop(
    detector: MTCNN,
    frame_rgb: np.ndarray,
    min_conf: float,
    margin_ratio: float,
    detector_max_side: int,
) -> Optional[FaceDetectionResult]:
    """
    Возвращает face crop + metadata или None.
    """
    orig_h, orig_w = frame_rgb.shape[:2]

    det_input, scale = resize_for_detector(frame_rgb, detector_max_side)
    pil_img = Image.fromarray(det_input)

    boxes, probs = detector.detect(pil_img)

    n_faces = 0
    best_box = None
    best_conf = 0.0

    if boxes is not None and probs is not None:
        for box, prob in zip(boxes, probs):
            if prob is not None and float(prob) >= min_conf:
                n_faces += 1
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                if best_box is None or area > (
                    max(0.0, best_box[2] - best_box[0]) * max(0.0, best_box[3] - best_box[1])
                ):
                    best_box = box
                    best_conf = float(prob)

    if best_box is None:
        return None

    best_box = np.asarray(best_box, dtype=np.float32)
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

    return FaceDetectionResult(
        crop=crop,
        confidence=round(best_conf, 4),
        bbox=(x1, y1, x2, y2),
        n_faces_in_frame=n_faces,
    )


def resize_face_crop(face_rgb: np.ndarray, output_size: int) -> np.ndarray:
    h, w = face_rgb.shape[:2]
    # INTER_AREA для downscaling, INTER_LANCZOS4 для upscaling
    if h >= output_size and w >= output_size:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LANCZOS4
    return cv2.resize(face_rgb, (output_size, output_size), interpolation=interp)


# =============================================================================
# TEMPORAL UTILS
# =============================================================================

def longest_contiguous_detections(detection_mask: Sequence[bool]) -> int:
    """
    Оценивает длину самой длинной непрерывной последовательности
    успешных детекций лиц среди target-кадров клипа.

    detection_mask: список bool — True если лицо найдено, False если нет.
    Порядок соответствует target_indices (т.е. порядку кадров в клипе).
    """
    if not detection_mask:
        return 0

    best = 0
    cur = 0

    for detected in detection_mask:
        if detected:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return best


# =============================================================================
# SINGLE VIDEO PROCESSING
# =============================================================================

def process_video(
    video_path: Path,
    detector: MTCNN,
    cfg: PreprocessConfig,
    input_root: Path,
) -> Dict:
    label = infer_label_from_path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Не удалось определить число кадров: {video_path}")

    target_indices = get_uniform_frame_indices(frame_count, cfg.max_frames)
    if not target_indices:
        cap.release()
        raise RuntimeError(f"Не удалось выбрать кадры из видео: {video_path}")

    saved_faces: List[np.ndarray] = []
    used_indices: List[int] = []
    detection_mask: List[bool] = []
    face_confidences: List[float] = []
    face_bboxes: List[Tuple[int, int, int, int]] = []
    retry_used: List[bool] = []
    detection_failures = 0

    for idx in target_indices:
        frame_rgb = safe_read_frame(cap, idx)
        result: Optional[FaceDetectionResult] = None
        was_retry = False

        if frame_rgb is not None:
            result = detect_face_crop(
                detector=detector,
                frame_rgb=frame_rgb,
                min_conf=cfg.min_face_confidence,
                margin_ratio=cfg.face_margin_ratio,
                detector_max_side=cfg.detector_max_side,
            )

        if result is None:
            for offset in cfg.retry_offsets:
                alt_idx = idx + offset
                if alt_idx < 0 or alt_idx >= frame_count:
                    continue

                alt_frame = safe_read_frame(cap, alt_idx)
                if alt_frame is None:
                    continue

                result = detect_face_crop(
                    detector=detector,
                    frame_rgb=alt_frame,
                    min_conf=cfg.min_face_confidence,
                    margin_ratio=cfg.face_margin_ratio,
                    detector_max_side=cfg.detector_max_side,
                )
                if result is not None:
                    idx = alt_idx
                    was_retry = True
                    break

        if result is None:
            detection_failures += 1
            detection_mask.append(False)
            continue

        face_crop = resize_face_crop(result.crop, cfg.output_size)
        saved_faces.append(face_crop)
        used_indices.append(idx)
        detection_mask.append(True)
        face_confidences.append(result.confidence)
        face_bboxes.append(result.bbox)
        retry_used.append(was_retry)

    cap.release()

    sampled_count = len(target_indices)
    saved_count = len(saved_faces)
    detection_ratio = saved_count / max(sampled_count, 1)
    contiguous_len = longest_contiguous_detections(detection_mask)

    if saved_count < cfg.min_saved_faces:
        return {
            "status": "dropped",
            "reason": f"saved_faces<{cfg.min_saved_faces}",
            "label": label,
            "video_path": str(video_path),
            "sampled_frames": sampled_count,
            "saved_faces": saved_count,
            "detection_failures": detection_failures,
            "detection_ratio": round(detection_ratio, 4),
            "max_contiguous_faces": contiguous_len,
        }

    if detection_ratio < cfg.min_detection_ratio:
        return {
            "status": "dropped",
            "reason": f"detection_ratio<{cfg.min_detection_ratio}",
            "label": label,
            "video_path": str(video_path),
            "sampled_frames": sampled_count,
            "saved_faces": saved_count,
            "detection_failures": detection_failures,
            "detection_ratio": round(detection_ratio, 4),
            "max_contiguous_faces": contiguous_len,
        }

    if cfg.strict_temporal and contiguous_len < cfg.min_contiguous_faces:
        return {
            "status": "dropped",
            "reason": f"max_contiguous_faces<{cfg.min_contiguous_faces}",
            "label": label,
            "video_path": str(video_path),
            "sampled_frames": sampled_count,
            "saved_faces": saved_count,
            "detection_failures": detection_failures,
            "detection_ratio": round(detection_ratio, 4),
            "max_contiguous_faces": contiguous_len,
        }

    video_id = make_safe_video_id(video_path, input_root)
    out_dir = Path(cfg.output_root) / label / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, face_rgb in enumerate(saved_faces):
        out_path = out_dir / f"{i:04d}.jpg"
        Image.fromarray(face_rgb).save(
            out_path,
            format="JPEG",
            quality=cfg.jpeg_quality,
            optimize=True,
        )

    return {
        "status": "saved",
        "reason": "",
        "label": label,
        "video_id": video_id,
        "video_path": str(video_path),
        "output_dir": str(out_dir),
        "frame_count": frame_count,
        "sampled_frames": sampled_count,
        "saved_faces": saved_count,
        "detection_failures": detection_failures,
        "detection_ratio": round(detection_ratio, 4),
        "max_contiguous_faces": contiguous_len,
        "used_frame_indices": used_indices,
        "mean_confidence": round(float(np.mean(face_confidences)), 4) if face_confidences else 0.0,
        "min_confidence": round(float(np.min(face_confidences)), 4) if face_confidences else 0.0,
        "retry_count": sum(retry_used),
    }


# =============================================================================
# MANIFEST / SUMMARY
# =============================================================================

def save_manifest_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, list):
                    normalized[key] = json.dumps(value, ensure_ascii=False)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _safe_mean(values: List[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _safe_median(values: List[float]) -> float:
    return round(float(np.median(values)), 4) if values else 0.0


def _count_by_reason(rows: List[Dict]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for row in rows:
        reason = row.get("reason", "")
        if reason:
            result[reason] = result.get(reason, 0) + 1
    return result


def _count_status_by_label(rows: List[Dict], label: str, status: str) -> int:
    return sum(1 for r in rows if r.get("label") == label and r.get("status") == status)


def build_summary(rows: List[Dict], cfg: PreprocessConfig) -> Dict:
    total = len(rows)
    saved_rows = [r for r in rows if r.get("status") == "saved"]
    dropped_rows = [r for r in rows if r.get("status") == "dropped"]
    error_rows = [r for r in rows if r.get("status") == "error"]

    detection_ratios = [float(r.get("detection_ratio", 0.0)) for r in saved_rows]
    saved_faces_values = [int(r.get("saved_faces", 0)) for r in saved_rows]
    contiguous_values = [int(r.get("max_contiguous_faces", 0)) for r in saved_rows]

    return {
        "config": asdict(cfg),
        "total_videos": total,
        "saved_videos": len(saved_rows),
        "dropped_videos": len(dropped_rows),
        "error_videos": len(error_rows),
        "real_total": sum(1 for r in rows if r.get("label") == "real"),
        "fake_total": sum(1 for r in rows if r.get("label") == "fake"),
        "real_saved": _count_status_by_label(rows, "real", "saved"),
        "fake_saved": _count_status_by_label(rows, "fake", "saved"),
        "real_dropped": _count_status_by_label(rows, "real", "dropped"),
        "fake_dropped": _count_status_by_label(rows, "fake", "dropped"),
        "drop_reasons": _count_by_reason(dropped_rows),
        "error_reasons": _count_by_reason(error_rows),
        "mean_detection_ratio_saved": _safe_mean(detection_ratios),
        "median_detection_ratio_saved": _safe_median(detection_ratios),
        "mean_saved_faces": _safe_mean(saved_faces_values),
        "median_saved_faces": _safe_median(saved_faces_values),
        "mean_max_contiguous_faces": _safe_mean(contiguous_values),
        "median_max_contiguous_faces": _safe_median(contiguous_values),
    }


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Face-centric preprocessing for deepfake videos")
    parser.add_argument("input_root", type=str, help="Корень raw video dataset")
    parser.add_argument("output_root", type=str, help="Куда сохранять processed face frames")

    parser.add_argument("--max-frames", type=int, default=None,
                        help="Максимум кадров на видео (default: из config.py num_frames)")
    parser.add_argument("--output-size", type=int, default=224, help="Размер face crop")
    parser.add_argument("--min-face-confidence", type=float, default=0.90, help="Порог confidence")
    parser.add_argument("--min-detection-ratio", type=float, default=0.55, help="Минимальная доля удачных detections")
    parser.add_argument("--min-saved-faces", type=int, default=16, help="Минимум сохранённых face crops")
    parser.add_argument("--face-margin-ratio", type=float, default=0.20, help="Отступ вокруг лица")
    parser.add_argument("--detector-max-side", type=int, default=960, help="Макс. длинная сторона для detector input")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--strict-temporal", action="store_true", help="Требовать более стабильную temporal-последовательность")
    parser.add_argument("--min-contiguous-faces", type=int, default=8, help="Минимальная длина непрерывной последовательности найденных лиц")
    parser.add_argument("--no-manifest", action="store_true", help="Не сохранять manifest.csv и summary.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # If --max-frames not specified, use num_frames from main Config
    from config import Config as TrainConfig
    max_frames = args.max_frames if args.max_frames is not None else TrainConfig().num_frames

    cfg = PreprocessConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        max_frames=max_frames,
        output_size=args.output_size,
        min_face_confidence=args.min_face_confidence,
        min_detection_ratio=args.min_detection_ratio,
        min_saved_faces=args.min_saved_faces,
        face_margin_ratio=args.face_margin_ratio,
        detector_max_side=args.detector_max_side,
        device=args.device,
        jpeg_quality=args.jpeg_quality,
        strict_temporal=args.strict_temporal,
        min_contiguous_faces=args.min_contiguous_faces,
        save_manifest=not args.no_manifest,
    )
    cfg.validate()

    input_root = Path(cfg.input_root)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = select_device(cfg.device)
    print(f"[INFO] Device: {device}")

    videos = find_videos(str(input_root))
    print(f"[INFO] Найдено видео: {len(videos)}")

    detector = MTCNN(
        image_size=cfg.output_size,
        margin=0,
        min_face_size=20,
        keep_all=True,
        post_process=False,
        device=device,
    )

    rows: List[Dict] = []

    for video_path in tqdm(videos, desc="Preprocessing videos"):
        try:
            row = process_video(video_path, detector, cfg, input_root=input_root)
            rows.append(row)
        except Exception as e:
            rows.append(
                {
                    "status": "error",
                    "reason": str(e),
                    "label": "",
                    "video_path": str(video_path),
                    "traceback": traceback.format_exc(limit=1),
                }
            )

    if cfg.save_manifest:
        manifest_path = output_root / "manifest.csv"
        summary_path = output_root / "summary.json"

        save_manifest_csv(rows, manifest_path)

        summary = build_summary(rows, cfg)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Manifest: {manifest_path}")
        print(f"[INFO] Summary : {summary_path}")

    saved_count = sum(1 for r in rows if r.get("status") == "saved")
    dropped_count = sum(1 for r in rows if r.get("status") == "dropped")
    error_count = sum(1 for r in rows if r.get("status") == "error")

    print("\n=== PREPROCESS FINISHED ===")
    print(f"Saved  : {saved_count}")
    print(f"Dropped: {dropped_count}")
    print(f"Errors : {error_count}")


if __name__ == "__main__":
    main()