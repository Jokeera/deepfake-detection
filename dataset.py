"""
Dataset pipeline for video-level deepfake detection.

Поддерживаемые форматы датасета:
1. Структура по подпапкам:
   real/video_001/frame_000.jpg, ...
   fake/video_123/frame_000.jpg, ...
2. Плоская структура:
   real/video001_000.jpg, ...
   fake/videoXYZ_015.jpg, ...

Каждый элемент выборки — видеоклип из T кадров + (T-1) frame differences.
Задача остаётся video-level: один label на весь клип.

Ключевые свойства финальной версии:
- split формируется ПО ВИДЕО;
- есть поддержка random / fixed split;
- train sampling более разнообразный, чем валидация/тест;
- augmentations clip-consistent;
- возможно использование weighted sampler через config;
- код ориентирован на стабильность под macOS/MPS и воспроизводимость ВКР.
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import Config


# =============================================================================
# INDEXING
# =============================================================================

def find_real_fake_dirs(root: str) -> Tuple[str, str]:
    """
    Находит папки real и fake внутри root.

    Поддерживает вариации:
      - real / fake
      - Real / Fake
      - original_sequences / manipulated_sequences
      - pristine / altered
      - и др.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"dataset_root не существует или не является директорией: {root}"
        )

    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

    real_dir = None
    fake_dir = None

    real_keywords = ["real", "original", "pristine", "actors"]
    fake_keywords = ["fake", "manipulated", "altered", "deepfake", "df"]

    for dirname in subdirs:
        name_lower = dirname.lower()

        if real_dir is None and any(kw in name_lower for kw in real_keywords):
            real_dir = os.path.join(root, dirname)

        if fake_dir is None and any(kw in name_lower for kw in fake_keywords):
            fake_dir = os.path.join(root, dirname)

    if real_dir is None or fake_dir is None:
        raise FileNotFoundError(
            f"Не удалось найти real/fake директории в '{root}'.\n"
            f"Найденные подкаталоги: {subdirs}\n"
            f"Ожидаются директории, содержащие в названии real/original и fake/manipulated.\n"
            f"Для диагностики проверьте структуру данных через scan_dataset.py."
        )

    return real_dir, fake_dir


def build_video_index(root: str) -> List[Dict]:
    """
    Сканирует датасет и строит индекс видео.

    Возвращает список словарей:
        [
            {
                "video_id": str,
                "frames": [path1, path2, ...],
                "label": int
            },
            ...
        ]

    label:
        0 = real
        1 = fake
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    real_dir, fake_dir = find_real_fake_dirs(root)

    index: List[Dict] = []

    for label, class_dir in [(0, real_dir), (1, fake_dir)]:
        label_name = "real" if label == 0 else "fake"

        all_images: List[str] = []
        for dirpath, _, filenames in os.walk(class_dir):
            for fname in sorted(filenames):
                if Path(fname).suffix.lower() in image_exts:
                    all_images.append(os.path.join(dirpath, fname))

        if not all_images:
            print(f"[ПРЕДУПРЕЖДЕНИЕ] Нет изображений в {class_dir}")
            continue

        videos = _group_frames_by_video(all_images, class_dir)

        for video_id, frames in videos.items():
            frames = sorted(frames)
            index.append(
                {
                    "video_id": f"{label_name}/{video_id}",
                    "frames": frames,
                    "label": label,
                }
            )

    if not index:
        raise RuntimeError("Индекс датасета пуст. Проверьте dataset_root и структуру папок.")

    # Защита от случайных дубликатов video_id
    video_ids = [v["video_id"] for v in index]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("Обнаружены дублирующиеся video_id в индексе датасета.")

    real_count = sum(1 for v in index if v["label"] == 0)
    fake_count = sum(1 for v in index if v["label"] == 1)
    print(f"[ДАННЫЕ] Всего видео: {len(index)} (real: {real_count}, fake: {fake_count})")

    return index


def _group_frames_by_video(image_paths: List[str], class_dir: str) -> Dict[str, List[str]]:
    """
    Группирует кадры по видео.

    Подход 1:
      если кадры лежат в подпапках -> первая подпапка считается video_id.

    Подход 2:
      если структура плоская -> video_id извлекается из имени файла.
    """
    videos: Dict[str, List[str]] = {}

    depths = set()
    for p in image_paths[:100]:
        rel = os.path.relpath(os.path.dirname(p), class_dir)
        depth = 0 if rel == "." else len(Path(rel).parts)
        depths.add(depth)

    if 0 not in depths and len(depths) > 0:
        # Кадры лежат по подпапкам
        for p in image_paths:
            rel = os.path.relpath(p, class_dir)
            parts = Path(rel).parts
            video_id = parts[0] if len(parts) > 1 else Path(p).stem
            videos.setdefault(video_id, []).append(p)
    else:
        # Плоская структура
        for p in image_paths:
            fname = Path(p).stem
            match = re.match(r"^(.+?)[-_]?\d+$", fname)
            video_id = match.group(1) if match else fname
            videos.setdefault(video_id, []).append(p)

    return videos


# =============================================================================
# SPLITS
# =============================================================================

def split_index(
    index: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Разделяет индекс на train / val / test ПО ВИДЕО.
    Стратификация выполняется отдельно по классам real/fake.
    """
    rng = np.random.RandomState(seed)

    real_videos = [v for v in index if v["label"] == 0]
    fake_videos = [v for v in index if v["label"] == 1]

    def _split(videos: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        videos = videos.copy()
        rng.shuffle(videos)
        n = len(videos)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            videos[:n_train],
            videos[n_train:n_train + n_val],
            videos[n_train + n_val:],
        )

    r_train, r_val, r_test = _split(real_videos)
    f_train, f_val, f_test = _split(fake_videos)

    train = r_train + f_train
    val = r_val + f_val
    test = r_test + f_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"[SPLIT] train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def save_split(cfg: Config, train_idx: List[Dict], val_idx: List[Dict], test_idx: List[Dict]) -> None:
    """
    Сохраняет split в JSON для воспроизводимости.
    """
    payload = {
        "dataset_name": cfg.dataset_name,
        "seed": cfg.seed,
        "train_video_ids": [x["video_id"] for x in train_idx],
        "val_video_ids": [x["video_id"] for x in val_idx],
        "test_video_ids": [x["video_id"] for x in test_idx],
    }
    with open(cfg.split_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_split(cfg: Config, index: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Загружает ранее сохранённый split.
    """
    split_path = cfg.split_path()
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Файл split не найден: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    by_id = {x["video_id"]: x for x in index}

    def _collect(ids: List[str]) -> List[Dict]:
        result = []
        missing = []
        for vid in ids:
            if vid in by_id:
                result.append(by_id[vid])
            else:
                missing.append(vid)
        if missing:
            raise RuntimeError(
                f"В split-файле есть video_id, отсутствующие в текущем dataset_root. "
                f"Пример: {missing[:5]}"
            )
        return result

    train_idx = _collect(payload["train_video_ids"])
    val_idx = _collect(payload["val_video_ids"])
    test_idx = _collect(payload["test_video_ids"])

    print(f"[SPLIT] loaded fixed split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx


# =============================================================================
# DATASET
# =============================================================================

class DeepfakeVideoDataset(Dataset):
    """
    PyTorch Dataset для video-level deepfake detection.

    Возвращает:
        {
            "spatial":  Tensor [T, 3, spatial_size, spatial_size],
            "temporal": Tensor [T-1, 3, temporal_size, temporal_size],
            "label":    Tensor scalar,
            "video_id": str,
        }
    """

    def __init__(self, video_index: List[Dict], cfg: Config, is_train: bool = False):
        self.cfg = cfg
        self.is_train = is_train
        self.num_frames = cfg.num_frames

        self.videos = [v for v in video_index if len(v["frames"]) >= cfg.min_frames_per_video]
        dropped = len(video_index) - len(self.videos)
        if dropped > 0:
            print(f"[ДАННЫЕ] Отброшено {dropped} видео с < {cfg.min_frames_per_video} кадрами")

        if not self.videos:
            raise RuntimeError("После фильтрации не осталось ни одного видео для Dataset.")

        self.spatial_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict:
        video_info = self.videos[idx]
        frame_paths = video_info["frames"]
        label = video_info["label"]
        video_id = video_info["video_id"]

        indices = self._sample_indices(len(frame_paths), self.num_frames, is_train=self.is_train)

        spatial_frames: List[np.ndarray] = []
        temporal_frames: List[np.ndarray] = []

        do_hflip = False
        if self.is_train and self.cfg.augment_hflip:
            do_hflip = (np.random.random() < 0.5)

        brightness_delta = 0.0
        contrast_factor = 1.0
        if self.is_train:
            brightness_delta = np.random.uniform(
                -self.cfg.augment_brightness,
                self.cfg.augment_brightness,
            )
            contrast_factor = np.random.uniform(
                1.0 - self.cfg.augment_contrast,
                1.0 + self.cfg.augment_contrast,
            )

        jpeg_apply = False
        jpeg_quality = self.cfg.augment_jpeg_quality_max
        if self.is_train and self.cfg.augment_jpeg_prob > 0:
            jpeg_apply = (np.random.random() < self.cfg.augment_jpeg_prob)
            jpeg_quality = np.random.randint(
                self.cfg.augment_jpeg_quality_min,
                self.cfg.augment_jpeg_quality_max + 1,
            )

        for i in indices:
            img = cv2.imread(frame_paths[i])
            if img is None:
                raise RuntimeError(f"Не удалось прочитать кадр: {frame_paths[i]}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            spatial_img = cv2.resize(img, (self.cfg.spatial_size, self.cfg.spatial_size))
            temporal_img = cv2.resize(img, (self.cfg.temporal_size, self.cfg.temporal_size))

            if self.is_train:
                spatial_img = self._augment_basic(
                    spatial_img,
                    hflip=do_hflip,
                    brightness=brightness_delta,
                    contrast=contrast_factor,
                )
                temporal_img = self._augment_basic(
                    temporal_img,
                    hflip=do_hflip,
                    brightness=brightness_delta,
                    contrast=contrast_factor,
                )

                if jpeg_apply:
                    spatial_img = self._apply_jpeg(spatial_img, jpeg_quality)
                    temporal_img = self._apply_jpeg(temporal_img, jpeg_quality)

            spatial_frames.append(spatial_img)
            temporal_frames.append(temporal_img)

        diff_frames: List[np.ndarray] = []
        for i in range(len(temporal_frames) - 1):
            diff = temporal_frames[i + 1].astype(np.float32) - temporal_frames[i].astype(np.float32)
            diff_frames.append(diff)

        spatial_tensor = self._frames_to_tensor(spatial_frames, normalize=True)
        temporal_tensor = self._diffs_to_tensor(diff_frames)

        return {
            "spatial": spatial_tensor,
            "temporal": temporal_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "video_id": video_id,
        }

    def _sample_indices(self, total_frames: int, t: int, is_train: bool) -> List[int]:
        """
        Выбор T индексов кадров.

        train:
            лёгкий случайный temporal jitter для более разнообразного обучения.
        val/test:
            детерминированный uniform sampling для воспроизводимой оценки.
        """
        if total_frames < 1:
            raise RuntimeError("Видео без кадров не может быть использовано в Dataset.")

        if total_frames >= t:
            if is_train:
                # Случайный выбор по равным сегментам
                bins = np.linspace(0, total_frames, t + 1, dtype=int)
                indices = []
                for i in range(t):
                    start = bins[i]
                    end = max(bins[i + 1], start + 1)
                    idx = np.random.randint(start, end)
                    indices.append(min(idx, total_frames - 1))
                return indices

            # Детерминированный uniform sampling для val/test
            return [int(k * total_frames / t) for k in range(t)]

        # Если кадров меньше, чем нужно, делаем циклическое повторение
        indices = list(range(total_frames))
        while len(indices) < t:
            indices.append(indices[len(indices) % total_frames])
        return indices[:t]

    def _augment_basic(
        self,
        img: np.ndarray,
        hflip: bool,
        brightness: float,
        contrast: float,
    ) -> np.ndarray:
        if hflip:
            img = np.fliplr(img).copy()

        img = img.astype(np.float32)
        img = img * contrast + brightness * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_jpeg(self, img: np.ndarray, quality: int) -> np.ndarray:
        """
        JPEG-компрессия с clip-consistent quality.
        """
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=int(quality))
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        return np.array(compressed)

    def _frames_to_tensor(self, frames: List[np.ndarray], normalize: bool = True) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for frame in frames:
            t = torch.from_numpy(frame).float() / 255.0
            t = t.permute(2, 0, 1)
            if normalize:
                t = self.spatial_normalize(t)
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    def _diffs_to_tensor(self, diffs: List[np.ndarray]) -> torch.Tensor:
        if len(diffs) == 0:
            raise RuntimeError("Список temporal diffs пуст.")

        stacked = np.stack(diffs, axis=0).astype(np.float32)

        mean = float(stacked.mean())
        std = float(stacked.std())
        if std < 1e-8:
            std = 1.0

        stacked = (stacked - mean) / std
        return torch.from_numpy(stacked).permute(0, 3, 1, 2)

    def labels(self) -> List[int]:
        """
        Удобный доступ к label'ам для sampler.
        """
        return [int(v["label"]) for v in self.videos]


# =============================================================================
# DATALOADERS
# =============================================================================

def _make_train_sampler(train_ds: DeepfakeVideoDataset, cfg: Config):
    """
    Создаёт WeightedRandomSampler при необходимости.
    """
    if not cfg.use_weighted_sampler:
        return None

    labels = train_ds.labels()
    class_counts = np.bincount(labels, minlength=2).astype(np.float32)

    if np.any(class_counts == 0):
        raise RuntimeError("Невозможно создать weighted sampler: один из классов отсутствует.")

    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_multi_dataset_index(dataset_roots: List[str], dataset_names: List[str]) -> List[Dict]:
    """
    Объединяет видео-индексы из нескольких датасетов.

    Каждому video_id добавляется префикс с именем датасета
    и поле "source" для отслеживания происхождения.

    Пример: video_id "real/vid_001" из "dfdc02" -> "dfdc02::real/vid_001"
    """
    combined: List[Dict] = []

    for root, name in zip(dataset_roots, dataset_names):
        index = build_video_index(root)
        for entry in index:
            entry["video_id"] = f"{name}::{entry['video_id']}"
            entry["source"] = name
            combined.append(entry)

    # Защита от дубликатов
    video_ids = [v["video_id"] for v in combined]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("Обнаружены дублирующиеся video_id в объединённом индексе.")

    real_count = sum(1 for v in combined if v["label"] == 0)
    fake_count = sum(1 for v in combined if v["label"] == 1)
    print(f"[MULTI-DATA] Всего видео: {len(combined)} (real: {real_count}, fake: {fake_count})")
    for name in dataset_names:
        n = sum(1 for v in combined if v.get("source") == name)
        r = sum(1 for v in combined if v.get("source") == name and v["label"] == 0)
        f = sum(1 for v in combined if v.get("source") == name and v["label"] == 1)
        print(f"  [{name}] {n} видео (real: {r}, fake: {f})")

    return combined


def create_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт train / val / test DataLoader'ы.

    Поддерживает multi-dataset: если cfg.dataset_root содержит "+" (напр.
    "/path/to/dfdc02+/path/to/dfd01"), а cfg.dataset_name содержит "+"
    (напр. "dfdc02+dfd01"), загружает несколько датасетов и объединяет.
    """
    cfg.validate()

    # Multi-dataset support
    if "+" in cfg.dataset_root and "+" in cfg.dataset_name:
        roots = [r.strip() for r in cfg.dataset_root.split("+")]
        names = [n.strip() for n in cfg.dataset_name.split("+")]
        if len(roots) != len(names):
            raise ValueError(
                f"Число путей ({len(roots)}) не совпадает с числом имён ({len(names)}). "
                f"dataset_root и dataset_name должны содержать одинаковое количество записей через '+'."
            )
        index = build_multi_dataset_index(roots, names)
    else:
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
        if cfg.save_split:
            save_split(cfg, train_idx, val_idx, test_idx)

    train_ds = DeepfakeVideoDataset(train_idx, cfg, is_train=True)
    val_ds = DeepfakeVideoDataset(val_idx, cfg, is_train=False)
    test_ds = DeepfakeVideoDataset(test_idx, cfg, is_train=False)

    train_sampler = _make_train_sampler(train_ds, cfg)

    # persistent_workers даёт ускорение, когда num_workers > 0.
    persistent = cfg.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader