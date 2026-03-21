"""
Сканирование структуры датасета.

Запуск:
    python scan_dataset.py /path/to/dataset
    python scan_dataset.py /path/to/dataset1 /path/to/dataset2
    python scan_dataset.py /path/to/dataset --json          # JSON-отчёт в stdout
    python scan_dataset.py /path/to/dataset --depth 5       # глубина дерева

Скрипт показывает:
- дерево каталогов (полное, без скрытой обрезки);
- статистику по расширениям;
- тип датасета (видео / кадры / смешанный);
- наличие real/fake структуры (поиск на всех уровнях вложенности);
- баланс классов (real vs fake);
- совместимость с dataset.py / preprocess_videos.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

REAL_KEYWORDS = ["real", "original", "pristine", "actors"]
FAKE_KEYWORDS = ["fake", "manipulated", "altered", "deepfake", "df"]


def _find_class_dir(
    subdirs: List[str],
    keywords: List[str],
) -> Optional[str]:
    """Substring-поиск директории класса (совпадает с логикой dataset.py)."""
    for dirname in subdirs:
        name_lower = dirname.lower()
        if any(kw in name_lower for kw in keywords):
            return dirname
    return None


def _search_class_dirs_recursive(
    root_path: Path,
) -> tuple[Optional[Path], Optional[Path]]:
    """Ищет real/fake директории на ВСЕХ уровнях вложенности."""
    for dirpath, dirnames, _ in os.walk(root_path):
        real_match = _find_class_dir(dirnames, REAL_KEYWORDS)
        fake_match = _find_class_dir(dirnames, FAKE_KEYWORDS)
        if real_match and fake_match:
            base = Path(dirpath)
            return base / real_match, base / fake_match
    return None, None


def scan_directory(root: str, max_depth: int = 4, as_json: bool = False) -> Dict:
    """Сканирует датасет и возвращает structured summary."""
    root_path = Path(root)
    result: Dict = {
        "root": str(root_path),
        "exists": root_path.exists(),
        "errors": [],
    }

    if not root_path.exists():
        result["errors"].append(f"Путь не существует: {root_path}")
        if not as_json:
            print(f"[ОШИБКА] Путь не существует: {root_path}")
        return result

    # -------------------------------------------------------------------------
    # 1. Дерево каталогов (полное, без скрытой обрезки)
    # -------------------------------------------------------------------------
    if not as_json:
        print("\n" + "=" * 72)
        print(f"СКАНИРОВАНИЕ: {root_path}")
        print("=" * 72)
        print("\n--- Дерево каталогов ---\n")

    dir_count = 0
    truncated_dirs = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_parts = Path(dirpath).relative_to(root_path).parts
        depth = len(rel_parts)

        if depth > max_depth:
            continue

        dir_count += 1

        if not as_json:
            indent = "  " * depth
            dirname = Path(dirpath).name if depth > 0 else root_path.name
            n_sub = len(dirnames)
            suffix = f" (+{n_sub - 15} ещё)" if n_sub > 15 else ""
            print(f"{indent}{dirname}/ ({len(filenames)} файлов, {n_sub} подпапок{suffix})")

        # Показываем до 15 подпапок на уровне, но НЕ обрезаем os.walk —
        # просто не печатаем остальные. Сканирование идёт полностью.
        if not as_json and len(dirnames) > 15 and depth < max_depth:
            truncated_dirs += 1

    # -------------------------------------------------------------------------
    # 2. Статистика файлов
    # -------------------------------------------------------------------------
    ext_counter: Counter = Counter()
    total_files = 0
    videos_found: List[str] = []
    images_found: List[str] = []
    # Полный путь родительской папки → кол-во изображений (без склейки)
    dirs_with_images: Dict[str, int] = defaultdict(int)

    for dirpath, _, filenames in os.walk(root_path):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            full_path = str(Path(dirpath) / fname)

            ext_counter[ext] += 1
            total_files += 1

            if ext in VIDEO_EXTS:
                videos_found.append(full_path)
            elif ext in IMAGE_EXTS:
                images_found.append(full_path)
                dirs_with_images[dirpath] += 1

    result["total_files"] = total_files
    result["total_dirs"] = dir_count
    result["extensions"] = dict(ext_counter.most_common(30))
    result["video_count"] = len(videos_found)
    result["image_count"] = len(images_found)

    if not as_json:
        print("\n--- Статистика файлов ---\n")
        print(f"Всего файлов   : {total_files}")
        print(f"Всего каталогов: {dir_count}")
        print("\nПо расширениям:")
        for ext, count in ext_counter.most_common(30):
            print(f"  {ext:>8s}: {count:>8d}")

    # -------------------------------------------------------------------------
    # 3. Тип данных
    # -------------------------------------------------------------------------
    if len(videos_found) > 0 and len(images_found) == 0:
        dataset_type = "video"
    elif len(images_found) > 0 and len(videos_found) == 0:
        dataset_type = "frames"
    elif len(videos_found) > 0 and len(images_found) > 0:
        dataset_type = "mixed"
    else:
        dataset_type = "unknown"

    result["dataset_type"] = dataset_type

    unique_video_dirs = len(dirs_with_images)
    avg_frames_per_dir = len(images_found) / max(unique_video_dirs, 1)
    result["unique_video_dirs"] = unique_video_dirs
    result["avg_frames_per_dir"] = round(avg_frames_per_dir, 1)

    if not as_json:
        print("\n--- Определение типа данных ---\n")

        if dataset_type == "video":
            print(f"ТИП: ВИДЕО-ДАТАСЕТ ({len(videos_found)} видеофайлов)")
            print("Примеры:")
            for p in videos_found[:5]:
                print(f"  {p}")
            print("\nРЕКОМЕНДАЦИЯ:")
            print("  - использовать preprocess_videos.py для извлечения кадров")

        elif dataset_type == "frames":
            print(f"ТИП: ДАТАСЕТ С КАДРАМИ ({len(images_found)} изображений)")
            if unique_video_dirs > 10 and avg_frames_per_dir > 5:
                print(
                    f"СТРУКТУРА: подпапки по видео "
                    f"({unique_video_dirs} папок, ~{avg_frames_per_dir:.0f} кадров/папку)"
                )
                print("Это формат для video-level обучения (совместим с dataset.py).")
            else:
                print("СТРУКТУРА: плоская или с малым числом подпапок")
                print("Потребуется группировка кадров по имени файла или по папкам.")
            print("\nПримеры путей:")
            for p in images_found[:5]:
                print(f"  {p}")

        elif dataset_type == "mixed":
            print(
                f"ТИП: СМЕШАННЫЙ ({len(videos_found)} видео, "
                f"{len(images_found)} изображений)"
            )
            print("Для обучения лучше использовать уже извлечённые кадры.")

        else:
            print("ТИП: НЕ ОПРЕДЕЛЁН (нет ни видео, ни изображений)")

    # -------------------------------------------------------------------------
    # 4. Проверка структуры классов (рекурсивный поиск)
    # -------------------------------------------------------------------------
    # Сначала верхний уровень (как dataset.py), потом рекурсивно
    top_subdirs = sorted(
        [d for d in os.listdir(root_path) if (root_path / d).is_dir()]
    )

    real_dir_name = _find_class_dir(top_subdirs, REAL_KEYWORDS)
    fake_dir_name = _find_class_dir(top_subdirs, FAKE_KEYWORDS)

    real_path: Optional[Path] = None
    fake_path: Optional[Path] = None
    class_search_level = "none"

    if real_dir_name and fake_dir_name:
        real_path = root_path / real_dir_name
        fake_path = root_path / fake_dir_name
        class_search_level = "top"
    else:
        # Рекурсивный поиск
        real_path, fake_path = _search_class_dirs_recursive(root_path)
        if real_path and fake_path:
            class_search_level = "nested"

    result["class_structure"] = {
        "found": real_path is not None and fake_path is not None,
        "level": class_search_level,
        "real_dir": str(real_path) if real_path else None,
        "fake_dir": str(fake_path) if fake_path else None,
    }

    # -------------------------------------------------------------------------
    # 5. Баланс классов
    # -------------------------------------------------------------------------
    real_count = 0
    fake_count = 0

    if real_path and fake_path and real_path.exists() and fake_path.exists():
        if dataset_type == "frames":
            # Считаем подпапки (= видео) в real/ и fake/
            real_count = sum(
                1 for d in os.listdir(real_path)
                if (real_path / d).is_dir()
            )
            fake_count = sum(
                1 for d in os.listdir(fake_path)
                if (fake_path / d).is_dir()
            )
        elif dataset_type == "video":
            # Считаем видеофайлы
            real_count = sum(
                1 for f in os.listdir(real_path)
                if Path(f).suffix.lower() in VIDEO_EXTS
            )
            fake_count = sum(
                1 for f in os.listdir(fake_path)
                if Path(f).suffix.lower() in VIDEO_EXTS
            )

    total_samples = real_count + fake_count
    balance_ratio = (
        round(max(real_count, fake_count) / max(min(real_count, fake_count), 1), 2)
        if total_samples > 0 else 0
    )

    result["class_balance"] = {
        "real": real_count,
        "fake": fake_count,
        "total": total_samples,
        "ratio": balance_ratio,
        "balanced": balance_ratio <= 1.5 if total_samples > 0 else None,
    }

    if not as_json:
        print("\n--- Проверка структуры классов ---\n")

        if real_path and fake_path:
            if class_search_level == "top":
                print(f"НАЙДЕНО (верхний уровень):")
            else:
                print(f"НАЙДЕНО (вложенная структура):")
            print(f"  real → {real_path}")
            print(f"  fake → {fake_path}")

            if total_samples > 0:
                print(f"\n  Баланс: real={real_count}, fake={fake_count}, "
                      f"ratio={balance_ratio:.2f}:1")
                if balance_ratio > 2.0:
                    print("  ⚠ Значительный дисбаланс! WeightedRandomSampler рекомендован.")
                elif balance_ratio > 1.5:
                    print("  Умеренный дисбаланс. WeightedRandomSampler желателен.")
                else:
                    print("  Баланс в норме.")
        else:
            print("Стандартная real/fake структура НЕ найдена.")
            print(f"Подкаталоги верхнего уровня: {top_subdirs[:20]}")
            print("\nПроверьте структуру вручную или адаптируйте dataset.py.")

    # -------------------------------------------------------------------------
    # 6. Совместимость с pipeline
    # -------------------------------------------------------------------------
    compatibility = []

    if dataset_type == "frames" and real_path and fake_path:
        if class_search_level == "top":
            compatibility.append("dataset.py: OK (build_video_index совместим)")
        else:
            compatibility.append(
                f"dataset.py: ВНИМАНИЕ — real/fake не на верхнем уровне. "
                f"Укажите dataset_root={real_path.parent}"
            )
        if avg_frames_per_dir >= 8:
            compatibility.append(
                f"num_frames: до {int(avg_frames_per_dir)} кадров/видео доступно"
            )
        else:
            compatibility.append(
                f"num_frames: ВНИМАНИЕ — в среднем {avg_frames_per_dir:.0f} кадров/видео, "
                f"возможно недостаточно для T=16"
            )
    elif dataset_type == "video":
        compatibility.append("dataset.py: нужен preprocess_videos.py сначала")
    elif dataset_type == "mixed":
        compatibility.append(
            "dataset.py: смешанный датасет — "
            "рекомендуется разделить raw-видео и preprocessed-кадры"
        )

    result["compatibility"] = compatibility

    if not as_json:
        print("\n--- Совместимость с pipeline ---\n")
        for note in compatibility:
            print(f"  {note}")
        print("\n" + "=" * 72 + "\n")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Сканирование структуры датасета для deepfake detection pipeline"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Пути к датасетам для сканирования",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Максимальная глубина дерева каталогов (default: 4)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Вывести результат как JSON (machine-readable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_results = []
    for path in args.paths:
        result = scan_directory(path, max_depth=args.depth, as_json=args.json)
        all_results.append(result)

    if args.json:
        output = all_results[0] if len(all_results) == 1 else all_results
        print(json.dumps(output, ensure_ascii=False, indent=2))
