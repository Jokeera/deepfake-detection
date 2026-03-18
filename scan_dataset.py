"""
Сканирование структуры датасета.

Запуск:
    python scan_dataset.py /path/to/dataset
    python scan_dataset.py /path/to/dataset1 /path/to/dataset2

Скрипт показывает:
- дерево каталогов;
- статистику по расширениям;
- тип датасета (видео / кадры / смешанный);
- наличие real/fake структуры;
- рекомендации для нового config.py.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def scan_directory(root: str, max_depth: int = 3) -> None:
    root_path = Path(root)

    if not root_path.exists():
        print(f"[ОШИБКА] Путь не существует: {root_path}")
        return

    print("\n" + "=" * 72)
    print(f"СКАНИРОВАНИЕ: {root_path}")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # 1. Дерево каталогов
    # -------------------------------------------------------------------------
    print("\n--- Дерево каталогов ---\n")
    dir_count = 0

    for dirpath, dirnames, filenames in os.walk(root_path):
        rel_parts = Path(dirpath).relative_to(root_path).parts
        depth = len(rel_parts)

        if depth > max_depth:
            continue

        indent = "  " * depth
        dirname = Path(dirpath).name if depth > 0 else root_path.name
        print(f"{indent}{dirname}/ ({len(filenames)} файлов)")
        dir_count += 1

        if len(dirnames) > 10:
            dirnames[:] = dirnames[:10]

    # -------------------------------------------------------------------------
    # 2. Статистика файлов
    # -------------------------------------------------------------------------
    print("\n--- Статистика файлов ---\n")
    ext_counter = Counter()
    total_files = 0
    videos_found = []
    images_found = []
    dirs_with_images = defaultdict(int)

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
                parent = Path(dirpath).name
                dirs_with_images[parent] += 1

    print(f"Всего файлов   : {total_files}")
    print(f"Всего каталогов: {dir_count}")
    print("\nПо расширениям:")
    for ext, count in ext_counter.most_common(20):
        print(f"  {ext:>8s}: {count:>8d}")

    # -------------------------------------------------------------------------
    # 3. Тип данных
    # -------------------------------------------------------------------------
    print("\n--- Определение типа данных ---\n")

    if len(videos_found) > 0 and len(images_found) == 0:
        print(f"ТИП: ВИДЕО-ДАТАСЕТ ({len(videos_found)} видеофайлов)")
        print("Примеры:")
        for p in videos_found[:3]:
            print(f"  {p}")

        print("\nРЕКОМЕНДАЦИЯ:")
        print("  - использовать preprocess_videos.py для извлечения кадров")
        print("  - либо указать уже готовый preprocessed dataset в Config.dataset_root")

    elif len(images_found) > 0 and len(videos_found) == 0:
        print(f"ТИП: ДАТАСЕТ С КАДРАМИ ({len(images_found)} изображений)")

        unique_parents = len(dirs_with_images)
        avg_per_dir = len(images_found) / max(unique_parents, 1)

        if unique_parents > 10 and avg_per_dir > 5:
            print(
                f"СТРУКТУРА: подпапки по видео "
                f"({unique_parents} папок, ~{avg_per_dir:.0f} кадров/папку)"
            )
            print("Это хороший формат для video-level обучения.")
        else:
            print("СТРУКТУРА: плоская или с малым числом подпапок")
            print("Потребуется группировка кадров по имени файла или по папкам.")

        print("\nПримеры путей:")
        for p in images_found[:5]:
            print(f"  {p}")

    elif len(videos_found) > 0 and len(images_found) > 0:
        print(
            f"ТИП: СМЕШАННЫЙ ({len(videos_found)} видео, "
            f"{len(images_found)} изображений)"
        )
        print("Для обучения обычно лучше использовать уже извлечённые кадры.")

    else:
        print("ТИП: НЕ ОПРЕДЕЛЁН (нет ни видео, ни изображений)")

    # -------------------------------------------------------------------------
    # 4. Проверка структуры классов
    # -------------------------------------------------------------------------
    print("\n--- Проверка структуры классов ---\n")
    subdirs = [d for d in os.listdir(root_path) if (root_path / d).is_dir()]
    subdirs_lower = {d.lower(): d for d in subdirs}

    real_dir = None
    fake_dir = None

    real_keys = ["real", "original", "pristine", "original_sequences"]
    fake_keys = ["fake", "manipulated", "altered", "deepfake", "manipulated_sequences"]

    for key in real_keys:
        if key in subdirs_lower:
            real_dir = subdirs_lower[key]
            break

    for key in fake_keys:
        if key in subdirs_lower:
            fake_dir = subdirs_lower[key]
            break

    if real_dir and fake_dir:
        print(f"НАЙДЕНО: real='{real_dir}', fake='{fake_dir}'")
        print("\nРЕКОМЕНДАЦИЯ ДЛЯ config.py:")
        print(f"  dataset_root = '{root_path}'")
    else:
        print("Стандартная real/fake структура не найдена.")
        print(f"Подкаталоги верхнего уровня: {subdirs[:10]}")
        print("\nНужно проверить build_video_index() и структуру датасета вручную.")

    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python scan_dataset.py /path/to/dataset")
        print("  python scan_dataset.py /path/to/dataset1 /path/to/dataset2")
        sys.exit(1)

    for path in sys.argv[1:]:
        scan_directory(path)