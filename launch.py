#!/usr/bin/env python3
"""
Единая точка входа: проверка окружения, данные, обучение, визуализация, демо.

Сценарии:
  python launch.py              — интерактивный режим (всё по шагам)
  python launch.py --demo       — только Flask MVP (ищет best_model.pt)
  python launch.py --train      — только обучение (ищет preprocessed данные)
  python launch.py --plots      — только визуализация из metrics.json
  python launch.py --check      — только проверка окружения и данных
  python launch.py --full       — полный pipeline от preprocessing до демо

Адаптируется под любой dataset:
  python launch.py --raw-data /path/to/videos --data /path/to/output --full
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ─── Цвета для терминала ───────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def info(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[!!]{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{CYAN}{BOLD}{'='*60}{RESET}")
    print(f"{CYAN}{BOLD}  {msg}{RESET}")
    print(f"{CYAN}{BOLD}{'='*60}{RESET}\n")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Запуск команды с выводом в консоль."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, **kwargs)


# ─── 1. Проверка окружения ─────────────────────────────────────────────
def check_environment() -> bool:
    header("1. Проверка окружения")
    ok = True

    # Python version
    v = sys.version_info
    if v >= (3, 10):
        info(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — нужен >= 3.10")
        ok = False

    # PyTorch
    try:
        import torch
        info(f"PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            info(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info("Apple MPS (Metal)")
        else:
            warn("GPU не обнаружен — будет использован CPU (медленно)")
    except ImportError:
        fail("PyTorch не установлен: pip install -r requirements.txt")
        ok = False

    # Ключевые зависимости
    for pkg in ["torchvision", "timm", "sklearn", "cv2", "PIL", "flask"]:
        try:
            __import__(pkg)
            info(f"{pkg}")
        except ImportError:
            fail(f"{pkg} не установлен")
            ok = False

    # facenet-pytorch (для preprocessing и infer raw video)
    try:
        from facenet_pytorch import MTCNN  # noqa: F401
        info("facenet-pytorch (MTCNN)")
    except ImportError:
        warn("facenet-pytorch не установлен — preprocessing raw video недоступен")

    return ok


# ─── 2. Поиск данных ──────────────────────────────────────────────────
def find_preprocessed_data(data_path: str | None) -> str | None:
    """Ищет preprocessed данные: real/ и fake/ с кадрами."""
    candidates = []
    if data_path:
        candidates.append(Path(data_path))

    # Стандартные пути
    project = Path(__file__).parent
    candidates += [
        project / "data" / "preprocessed_data" / "preprocessed_DFDC02_16",
        project / "data" / "preprocessed_data",
        project / "data" / "preprocessed",
        project / "preprocessed_frames",
    ]

    for p in candidates:
        if not p.is_dir():
            continue
        # Проверяем наличие real/ и fake/
        real_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.lower() in
                     ("real", "original", "pristine")]
        fake_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.lower() in
                     ("fake", "manipulated", "altered", "deepfake")]
        if real_dirs and fake_dirs:
            real_count = sum(1 for d in real_dirs[0].iterdir() if d.is_dir())
            fake_count = sum(1 for d in fake_dirs[0].iterdir() if d.is_dir())
            if real_count > 0 and fake_count > 0:
                return str(p)

    return None


def find_raw_data(raw_path: str | None) -> str | None:
    """Ищет raw видео (mp4/avi) с real/fake структурой."""
    candidates = []
    if raw_path:
        candidates.append(Path(raw_path))

    project = Path(__file__).parent
    candidates += [
        project / "data" / "DFDC_Dataset_02",
        project / "data" / "raw",
        project / "data",
    ]

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for p in candidates:
        if not p.is_dir():
            continue
        real_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.lower() in
                     ("real", "original")]
        fake_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.lower() in
                     ("fake", "manipulated")]
        if real_dirs and fake_dirs:
            # Проверяем что внутри есть видео
            sample_files = list(real_dirs[0].iterdir())[:5]
            if any(f.suffix.lower() in video_exts for f in sample_files):
                return str(p)

    return None


def find_checkpoints(experiments_dir: str | None) -> list[str]:
    """Ищет best_model.pt в experiments/."""
    candidates = []
    if experiments_dir:
        candidates.append(Path(experiments_dir))

    project = Path(__file__).parent
    candidates += [
        project / "experiments",
        project / "kaggle_output" / "experiments" / "experiments",
        project / "kaggle_output" / "project" / "experiments",
    ]

    found = []
    for p in candidates:
        if not p.is_dir():
            continue
        for pt in p.rglob("best_model.pt"):
            found.append(str(pt))

    return found


def find_metrics(experiments_dir: str | None) -> list[str]:
    """Ищет metrics.json в experiments/."""
    candidates = []
    if experiments_dir:
        candidates.append(Path(experiments_dir))

    project = Path(__file__).parent
    candidates += [
        project / "experiments",
        project / "kaggle_output" / "experiments" / "experiments",
    ]

    for p in candidates:
        if not p.is_dir():
            continue
        metrics_files = list(p.rglob("metrics.json"))
        if metrics_files:
            return [str(p)]

    return []


# ─── 3. Действия ──────────────────────────────────────────────────────
def do_preprocess(raw_data: str, output_dir: str) -> bool:
    header("Preprocessing: извлечение лиц из видео")
    project = Path(__file__).parent
    script = project / "preprocess_videos.py"
    if not script.exists():
        fail(f"Скрипт не найден: {script}")
        return False

    result = run([
        sys.executable, str(script),
        raw_data, output_dir,
        "--max-frames", "16",
        "--output-size", "224",
        "--device", "auto",
    ])
    return result.returncode == 0


def do_train(dataset_root: str, output_dir: str, batch_size: int = 16,
             max_epochs: int = 30, device: str = "auto") -> bool:
    header("Обучение: 4 эксперимента (ablation study)")
    project = Path(__file__).parent
    script = project / "run_experiments.py"

    result = run([
        sys.executable, str(script),
        "--dataset_root", dataset_root,
        "--dataset_name", "dfdc02",
        "--output_dir", output_dir,
        "--level", "mandatory",
        "--batch_size", str(batch_size),
        "--max_epochs", str(max_epochs),
        "--seed", "42",
        "--device", device,
    ])
    return result.returncode == 0


def do_plots(experiments_dir: str) -> bool:
    header("Визуализация: генерация графиков")
    project = Path(__file__).parent
    script = project / "plot_training_curves.py"

    result = run([
        sys.executable, str(script),
        "--experiments_dir", experiments_dir,
    ])
    return result.returncode == 0


def do_demo() -> bool:
    header("Запуск Flask MVP")
    project = Path(__file__).parent
    script = project / "app.py"

    info("Flask MVP: http://127.0.0.1:7860")
    info("Ctrl+C для остановки")
    print()

    result = run([sys.executable, str(script)])
    return result.returncode == 0


# ─── 4. Главная логика ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Detection — единая точка входа",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python launch.py                              # интерактивный режим
  python launch.py --demo                       # только Flask MVP
  python launch.py --check                      # только проверка окружения
  python launch.py --full                       # полный pipeline
  python launch.py --full --raw-data /path/to/videos  # с нуля из raw видео
  python launch.py --train --data /path/to/preprocessed  # только обучение
        """,
    )
    parser.add_argument("--check", action="store_true", help="Только проверка окружения")
    parser.add_argument("--demo", action="store_true", help="Только Flask MVP")
    parser.add_argument("--train", action="store_true", help="Только обучение")
    parser.add_argument("--plots", action="store_true", help="Только визуализация")
    parser.add_argument("--full", action="store_true", help="Полный pipeline")
    parser.add_argument("--data", type=str, default=None,
                        help="Путь к preprocessed данным (real/fake с кадрами)")
    parser.add_argument("--raw-data", type=str, default=None,
                        help="Путь к raw видео (для preprocessing)")
    parser.add_argument("--output", type=str, default="./experiments",
                        help="Директория для результатов (default: ./experiments)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    project = Path(__file__).parent
    os.chdir(project)

    print(f"\n{BOLD}Deepfake Detection — Video-Level Dual-Path Analysis{RESET}")
    print(f"Проект: {project}\n")

    # ── Проверка окружения ──
    env_ok = check_environment()
    if args.check:
        # Показать что есть
        header("Состояние данных и моделей")
        pp = find_preprocessed_data(args.data)
        if pp:
            info(f"Preprocessed данные: {pp}")
        else:
            warn("Preprocessed данные не найдены")

        raw = find_raw_data(args.raw_data)
        if raw:
            info(f"Raw видео: {raw}")
        else:
            warn("Raw видео не найдены")

        ckpts = find_checkpoints(args.output)
        if ckpts:
            info(f"Обученные модели ({len(ckpts)}):")
            for c in ckpts:
                print(f"    {c}")
        else:
            warn("Обученные модели не найдены")

        metrics = find_metrics(args.output)
        if metrics:
            info(f"Метрики: {metrics[0]}")
        else:
            warn("Метрики не найдены")
        return

    if not env_ok:
        fail("Окружение не готово. Установите зависимости: pip install -r requirements.txt")
        sys.exit(1)

    # ── Только демо ──
    if args.demo:
        ckpts = find_checkpoints(args.output)
        if not ckpts:
            fail("Модели не найдены. Сначала запустите обучение: python launch.py --train")
            sys.exit(1)
        info(f"Найдено {len(ckpts)} моделей")
        do_demo()
        return

    # ── Только графики ──
    if args.plots:
        metrics = find_metrics(args.output)
        if not metrics:
            fail("Метрики не найдены. Сначала запустите обучение.")
            sys.exit(1)
        do_plots(metrics[0])
        return

    # ── Train или Full ──
    if args.train or args.full:
        # Шаг 1: Найти preprocessed данные
        header("2. Поиск данных")
        preprocessed = find_preprocessed_data(args.data)

        if preprocessed:
            info(f"Preprocessed данные найдены: {preprocessed}")
        else:
            warn("Preprocessed данные не найдены")

            # Попробовать preprocessing из raw
            raw = find_raw_data(args.raw_data)
            if raw:
                info(f"Raw видео найдены: {raw}")
                output_preprocessed = args.data or str(
                    project / "data" / "preprocessed_data" / "preprocessed_auto"
                )
                info(f"Запускаю preprocessing → {output_preprocessed}")
                if not do_preprocess(raw, output_preprocessed):
                    fail("Preprocessing завершился с ошибкой")
                    sys.exit(1)
                preprocessed = find_preprocessed_data(output_preprocessed)
                if not preprocessed:
                    fail("После preprocessing данные не найдены")
                    sys.exit(1)
                info(f"Preprocessing завершён: {preprocessed}")
            else:
                fail("Данные не найдены. Укажите путь:")
                print("  python launch.py --data /path/to/preprocessed")
                print("  python launch.py --raw-data /path/to/raw_videos --full")
                sys.exit(1)

        # Шаг 2: Проверить, есть ли уже обученные модели
        header("3. Проверка обученных моделей")
        ckpts = find_checkpoints(args.output)

        if ckpts and not args.full:
            info(f"Найдено {len(ckpts)} обученных моделей — пропускаю обучение")
            info("Для переобучения: python launch.py --full")
        else:
            if ckpts:
                warn(f"Найдено {len(ckpts)} моделей, но запрошен --full — переобучаю")
            if not do_train(
                preprocessed, args.output,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                device=args.device,
            ):
                fail("Обучение завершилось с ошибкой")
                sys.exit(1)
            info("Обучение завершено")

        # Шаг 3: Графики
        metrics = find_metrics(args.output)
        if metrics:
            do_plots(metrics[0])
        else:
            warn("Метрики не найдены — графики не сгенерированы")

        # Шаг 4: Демо (если --full)
        if args.full:
            do_demo()

        return

    # ── Интерактивный режим (по умолчанию) ──
    header("2. Поиск данных и моделей")

    preprocessed = find_preprocessed_data(args.data)
    if preprocessed:
        info(f"Preprocessed данные: {preprocessed}")
    else:
        warn("Preprocessed данные не найдены")

    ckpts = find_checkpoints(args.output)
    if ckpts:
        info(f"Обученные модели: {len(ckpts)}")
    else:
        warn("Обученные модели не найдены")

    metrics = find_metrics(args.output)
    if metrics:
        info(f"Метрики: {metrics[0]}")

    print()
    print(f"{BOLD}Доступные действия:{RESET}")
    print(f"  python launch.py --check      # проверка окружения")
    if preprocessed and not ckpts:
        print(f"  python launch.py --train      # запуск обучения")
    if ckpts:
        print(f"  python launch.py --demo       # запуск Flask MVP")
    if metrics:
        print(f"  python launch.py --plots      # генерация графиков")
    if preprocessed:
        print(f"  python launch.py --full       # полный pipeline до демо")
    print()

    # Если есть модели — предложить демо
    if ckpts:
        print(f"{BOLD}Хотите запустить демо? (y/n):{RESET} ", end="", flush=True)
        try:
            answer = input().strip().lower()
            if answer in ("y", "yes", "д", "да", ""):
                do_demo()
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
