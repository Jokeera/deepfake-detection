# Deepfake Detection — Video-Level Dual-Path Spatiotemporal Analysis

## О проекте

Это thesis-final реализация метода детектирования манипулированных видеопоследовательностей для ВКР НИЯУ МИФИ.

Проект решает задачу **video-level deepfake detection**, а не классификацию одиночных кадров.

### Финальная архитектура
- **Spatial Branch**: CNN-backbone для пространственных признаков кадров
- **Temporal Branch**: frame differences + CNN + Transformer для межкадровой динамики
- **Fusion**: adaptive / concat / gate
- **Classification Head**: бинарная классификация real / fake

### Ключевые финальные принципы
- задача остаётся **video-level**
- базовая длина клипа: **16 кадров**
- основная метрика выбора лучшей модели: **ROC-AUC**
- `unfreeze_last_n_blocks = 0` оставлен как стабильный baseline для текущей среды
- проект поддерживает **CPU / CUDA / MPS**
- для raw video используется **face-centric preprocessing**

---

## Структура проекта

```text
deepfake_detection/
├── app.py                         # Flask MVP для single-sample inference
├── config.py                      # Единая конфигурация проекта
├── scan_dataset.py                # Диагностика структуры датасета
├── preprocess_videos.py           # Face-centric preprocessing raw video -> frame folders
├── dataset.py                     # Video-level Dataset + DataLoader
├── train.py                       # Обучение
├── evaluate.py                    # Оценка val / test / all
├── infer.py                       # Single-sample inference
├── run_experiments.py             # Серия экспериментов для ВКР
├── run_vkr_experiments_pipeline.sh# Оркестрация серии запусков
├── utils.py                       # Метрики, seed, логирование, JSON
├── requirements.txt               # Зависимости
├── README.md                      # Документация
└── models/
    ├── __init__.py
    ├── dual_path.py
    ├── spatial_branch.py
    ├── temporal_branch.py
    ├── spatial_only.py
    ├── temporal_only.py
    └── sequential.py
```

---

## Требования

- Python **3.9+**
- Для ускорения желательно:
  - **CUDA GPU** на Linux/Windows, или
  - **Apple Silicon / MPS** на macOS
- Но проект также может работать на **CPU**

Установка:

```bash
pip install -r requirements.txt
```

---

## Ожидаемая структура данных

Проект работает с **предобработанными frame folders**.

Ожидаемая структура:

```text
dataset_root/
├── real/
│   ├── video_001/
│   │   ├── 0000.jpg
│   │   ├── 0001.jpg
│   │   └── ...
│   └── ...
└── fake/
    ├── video_002/
    │   ├── 0000.jpg
    │   ├── 0001.jpg
    │   └── ...
    └── ...
```

Также `dataset.py` умеет распознавать и некоторые вариации real/fake имён директорий.

---

## 1. Диагностика датасета

Перед обучением желательно проверить структуру:

```bash
python scan_dataset.py /path/to/preprocessed_dataset
```

---

## 2. Предобработка raw video

Если у вас исходный датасет состоит из видеофайлов, сначала извлеките face crops:

```bash
python preprocess_videos.py /path/to/raw_video_dataset /path/to/output_preprocessed \
  --max-frames 16 \
  --device auto
```

Полезные параметры:

```bash
--output-size 224
--min-face-confidence 0.90
--min-detection-ratio 0.55
--min-saved-faces 16
--strict-temporal
--min-contiguous-faces 8
```

После этого получите структуру вида:

```text
output_preprocessed/
├── real/
│   └── <video_id>/0000.jpg ...
└── fake/
    └── <video_id>/0000.jpg ...
```

---

## 3. Обучение одной модели

### Основная модель (рекомендуемый thesis baseline)

```bash
python train.py \
  --dataset_root /path/to/preprocessed_dataset \
  --dataset_name dfdc02 \
  --model_type full \
  --fusion_type adaptive \
  --seed 42 \
  --batch_size 8 \
  --max_epochs 30 \
  --num_frames 16 \
  --device auto \
  --output_dir ./experiments
```

### Быстрый smoke-test

```bash
python train.py \
  --dataset_root /path/to/preprocessed_dataset \
  --dataset_name dfdc02 \
  --model_type full \
  --fusion_type adaptive \
  --batch_size 2 \
  --max_epochs 1 \
  --num_frames 16 \
  --device auto \
  --output_dir ./experiments_smoke
```

---

## 4. Оценка обученной модели

### Test split

```bash
python evaluate.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs8_T16_adaptive/best_model.pt \
  --split test \
  --device auto
```

### Cross-dataset evaluation

```bash
python evaluate.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs8_T16_adaptive/best_model.pt \
  --dataset_root /path/to/another_preprocessed_dataset \
  --dataset_name dfd01 \
  --split all \
  --device auto
```

---

## 5. Запуск серии экспериментов для ВКР

### Mandatory (достаточно для базовой защиты)

```bash
python run_experiments.py \
  --dataset_root /path/to/preprocessed_dataset \
  --dataset_name dfdc02 \
  --output_dir ./experiments \
  --level mandatory \
  --device auto \
  --batch_size 8 \
  --max_epochs 30 \
  --seed 42
```

### Full (все эксперименты)

```bash
python run_experiments.py \
  --dataset_root /path/to/preprocessed_dataset \
  --dataset_name dfdc02 \
  --output_dir ./experiments \
  --level full \
  --device auto \
  --batch_size 8 \
  --max_epochs 30 \
  --seed 42
```

### Full + cross-dataset

```bash
python run_experiments.py \
  --dataset_root /path/to/preprocessed_dataset \
  --dataset_name dfdc02 \
  --output_dir ./experiments \
  --level full \
  --device auto \
  --batch_size 8 \
  --max_epochs 30 \
  --seed 42 \
  --cross_dataset_root /path/to/another_preprocessed_dataset \
  --cross_dataset_name dfd01
```

---

## 6. Single-sample inference (CLI)

Поддерживаются два режима входа:
- raw video
- folder с уже предобработанными кадрами

### Видео

```bash
python infer.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs8_T16_adaptive/best_model.pt \
  --input /path/to/video.mp4 \
  --device auto \
  --output ./infer_result.json
```

### Папка кадров

```bash
python infer.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs8_T16_adaptive/best_model.pt \
  --input /path/to/frames_folder \
  --device auto \
  --output ./infer_result.json
```

---

## 7. Flask MVP

Локальный MVP для демонстрации single-sample inference:

```bash
python app.py
```

После запуска открой:

```text
http://127.0.0.1:7860
```

### Что умеет MVP
- автопоиск `best_model.pt`
- короткие human-readable названия моделей
- upload raw video
- upload frames folder
- preview видео и sampled frames
- structured result panel
- сохранение `app_infer_*.json`
- fallback `MPS -> CPU` для проблемных runtime-case

### Ограничения MVP
- current clip length берётся из checkpoint config
- свободное переключение `16 / 20 / 40 / 60` для inference не включено
- raw video path — **face-centric**
- если лицо не найдено, inference завершится ошибкой
- в multi-face сценах текущая логика использует **largest detected face**

---

## 8. Где лежат результаты

Для каждого эксперимента создаётся директория вида:

```text
experiments/
└── dfdc02_full_seed42_bs8_T16_adaptive/
    ├── best_model.pt
    ├── metrics.json
    ├── predictions.csv
    ├── fusion_weights.json
    ├── training.log
    └── ...
```

Сводные артефакты серии:

```text
experiments/
├── run_manifest.json
├── all_results.json
├── results_table.txt
└── final_summary.md
```

---

## 9. Практические замечания для ВКР

- Проект валиден именно как **video-level** постановка
- Для текущей среды используется консервативный stable baseline
- На Apple Silicon агрессивный fine-tuning spatial backbone был признан нестабильным
- Поэтому `unfreeze_last_n_blocks = 0` оставлен намеренно
- Основная защищаемая модель: **full + adaptive fusion**
- Рекомендуемый UI/demo checkpoint:
  `dfdc02_full_seed42_bs8_T16_adaptive`

---

## 10. Что проверять при проблемах

| Проблема | Что делать |
|----------|------------|
| `dataset_root` не распознаётся | Запустить `scan_dataset.py` |
| raw video не проходит preprocessing | Проверить, есть ли в видео лицо |
| `mps` падает на inference | Использовать `auto` или `cpu` |
| `CUDA out of memory` | Уменьшить `batch_size` |
| very small dataset | уменьшить `batch_size`, проверить `min_frames_per_video` и preprocessing |
| файл не открывается как видео | проверить формат и целостность файла |

---

## 11. Главные thesis-компоненты в коде

| Компонент | Файл |
|----------|------|
| Конфигурация эксперимента | `config.py` |
| Video-level dataset pipeline | `dataset.py` |
| Spatial branch | `spatial_branch.py` |
| Temporal branch | `temporal_branch.py` |
| Dual-path fusion | `dual_path.py` |
| Sequential baseline | `sequential.py` |
| Training protocol | `train.py` |
| Evaluation | `evaluate.py` |
| Single inference | `infer.py` |
| MVP | `app.py` |
| Серия экспериментов | `run_experiments.py` |
