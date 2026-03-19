# Deepfake Detection — Video-Level Dual-Path Spatiotemporal Analysis

Магистерская диссертация, НИЯУ МИФИ, 2026.

**Репозиторий:** [github.com/Jokeera/deepfake-detection](https://github.com/Jokeera/deepfake-detection)

## О проекте

Метод детектирования манипулированных видеопоследовательностей на основе
пространственно-временного анализа лицевых областей.

Проект решает задачу **video-level deepfake detection** (не классификация одиночных кадров).

### Архитектура
- **Spatial Branch**: EfficientNet-B4 — покадровые пространственные признаки (224×224)
- **Temporal Branch**: EfficientNet-B0 + Temporal Transformer — межкадровая динамика через frame differences (128×128)
- **Fusion**: adaptive weighted fusion (обучаемые веса)
- **Classification Head**: бинарная классификация real / fake (BCEWithLogitsLoss)

### Результаты (DFDC02, seed=42, 30 epochs, Kaggle GPU)

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| A1: Full (dual-path) | 0.9777 | 0.9253 | 0.9201 | 0.0749 | 18 |
| **A2: Spatial-only** | **0.9835** | **0.9273** | **0.9250** | **0.0606** | 29 |
| A3: Temporal-only | 0.9655 | 0.9152 | 0.9075 | 0.0950 | 14 |
| A4: Sequential (BiLSTM) | 0.9715 | 0.9172 | 0.9130 | 0.0787 | 29 |

Результаты воспроизведены дважды (v6, v7) с идентичными метриками.

---

## Быстрый старт

### Требования

- Python **3.10+**
- PyTorch **2.0+**
- GPU (NVIDIA CUDA) — рекомендуется для обучения
- Также поддерживаются: Apple Silicon (MPS), CPU

### Установка

```bash
git clone https://github.com/Jokeera/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

### Структура данных

Проект работает с предобработанными frame folders:

```text
data/preprocessed_data/preprocessed_DFDC02_16/
├── real/
│   ├── video_001/
│   │   ├── 0000.jpg ... 0015.jpg
│   └── ...
└── fake/
    ├── video_002/
    │   ├── 0000.jpg ... 0015.jpg
    └── ...
```

---

## Воспроизведение экспериментов

### Вариант 1: Локально (GPU)

```bash
# Полная серия ablation study (4 эксперимента, ~4 часа на T4)
python run_experiments.py \
  --dataset_root data/preprocessed_data/preprocessed_DFDC02_16 \
  --dataset_name dfdc02 \
  --output_dir ./experiments \
  --level mandatory \
  --device auto \
  --batch_size 16 \
  --max_epochs 30 \
  --seed 42
```

### Вариант 2: Kaggle GPU

1. Загрузить preprocessed_DFDC02_16 как Kaggle Dataset
2. Открыть `kaggle-train.ipynb`
3. Включить GPU accelerator
4. Запустить ячейку — все 4 эксперимента выполнятся последовательно

### Одна модель

```bash
python train.py \
  --dataset_root data/preprocessed_data/preprocessed_DFDC02_16 \
  --dataset_name dfdc02 \
  --model_type full \
  --fusion_type adaptive \
  --seed 42 \
  --batch_size 16 \
  --max_epochs 30 \
  --device auto \
  --output_dir ./experiments
```

---

## Оценка и инференс

### Оценка на test split

```bash
python evaluate.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs16_T16_adaptive/best_model.pt \
  --split test \
  --device auto
```

### Инференс одного видео

```bash
python infer.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs16_T16_adaptive/best_model.pt \
  --input /path/to/video.mp4 \
  --device auto
```

### Flask MVP (веб-интерфейс)

```bash
python app.py
# Открыть http://127.0.0.1:7860
```

---

## Визуализация результатов

```bash
python plot_training_curves.py \
  --experiments_dir ./experiments
```

Генерирует 6 графиков: loss curves, val AUC, ablation study, overfit analysis, EER, convergence.

---

## Структура проекта

```text
deepfake_detection/
├── config.py                  # Единая конфигурация (100+ параметров)
├── train.py                   # Обучение с warmup, fine-tuning, early stopping
├── evaluate.py                # Оценка и анализ ошибок
├── infer.py                   # Single-sample inference (CLI)
├── app.py                     # Flask MVP (двуязычный RU/EN)
├── dataset.py                 # Video-level dataset + DataLoader
├── preprocess_videos.py       # Face-centric preprocessing (MTCNN)
├── run_experiments.py         # Автоматизация ablation study
├── plot_training_curves.py    # Визуализация результатов
├── utils.py                   # Метрики, seed, логирование
├── scan_dataset.py            # Диагностика структуры датасета
├── requirements.txt           # Зависимости
├── Dockerfile                 # Контейнеризация
├── kaggle-train.ipynb         # Kaggle GPU training
├── models/
│   ├── __init__.py            # Factory: build_model()
│   ├── dual_path.py           # Full model (spatial + temporal + fusion)
│   ├── spatial_branch.py      # EfficientNet-B4 backbone
│   ├── temporal_branch.py     # Frame diffs + EfficientNet-B0 + Transformer
│   ├── spatial_only.py        # Ablation: spatial-only
│   ├── temporal_only.py       # Ablation: temporal-only
│   └── sequential.py          # Ablation: CNN→BiLSTM
├── splits/
│   └── split_seed42.json      # Фиксированное разбиение (70/15/15)
└── EDA/
    ├── VKR_EDA_DFDC02_v4.ipynb  # Exploratory Data Analysis
    └── reports_final/           # Отчёты, графики, таблицы
```

---

## Ключевые принципы воспроизводимости

| Аспект | Реализация |
|--------|------------|
| Random seed | `set_seed(42)`: torch, numpy, random, CUDA, cudnn.deterministic=True |
| Data split | `splits/split_seed42.json` — фиксированный, стратифицированный по классам |
| Model loading | `strict=True` — точное совпадение state_dict |
| Config | Единый `config.py` с валидацией всех параметров |
| Fine-tuning | `add_param_group()` после unfreeze backbone (epoch 6+) |

---

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| `CUDA out of memory` | Уменьшить `batch_size` (8 или 4) |
| MPS падает на inference | Использовать `--device cpu` |
| Dataset не распознаётся | Запустить `python scan_dataset.py <path>` |
| Лицо не найдено в видео | Проверить, что видео содержит чёткое лицо |

---

## Лицензия

Проект выполнен в рамках магистерской диссертации НИЯУ МИФИ, 2026.
