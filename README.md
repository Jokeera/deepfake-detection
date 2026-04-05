# Deepfake Detection — Video-Level Dual-Path Spatiotemporal Analysis

Магистерская диссертация, НИЯУ МИФИ, 2026.

**Репозиторий:** [github.com/Jokeera/deepfake-detection](https://github.com/Jokeera/deepfake-detection)

---

## Быстрый старт (для проверяющего)

```bash
# 1. Установить Git LFS (нужен для моделей и видео)
git lfs install

# 2. Клонировать
git clone https://github.com/Jokeera/deepfake-detection.git
cd deepfake-detection

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Запустить Flask демо (загрузить видео → получить real/fake)
python app.py
# → http://127.0.0.1:7860
```

> **Без GPU**: Flask демо и инференс работают на CPU/MPS. GPU нужен только для обучения.

---

## О проекте

Метод детектирования манипулированных видеопоследовательностей на основе
пространственно-временного анализа лицевых областей.

Проект решает задачу **video-level deepfake detection** (не классификация одиночных кадров).

### Архитектура

```
Input: Video (T frames, face-cropped via MTCNN)
  │
  ├─→ Spatial Branch (EfficientNet-B4, 224×224)
  │   Per-frame CNN → Mean Pooling → Projection [512]
  │
  └─→ Temporal Branch (EfficientNet-B0 + Transformer, 128×128)
      Frame Differences → Per-diff CNN → Positional Encoding
      → Transformer Encoder (2 layers, 4 heads) → CLS-token [512]
  │
  ▼
  Adaptive Weighted Fusion (обучаемые α_s, α_t)
  → h_fused = α_s·h_s + α_t·h_t  [512]
  │
  ▼
  Classification Head → BCEWithLogitsLoss → P(fake) ∈ [0,1]
```

- **Spatial Branch**: EfficientNet-B4 — покадровые пространственные признаки
- **Temporal Branch**: EfficientNet-B0 + Temporal Transformer — межкадровая динамика через frame differences
- **Fusion**: adaptive weighted fusion (обучаемые веса)
- **Classification Head**: бинарная классификация real / fake

---

## Результаты

### Эксперимент 1: Ablation Study (DFDC02, T=16, seed=42, 30 epochs)

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| A1: Full (dual-path) | 0.9777 | 0.9253 | 0.9201 | 0.0749 | 18 |
| **A2: Spatial-only** | **0.9835** | **0.9273** | **0.9250** | **0.0606** | 29 |
| A3: Temporal-only | 0.9655 | 0.9152 | 0.9075 | 0.0950 | 14 |
| A4: Sequential (BiLSTM) | 0.9715 | 0.9172 | 0.9130 | 0.0787 | 29 |

### Эксперимент 2: Cross-dataset evaluation (DFDC02 → DFD01)

Модели обучены на DFDC02, протестированы на DFD01 (3420 видео, другой метод генерации).

| Модель | DFDC02 AUC | DFD01 AUC | Δ AUC |
|--------|-----------|-----------|-------|
| **A1: Full (dual-path)** | 0.9777 | **0.5531** | -0.4246 |
| A2: Spatial-only | 0.9835 | 0.5037 | -0.4798 |

**Вывод:** Dual-path модель обобщает лучше — temporal branch снижает падение AUC на 5.5%.

### Эксперимент 3: Multi-dataset training (DFDC02 + DFD01, T=16)

Обучение на объединённом датасете (6724 видео).

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| A3: Temporal-only | **0.9002** | 0.8760 | 0.9108 | 0.1944 | 22 |
| A1: Full (dual-path) | 0.8990 | 0.8770 | 0.9114 | 0.1937 | 10 |
| A2: Spatial-only | 0.8965 | 0.8909 | 0.9215 | 0.1717 | 10 |
| A4: Sequential (BiLSTM) | 0.8950 | 0.8740 | 0.9088 | 0.1876 | 23 |

### Эксперимент 4: 3-Dataset training (DFDC02 + DFD01 + CelebDF, T=16)

Обучение на трёх датасетах (10023 видео). Добавлен Celeb-DF v2 (3299 видео).

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| **A1: Full (dual-path)** | **0.9587** | **0.9436** | **0.9622** | **0.1211** | 16 |
| A3: Temporal-only | 0.9539 | 0.9150 | 0.9412 | 0.1168 | 17 |
| A2: Spatial-only | 0.9494 | 0.9303 | 0.9538 | 0.1479 | 18 |
| A4: Sequential (BiLSTM) | 0.9490 | 0.9203 | 0.9464 | 0.1393 | 20 |

**Вывод:** Full dual-path модель — лучшая по AUC (0.9587). Мультидатасетное обучение значительно улучшает генерализацию.

### Эксперимент 5: 3-Dataset training (DFDC02 + DFD01 + CelebDF, T=32)

Увеличение числа кадров до 32 (batch_size=8). Те же 3 датасета (~9953 видео после фильтрации).

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| **A1: Full (dual-path)** | **0.9943** | **0.9672** | **0.9776** | **0.0406** | 28 |
| A2: Spatial-only | 0.9904 | 0.9579 | 0.9714 | 0.0467 | 19 |
| A3: Temporal-only | 0.9883 | 0.9538 | 0.9687 | 0.0458 | 24 |
| A4: Sequential (BiLSTM) | 0.9844 | 0.9492 | 0.9656 | 0.0571 | 11 |

**Вывод:** T=32 значительно улучшает все метрики по сравнению с T=16. Full dual-path модель достигает AUC 0.9943, EER снижается с 0.1211 до 0.0406 (−66%). Ablation study подтверждён: full model лучше всех компонентов по отдельности.

### Сводная таблица: влияние T и числа датасетов (Full model)

| Конфигурация | T | Test AUC | EER |
|---|---|---|---|
| 1 датасет (DFDC02) | 16 | 0.9777 | 0.0749 |
| 2 датасета (DFDC02+DFD01) | 16 | 0.8990 | 0.1937 |
| 3 датасета (+CelebDF) | 16 | 0.9587 | 0.1211 |
| **3 датасета (+CelebDF)** | **32** | **0.9943** | **0.0406** |

---

## Датасеты

| Датасет | Видео | Real | Fake | Источник |
|---------|-------|------|------|----------|
| DFDC02 | 3293 | 1727 | 1566 | Facebook DFDC |
| DFD01 | 3431 | 363 | 3068 | Google/Jigsaw DFD |
| CelebDF v2 | 3299 | 589 | 2710 | Celeb-DF |

Данные предобработаны через MTCNN (face detection + crop):
- **T=16**: 16 equidistant frames per video, 224×224
- **T=32**: 32 equidistant frames per video, 224×224

---

## Воспроизведение экспериментов

### Вариант 1: Kaggle GPU (рекомендуется)

1. Загрузить preprocessed_DFDC02_16 как Kaggle Dataset
2. Открыть `kaggle-train.ipynb`
3. Включить GPU accelerator
4. Запустить ячейку — все 4 эксперимента выполнятся последовательно

### Вариант 2: Локально (одна модель)

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
# Загрузить видео → получить вердикт real/fake с вероятностью
```

---

## Структура проекта

```text
deepfake-detection/
├── config.py                  # Единая конфигурация (78+ параметров)
├── train.py                   # Обучение с warmup, fine-tuning, early stopping
├── evaluate.py                # Оценка и анализ ошибок
├── infer.py                   # Single-sample inference (video / frame folder)
├── app.py                     # Flask MVP (двуязычный RU/EN)
├── dataset.py                 # Video-level dataset + DataLoader
├── preprocess_videos.py       # Face-centric preprocessing (MTCNN)
├── utils.py                   # Метрики, seed, логирование
├── requirements.txt           # Зависимости
├── Dockerfile                 # Контейнеризация (Flask MVP)
│
├── models/
│   ├── __init__.py            # Factory: build_model(cfg)
│   ├── dual_path.py           # A1: Full model (spatial + temporal + fusion)
│   ├── spatial_branch.py      # EfficientNet-B4 backbone
│   ├── temporal_branch.py     # EfficientNet-B0 + Transformer
│   ├── spatial_only.py        # A2: Spatial-only ablation
│   ├── temporal_only.py       # A3: Temporal-only ablation
│   └── sequential.py          # A4: CNN → BiLSTM
│
├── splits/
│   ├── split_seed42.json          # Split: DFDC02 single-dataset
│   ├── split_multi_seed42.json    # Split: DFDC02+DFD01 multi-dataset
│   └── split_seed42_3ds_t16.json  # Split: 3-dataset (DFDC02+DFD01+CelebDF)
│
├── kaggle-train.ipynb             # Kaggle: ablation study (single-dataset)
├── kaggle-multi-train.ipynb       # Kaggle: multi-dataset training (2 datasets)
├── kaggle-train-3ds-t16.ipynb     # Kaggle: 3-dataset T=16 training
├── kaggle-train-3ds-t32-A1..A4.ipynb  # Kaggle: 3-dataset T=32 (4 parts)
├── kaggle-preprocess.ipynb        # Kaggle: DFDC02 preprocessing
├── kaggle-preprocess-dfd01-32.ipynb   # Kaggle: DFD01 T=32 preprocessing
├── kaggle-preprocess-celebdf.ipynb    # Kaggle: CelebDF preprocessing
├── kaggle-cross-eval.ipynb        # Kaggle: cross-dataset evaluation
│
├── EDA/
│   ├── VKR_EDA_Final_v5.ipynb # Exploratory Data Analysis
│   └── reports_v5/            # EDA отчёты, графики, таблицы
│
└── VKRDoc/
    ├── VKR_FINAL(v16).docx    # Текст диссертации
    ├── PROJECT_GUIDE.md        # Подробный гайд по проекту
    ├── Defense_Presentation_v2.pptx
    └── Technical_Prep_v2.pptx
```

---

## Workflow: локальная разработка + Kaggle GPU

```text
┌─ ЛОКАЛЬНО ──────────────────────────┐    ┌─ KAGGLE GPU ─────────────────────┐
│                                      │    │                                   │
│  git push ──────────────────────────────→ git clone (в notebook)             │
│                                      │    │                                   │
│  preprocess_videos.py ──→ upload ────────→ /kaggle/input/ (Dataset)          │
│                                      │    │                                   │
│                                      │    │  subprocess → train.py            │
│                                      │    │  (тот же код, те же параметры)    │
│                                      │    │                                   │
│  experiments/ ←── download .tar.gz ←─────── /kaggle/working/experiments/     │
│                                      │    │                                   │
│  app.py (auto-discover models)       │    │                                   │
│  evaluate.py, infer.py               │    │                                   │
└──────────────────────────────────────┘    └───────────────────────────────────┘
```

Kaggle notebooks — это **обёртки**, вызывающие `subprocess.run(["python", "train.py", ...])` с тем же кодом из GitHub.

---

## Переносимость чекпоинтов

Каждый `best_model.pt` — self-contained:

```python
checkpoint = torch.load("best_model.pt", map_location="cpu")
cfg = Config(); cfg.__dict__.update(checkpoint["config"])
model = build_model(cfg)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

Содержит: `model_state_dict`, `config` (все 78+ параметров), `metrics`, `epoch`.

---

## Принципы воспроизводимости

| Аспект | Реализация |
|--------|------------|
| Random seed | `set_seed(42)`: torch, numpy, random, CUDA, cudnn.deterministic=True |
| Data split | `splits/split_seed42.json` — фиксированный, стратифицированный по классам |
| Model loading | `strict=True` — точное совпадение state_dict |
| Config | Единый `config.py` с валидацией всех параметров |
| Fine-tuning | `add_param_group()` после unfreeze backbone (epoch 6+) |

---

## Docker

```bash
docker build -t deepfake-detection .
docker run -p 7860:7860 -v $(pwd)/experiments:/app/experiments deepfake-detection
# → http://localhost:7860
```

---

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| `CUDA out of memory` | Уменьшить `batch_size` (8 или 4) |
| MPS падает на inference | Использовать `--device cpu` |
| Dataset не распознаётся | Проверить структуру: `real/` и `fake/` с JPEG кадрами |
| Лицо не найдено в видео | Проверить, что видео содержит чёткое лицо |
| `No models found` в app.py | Скачать output тренировки с Kaggle и распаковать в `experiments/` |

### Получение весов модели

```bash
# Скачать output Kaggle notebook (нужен Kaggle CLI)
pip install kaggle
kaggle kernels output alexandertarakanov/kaggle-train-3ds-t16 -p ./kaggle_output/

# Распаковать
tar xzf kaggle_output/results_3ds_t16.tar.gz -C .
# → experiments/<exp_name>/best_model.pt появится
```

---

## Лицензия

Проект выполнен в рамках магистерской диссертации НИЯУ МИФИ, 2026.
