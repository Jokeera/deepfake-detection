# Deepfake Detection — Полное руководство по проекту

> Этот документ описывает ВСЁ, что нужно знать о проекте для подготовки к защите ВКР.
> Читатель — человек, который видит проект впервые.

---

## 1. Что это за проект

Магистерская диссертация НИЯУ МИФИ, 2026.
Тема: **детектирование deepfake-видео** на основе пространственно-временного анализа лицевых областей.

Ключевая идея: не классифицировать отдельные кадры, а анализировать **видео целиком** (video-level detection),
объединяя пространственные артефакты на лицах и временные аномалии между кадрами.

### Архитектура модели (dual-path)

```
Видео (T кадров)
      │
      ├──► Spatial Branch (EfficientNet-B4, 224×224)
      │        Покадровые признаки → mean pool → проекция [512]
      │
      └──► Temporal Branch (EfficientNet-B0 + Transformer, 128×128)
               Frame differences (T-1 штук) → признаки → Transformer → проекция [512]
      │
      ▼
  Adaptive Weighted Fusion
      α_s · h_spatial + α_t · h_temporal    (α обучаемые, softmax)
      │
      ▼
  Classification Head
      Linear(512→256) → BN → ReLU → Dropout(0.3)
      Linear(256→128) → BN → ReLU → Dropout(0.3)
      Linear(128→1) → BCEWithLogitsLoss
      │
      ▼
  Результат: probability_fake ∈ [0, 1]
```

### Ablation study (4 обязательных эксперимента)

| ID | Модель | Что тестирует |
|----|--------|---------------|
| A1 | Full (dual-path + adaptive fusion) | Основная архитектура |
| A2 | Spatial-only | Достаточно ли пространственного анализа? |
| A3 | Temporal-only | Достаточно ли временного анализа? |
| A4 | Sequential (CNN→BiLSTM) | Альтернатива Transformer |

Дополнительно реализованы (но не обучены): A5 Concat fusion, A6 Gated fusion.

---

## 2. Структура файлов проекта

```
deepfake_detection/
│
├── config.py                  # Все параметры (78+), единая точка настройки
├── dataset.py                 # Загрузка данных, сплиты, аугментации
├── train.py                   # Обучение (warmup → fine-tuning → early stopping)
├── evaluate.py                # Оценка на test split
├── infer.py                   # Инференс одного видео (CLI)
├── app.py                     # Flask веб-интерфейс (RU/EN)
├── preprocess_videos.py       # Извлечение лиц из сырых видео (MTCNN)
├── run_experiments.py         # Запуск всех 4-6 экспериментов
├── plot_training_curves.py    # Графики обучения
├── utils.py                   # set_seed(), compute_metrics(), логирование
├── scan_dataset.py            # Диагностика структуры датасета
│
├── models/
│   ├── __init__.py            # Фабрика build_model(cfg) — реестр моделей
│   ├── dual_path.py           # A1: DualPathModel + 3 типа fusion
│   ├── spatial_branch.py      # EfficientNet-B4 backbone
│   ├── temporal_branch.py     # EfficientNet-B0 + Temporal Transformer
│   ├── spatial_only.py        # A2: только spatial ветка
│   ├── temporal_only.py       # A3: только temporal ветка
│   └── sequential.py          # A4: CNN→BiLSTM
│
├── splits/
│   └── split_seed42.json      # Фиксированное разбиение 70/15/15
│
├── experiments/               # Результаты обучения (создаётся автоматически)
│   └── dfdc02_full_seed42_bs16_T16_adaptive/
│       ├── best_model.pt      # Чекпоинт (веса + config + метрики)
│       ├── history.csv        # Метрики по эпохам
│       └── metrics.json       # Итоговые метрики
│
├── requirements.txt           # Зависимости Python
├── Dockerfile                 # Контейнеризация Flask MVP
├── README.md                  # Описание проекта
│
├── kaggle-train.ipynb         # Kaggle: запуск ablation study
├── kaggle-multi-train.ipynb   # Kaggle: multi-dataset training
├── kaggle-preprocess.ipynb    # Kaggle: препроцессинг больших датасетов
├── kaggle-cross-eval.py       # Cross-dataset evaluation скрипт
│
├── EDA/
│   ├── VKR_EDA_Final_v5.ipynb   # Exploratory Data Analysis (latest)
│   └── reports_v5/              # EDA отчёты, графики, таблицы
│
└── VKRDoc/                    # Документация ВКР
    ├── VKR_FINAL(v12).docx    # Текст диссертации
    ├── Defense_Presentation_v2.pptx
    ├── Technical_Prep_v2.pptx
    └── PROJECT_GUIDE.md       # ← Этот файл
```

---

## 3. Данные

### Датасет DFDC02

- **Источник**: DeepFake Detection Challenge (Facebook/Meta)
- **Объём**: 3292 видео (1566 fake / 1727 real)
- **Формат после препроцессинга**: папки с 16 кадрами (лица, вырезанные MTCNN)
- **Разбиение** (split_seed42.json): 2303 train / 493 val / 496 test (70/15/15)

### Датасет DFD01

- **Источник**: DeepFakeDetection (Google)
- **Используется для**: cross-dataset evaluation (модели, обученные на DFDC02)
- **Объём**: 3420 видео

### Структура на диске

```
data/preprocessed_data/preprocessed_DFDC02_16/
├── real/
│   ├── video_001/
│   │   ├── 0000.jpg    # 224×224 лицо, кадр 0
│   │   ├── 0001.jpg    # кадр 1
│   │   └── ...0015.jpg # кадр 15
│   └── video_002/
│       └── ...
└── fake/
    ├── video_003/
    │   └── ...0015.jpg
    └── ...
```

Каждое видео = папка с 16 JPEG-файлами (лицевые кропы).

### Как создать данные из сырых видео

```bash
python preprocess_videos.py \
  --input_root ./data/raw_DFDC \
  --output_root ./data/preprocessed_data/preprocessed_DFDC02_16 \
  --max_frames 16 \
  --output_size 224 \
  --device auto
```

Pipeline: видео → uniform sampling 16 кадров → MTCNN face detection → crop + margin 20% → resize 224×224 → JPEG 95%.

---

## 4. Конфигурация (config.py)

Все параметры хранятся в одном `@dataclass Config`. Значения по умолчанию:

### Данные
| Параметр | Значение | Назначение |
|----------|----------|------------|
| `num_frames` | 16 | Кадров на видео |
| `spatial_size` | 224 | Размер входа spatial ветки |
| `temporal_size` | 128 | Размер входа temporal ветки |
| `train_ratio` | 0.70 | Доля train |
| `val_ratio` | 0.15 | Доля validation |
| `split_by_video` | True | Сплит по видео, не по кадрам |

### Модель
| Параметр | Значение | Назначение |
|----------|----------|------------|
| `model_type` | "full" | full/spatial/temporal/sequential |
| `spatial_backbone` | "efficientnet_b4" | Backbone spatial ветки |
| `temporal_backbone` | "efficientnet_b0" | Backbone temporal ветки |
| `projection_dim` | 512 | Размерность пространства fusion |
| `transformer_layers` | 2 | Слои Transformer в temporal |
| `transformer_heads` | 4 | Головы внимания |
| `transformer_ff_dim` | 1024 | FFN dimension в Transformer |
| `fusion_type` | "adaptive" | adaptive/concat/gate |
| `head_dropout` | 0.3 | Dropout в классификаторе |

### Обучение
| Параметр | Значение | Назначение |
|----------|----------|------------|
| `batch_size` | 8 | Размер батча (на Kaggle: 16) |
| `max_epochs` | 30 | Максимум эпох |
| `lr_backbone` | 1e-4 | LR для backbone |
| `lr_head` | 3e-4 | LR для головы |
| `patience` | 7 | Early stopping patience |
| `warmup_epochs` | 5 | Эпох с замороженным spatial backbone |
| `unfreeze_last_n_blocks` | 2 | Блоков backbone для fine-tuning |
| `seed` | 42 | Random seed |
| `use_amp` | False | Mixed precision (только CUDA) |

### Аугментации (clip-consistent)
| Параметр | Значение | Назначение |
|----------|----------|------------|
| `augment_hflip` | True | Горизонтальный flip |
| `augment_brightness` | 0.1 | Яркость ±10% |
| `augment_contrast` | 0.1 | Контраст ±10% |
| `augment_jpeg_prob` | 0.3 | Вероятность JPEG-артефактов |
| `clip_consistent_jpeg` | True | Одинаковое quality на весь клип |

**Важно**: все аугментации применяются **clip-consistent** — одинаковые параметры для всех кадров одного видео.

---

## 5. Обучение

### Фазы обучения

```
Эпохи 1-5:   WARMUP
              Spatial backbone заморожен
              Обучаются: temporal branch + fusion + head
              LR: LinearLR (1e-3 → 1.0)

Эпохи 6-30:  FINE-TUNING
              Разморожены последние 2 блока spatial backbone
              Обучается вся модель
              LR: CosineAnnealingLR (cosine decay)

              Early stopping: если AUC на val не улучшается 7 эпох — стоп
```

### Запуск одной модели

```bash
python train.py \
  --model_type full \
  --fusion_type adaptive \
  --seed 42 \
  --batch_size 16 \
  --max_epochs 30 \
  --dataset_root ./data/preprocessed_data/preprocessed_DFDC02_16 \
  --dataset_name dfdc02 \
  --device auto \
  --output_dir ./experiments
```

### Запуск всех экспериментов

```bash
python run_experiments.py \
  --dataset_root ./data/preprocessed_data/preprocessed_DFDC02_16 \
  --dataset_name dfdc02 \
  --output_dir ./experiments \
  --level mandatory \
  --device auto \
  --batch_size 16 \
  --max_epochs 30 \
  --seed 42
```

Это последовательно обучит A1, A2, A3, A4 (~4-8 часов на T4/P100).

### Что получается после обучения

Для каждого эксперимента создаётся папка:
```
experiments/dfdc02_full_seed42_bs16_T16_adaptive/
├── best_model.pt       # Чекпоинт (всё внутри)
├── history.csv         # Метрики по эпохам
├── metrics.json        # Итоговые результаты
├── confusion_matrix_test.png
├── roc_curve_test.png
└── ...
```

### Формат чекпоинта (best_model.pt)

```python
{
    "model_state_dict": OrderedDict(...),   # Веса модели
    "config": {                              # Полный Config.__dict__
        "model_type": "full",
        "fusion_type": "adaptive",
        "num_frames": 16,
        "spatial_size": 224,
        "batch_size": 16,
        ...                                  # все 78+ параметров
    },
    "metrics": {                             # Результаты
        "test_auc": 0.9777,
        "test_acc": 0.9253,
        ...
    },
    "epoch": 18                              # Best epoch
}
```

**Самодостаточность**: из одного файла `best_model.pt` можно полностью восстановить модель:
```python
checkpoint = torch.load("best_model.pt", map_location="cpu")
cfg = Config()
cfg.__dict__.update(checkpoint["config"])
model = build_model(cfg)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

---

## 6. Оценка и инференс

### Оценка на test split

```bash
python evaluate.py \
  --checkpoint ./experiments/dfdc02_full_seed42_bs16_T16_adaptive/best_model.pt \
  --split test \
  --device auto
```

Выдаёт: AUC, Accuracy, F1, EER, AP, Balanced Accuracy, confusion matrix, ROC curve.

### Cross-dataset evaluation (DFDC02 → DFD01)

```bash
python kaggle-cross-eval.py \
  --checkpoints_dir ./experiments \
  --cross_dataset ./data/preprocessed_data/preprocessed_DFD01_16 \
  --output_dir ./cross_eval_results \
  --device auto
```

Оценивает все модели из `experiments/` на другом датасете. Показывает обобщающую способность.

### Инференс одного видео

```bash
# Из сырого видео (автоматическое извлечение лиц)
python infer.py \
  --checkpoint ./experiments/.../best_model.pt \
  --input /path/to/video.mp4 \
  --device auto

# Из папки с кадрами
python infer.py \
  --checkpoint ./experiments/.../best_model.pt \
  --input /path/to/frames_folder/ \
  --device auto
```

### Flask веб-интерфейс

```bash
python app.py
# Открыть http://127.0.0.1:7860
```

- Автоматически находит все модели в `./experiments/`
- Поддерживает загрузку видео и папок с кадрами
- Двуязычный интерфейс (RU/EN)
- Показывает probability_fake, fusion weights, детали модели

---

## 7. Workflow: локальная машина + Kaggle

### Почему две среды?

Обучение 4 моделей × 30 эпох требует GPU с 14+ GB VRAM. При отсутствии мощного GPU используется Kaggle (бесплатный T4/P100). Это **стандартная практика** в ML, а не "костыль".

### Что где выполняется

| Задача | Локально | Kaggle GPU |
|--------|----------|------------|
| Разработка кода | config.py, models/, train.py | — |
| Препроцессинг (<1000 видео) | preprocess_videos.py | — |
| Препроцессинг (>5000 видео) | — | kaggle-preprocess.ipynb |
| Обучение (30 эпох × 4 модели) | При наличии GPU | kaggle-train.ipynb |
| Multi-dataset training | При наличии GPU | kaggle-multi-train.ipynb |
| Cross-dataset evaluation | kaggle-cross-eval.py | kaggle-cross-eval.py |
| Визуализация | plot_training_curves.py | — |
| Flask MVP / инференс | app.py, infer.py | — |
| EDA | EDA/*.ipynb | — |

### Как это работает

```
┌─ ЛОКАЛЬНО ─────────────────────────┐    ┌─ KAGGLE GPU ──────────────────────┐
│                                     │    │                                    │
│  git push ─────────────────────────────→ git clone (в notebook)              │
│                                     │    │                                    │
│  preprocess_videos.py ─→ upload ───────→ /kaggle/input/ (Dataset)           │
│                                     │    │                                    │
│                                     │    │  subprocess → train.py            │
│                                     │    │  (тот же код, те же параметры)    │
│                                     │    │                                    │
│  experiments/ ←── download .tar.gz ←───── /kaggle/working/experiments/      │
│                                     │    │                                    │
│  app.py (auto-discover models)      │    │                                    │
│  evaluate.py, plot_*.py             │    │                                    │
└─────────────────────────────────────┘    └────────────────────────────────────┘
```

**Принцип**: один код — две среды. Kaggle notebooks — обёртки, вызывающие тот же `train.py` через `subprocess`.

### Kaggle notebooks

1. **kaggle-train.ipynb** — запуск 4 экспериментов ablation study
2. **kaggle-multi-train.ipynb** — обучение на объединённом DFDC02+DFD01
3. **kaggle-preprocess.ipynb** — препроцессинг больших датасетов на GPU

Каждый notebook делает:
```
git clone repo → pip install → subprocess.run(["python", "train.py", ...]) → tar.gz результатов
```

### Перенос результатов

1. На Kaggle: результаты упаковываются в `.tar.gz`
2. Скачать архив → распаковать в `./experiments/`
3. `app.py` автоматически находит все `best_model.pt` — готово к инференсу

---

## 8. Результаты экспериментов

### DFDC02 (seed=42, 30 epochs, batch_size=16)

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| A1: Full (dual-path) | 0.9777 | 0.9253 | 0.9201 | 0.0749 | 18 |
| **A2: Spatial-only** | **0.9835** | **0.9273** | **0.9250** | **0.0606** | 29 |
| A3: Temporal-only | 0.9655 | 0.9152 | 0.9075 | 0.0950 | 14 |
| A4: Sequential (BiLSTM) | 0.9715 | 0.9172 | 0.9130 | 0.0787 | 29 |

### Cross-dataset (DFDC02 → DFD01)

| Модель | DFDC02 AUC | DFD01 AUC | Delta AUC |
|--------|-----------|-----------|-----------|
| **A1: Full (dual-path)** | 0.9777 | **0.5531** | -0.4246 |
| A2: Spatial-only | 0.9835 | 0.5037 | -0.4798 |

**Вывод**: dual-path обобщает лучше. Temporal branch снижает падение AUC на 5.5% при cross-dataset.
A2 лучше на in-domain, но A1 лучше при переносе на другой датасет.

### Multi-dataset training (DFDC02 + DFD01, T=16)

Обучение на объединённом датасете (~6700 видео). Проверяем, улучшит ли дополнительные данные обобщение.

| Модель | Test AUC | Test Acc | Test F1 | EER | Best Epoch |
|--------|----------|----------|---------|-----|------------|
| A3: Temporal-only | **0.9002** | 0.8760 | 0.9108 | 0.1944 | 22 |
| A1: Full (dual-path) | 0.8990 | 0.8770 | 0.9114 | 0.1937 | 10 |
| A2: Spatial-only | 0.8965 | 0.8909 | 0.9215 | 0.1717 | 10 |
| A4: Sequential (BiLSTM) | 0.8950 | 0.8740 | 0.9088 | 0.1876 | 23 |

> **Примечание**: AUC на multi-dataset ниже, чем на DFDC02, потому что DFD01 имеет сильный дисбаланс классов
> (363 real vs 3068 fake) и другую природу генерации. Задача сложнее, но модель учится обобщать.

### Эксперимент T=32 (увеличенное число кадров)

32 кадра на видео (vs. 16 в базовом эксперименте). Проверяем влияние временного разрешения.
Препроцессинг: DFDC02 T=32 завершён (3240 видео), DFD01 T=32 в процессе.

*Результаты тренировки появятся после завершения.*

---

## 9. Воспроизводимость

| Что | Как обеспечено |
|-----|----------------|
| Случайность | `set_seed(42)`: torch, numpy, random, CUDA, cudnn.deterministic=True |
| Разбиение данных | `splits/split_seed42.json` — стратифицированный, по видео |
| Загрузка модели | `strict=True` — точное совпадение state_dict |
| Конфигурация | config.py + сохранение в чекпоинт |
| Fine-tuning | `add_param_group()` после unfreeze backbone |
| Аугментации | clip-consistent (одинаковые параметры на все кадры видео) |

Результаты воспроизведены дважды с идентичными метриками.

---

## 10. Как подготовиться к защите

### Ключевые вопросы и ответы

**Q: Почему dual-path, а не просто spatial?**
A: На in-domain (DFDC02) spatial-only даже лучше (AUC 0.9835 vs 0.9777). Но при cross-dataset (DFDC02→DFD01) dual-path обобщает значительно лучше (0.5531 vs 0.5037). Temporal branch захватывает артефакты, специфичные не для конкретного датасета, а для процесса генерации deepfake.

**Q: Почему EfficientNet, а не ViT / ResNet?**
A: EfficientNet-B4 — оптимальный баланс accuracy/parameters для fine-tuning. ViT требует больше данных для предобучения. ResNet уступает по качеству при сопоставимом размере.

**Q: Почему frame differences, а не оптический поток?**
A: Frame differences — лёгкая аппроксимация оптического потока, не требующая дополнительных моделей. Вычисляются "на лету" в dataset.py.

**Q: Как работает adaptive fusion?**
A: Обучаемые веса через softmax: `alpha = softmax(Linear(concat(h_s, h_t)))`. Модель сама решает, какой ветке доверять больше для каждого сэмпла.

**Q: Почему warmup 5 эпох?**
A: Spatial backbone (EfficientNet-B4) предобучен на ImageNet. Если сразу обучать все веса, случайные градиенты от нерабочей temporal ветки "испортят" хорошие пространственные представления. Warmup даёт temporal ветке время стабилизироваться.

**Q: Cross-dataset AUC 0.55 — это плохо?**
A: Это ожидаемо при полном отсутствии DFD01 в обучении. Domain gap между DFDC02 (метод генерации, качество) и DFD01 огромен. Важно, что dual-path обобщает *лучше* spatial-only. Направления улучшения: multi-dataset training, domain adaptation.

**Q: Зачем multi-dataset training?**
A: Cross-dataset показал, что модель плохо обобщает (AUC 0.55). Multi-dataset training на DFDC02+DFD01 учит модель на разных методах генерации, что должно улучшить обобщение. AUC ниже (0.899 vs 0.978), но модель более робастная.

**Q: Зачем T=32?**
A: 16 кадров — компромисс скорость/качество. 32 кадра дают temporal ветке больше информации о межкадровой динамике. Проверяем, улучшит ли это качество.

**Q: Зачем нужен Kaggle?**
A: Обучение 4 моделей × 30 эпох при batch_size=16 требует GPU с 14+ GB VRAM. Kaggle даёт бесплатный T4/P100. Код одинаковый — notebooks просто вызывают `train.py`.

### Файлы для изучения (в порядке приоритета)

1. `config.py` — понять все параметры
2. `models/dual_path.py` — архитектура
3. `train.py` — процесс обучения
4. `dataset.py` — как данные подаются в модель
5. `README.md` — обзор и CLI команды
6. `VKRDoc/VKR_FINAL(v12).docx` — текст диссертации

### Команды для быстрого старта

```bash
# Установка
pip install -r requirements.txt

# Проверка структуры данных
python scan_dataset.py ./data/preprocessed_data/preprocessed_DFDC02_16

# Инференс (если есть чекпоинт)
python infer.py --checkpoint ./experiments/.../best_model.pt --input video.mp4

# Веб-интерфейс
python app.py
```

---

## 11. Зависимости (requirements.txt)

```
numpy>=1.26.0,<2.0          # Массивы
torch>=2.0.0,<3.0           # PyTorch
torchvision>=0.15.0,<1.0    # Трансформации изображений
timm>=0.9.0,<2.0            # EfficientNet backbone'ы
facenet-pytorch>=2.5.2,<3.0 # MTCNN face detection
pandas>=2.1.0               # Таблицы данных
scipy>=1.11.0,<1.17         # Научные вычисления
scikit-learn>=1.3.0          # Метрики (AUC, F1, EER)
opencv-python>=4.8.0         # Чтение видео, обработка изображений
Pillow>=10.0.0               # Загрузка JPEG
tqdm>=4.65.0                 # Прогресс-бары
matplotlib>=3.7.0            # Графики
seaborn>=0.12.0              # Визуализация
flask>=3.0.0                 # Веб-интерфейс
```

---

## 12. Docker

```bash
docker build -t deepfake-detection .
docker run -p 7860:7860 -v $(pwd)/experiments:/app/experiments deepfake-detection
```

Открыть http://localhost:7860 — Flask MVP с автоматическим обнаружением моделей.

---

## 13. Глоссарий

| Термин | Значение |
|--------|----------|
| **AUC** | Area Under ROC Curve — основная метрика качества |
| **EER** | Equal Error Rate — порог, где FPR = FNR |
| **Ablation study** | Эксперименты, где отключаются компоненты для оценки их вклада |
| **Cross-dataset** | Оценка модели на данных, которых не было при обучении |
| **Frame differences** | Попиксельная разность соседних кадров — вход temporal ветки |
| **Fusion** | Объединение признаков двух веток в единый вектор |
| **MTCNN** | Multi-task Cascaded CNN — детектор лиц |
| **Warmup** | Начальные эпохи с замороженным backbone |
| **Fine-tuning** | Дообучение предобученного backbone на целевой задаче |
| **BCEWithLogitsLoss** | Binary Cross-Entropy с встроенным sigmoid |
| **Video-level** | Одна метка (real/fake) на всё видео, не на отдельный кадр |
