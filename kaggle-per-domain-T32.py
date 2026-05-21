"""
KAGGLE NOTEBOOK: Per-domain T=32 breakdown for A1 Full dual-path model
=======================================================================
Время: ~5 минут на P100/T4 GPU

ЧТО ДЕЛАЕТ:
  Прогоняет обученную модель A1 (3DS T=32) отдельно на каждом из 3 датасетов
  и выводит AUC/Accuracy/F1/EER per-domain.

ЧТО НУЖНО ПРИКРЕПИТЬ К НОУТБУКУ КАК DATASET:
  1. project-code/        — папка с config.py, dataset.py, models/, utils.py
  2. checkpoint/          — папка содержащая experiments/3ds_full_seed42_bs8_T32_adaptive/best_model.pt
  3. preprocessed-t32/    — три папки: preprocessed_DFDC02_32, preprocessed_DFD01_32, preprocessed_CelebDF_32

ВЫХОД (в /kaggle/working/):
  - per_domain_T32.json   — реальные числа AUC по доменам
  - per_domain_T32.png    — bar chart визуализация

Скачать оба файла, прислать мне для интеграции в ВКР.
"""

import sys, json, os, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

# === KAGGLE PATHS (поправь если у тебя другие имена datasets) ===
PROJECT_PATH = '/kaggle/input/project-code'
CKPT = '/kaggle/input/checkpoint/3ds_full_seed42_bs8_T32_adaptive/best_model.pt'

# Try several common dataset paths
DOMAIN_PATHS = {}
candidates = [
    ('DFDC02', '/kaggle/input/preprocessed-t32/preprocessed_DFDC02_32'),
    ('DFD01',  '/kaggle/input/preprocessed-t32/preprocessed_DFD01_32'),
    ('CelebDF', '/kaggle/input/preprocessed-t32/preprocessed_CelebDF_32'),
    ('DFDC02', '/kaggle/input/preprocessed-dfdc02-32/preprocessed_DFDC02_32'),
    ('DFD01',  '/kaggle/input/preprocessed-dfd01-32/preprocessed_DFD01_32'),
    ('CelebDF', '/kaggle/input/preprocessed-celebdf-32/preprocessed_CelebDF_32'),
]
for d, p in candidates:
    if d not in DOMAIN_PATHS and os.path.exists(p):
        DOMAIN_PATHS[d] = p

OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)

print('=== Found datasets ===')
for d, p in DOMAIN_PATHS.items():
    print(f'  {d}: {p}')
if not DOMAIN_PATHS:
    print('NO T=32 DATASETS FOUND! Check that you attached preprocessed-t32 dataset.')

sys.path.insert(0, PROJECT_PATH)
from config import Config
from dataset import DeepfakeVideoDataset, build_video_index
from models import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nDevice: {device}')

# Load model once
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = Config()
for k, v in ckpt['config'].items():
    setattr(cfg, k, v)
print(f'Model: {cfg.model_type}/{cfg.fusion_type}, T={cfg.num_frames}, val AUC was {ckpt.get("best_val_metrics", {}).get("auc", "?")}')

model = build_model(cfg).to(device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

results = {}
for domain, path in DOMAIN_PATHS.items():
    t0 = time.time()
    print(f'\n=== {domain} ===')
    cfg.dataset_root = path

    index = build_video_index(path)
    print(f'  videos: {len(index)}')

    dataset = DeepfakeVideoDataset(index, cfg, is_train=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    all_probs, all_y = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            spatial = batch['spatial'].to(device, non_blocking=True)
            temporal = batch['temporal'].to(device, non_blocking=True)
            labels = batch['label']
            out = model(spatial_input=spatial, temporal_input=temporal)
            logits = out[0] if isinstance(out, tuple) else out
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
            all_y.extend(labels.cpu().numpy().tolist())

    y = np.array(all_y)
    p = np.array(all_probs)

    m = {
        'n_total': int(len(y)),
        'n_real': int((y==0).sum()),
        'n_fake': int((y==1).sum()),
        'auc': float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        'acc': float(accuracy_score(y, (p>=0.5).astype(int))),
        'bal_acc': float(balanced_accuracy_score(y, (p>=0.5).astype(int))) if len(set(y)) > 1 else None,
        'f1': float(f1_score(y, (p>=0.5).astype(int), zero_division=0)),
        'mean_prob': float(p.mean()),
        'elapsed_s': time.time() - t0,
    }
    results[domain] = m
    print(f'  AUC={m["auc"]:.4f}, Acc={m["acc"]:.4f}, F1={m["f1"]:.4f}, time={m["elapsed_s"]:.0f}s')

with open(f'{OUT}/per_domain_T32.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n✓ {OUT}/per_domain_T32.json saved')

# Build comparison chart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

domains = list(results.keys())
aucs = [results[d]['auc'] for d in domains]
accs = [results[d]['acc'] for d in domains]
f1s = [results[d]['f1'] for d in domains]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(domains))
w = 0.27
b1 = ax.bar(x-w, aucs, w, label='AUC', color='#1f77b4')
b2 = ax.bar(x,   accs, w, label='Accuracy', color='#2ca02c')
b3 = ax.bar(x+w, f1s,  w, label='F1', color='#ff7f0e')
for bars in [b1, b2, b3]:
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f'{b.get_height():.3f}', ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(domains, fontsize=12)
ax.set_ylabel('Метрика', fontsize=12)
ax.set_title('Per-domain breakdown модели A1 Full dual-path (3DS T=32)', fontsize=13)
ax.set_ylim([0, 1.15])
ax.legend(loc='upper right'); ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{OUT}/per_domain_T32.png', dpi=180, bbox_inches='tight')
plt.close()
print(f'✓ {OUT}/per_domain_T32.png saved')

print('\n=== DONE — download both files from /kaggle/working/ ===')
