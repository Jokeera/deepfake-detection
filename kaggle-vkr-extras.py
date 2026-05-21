"""
KAGGLE NOTEBOOK: ALL VKR EXTRAS in one run
==========================================
Запустить на Kaggle GPU (P100 или T4). Время ~25-40 мин.

ПРИКРЕПИ ДАТАСЕТЫ К ноутбуку:
  1. checkpoints (вся папка experiments/ + kaggle_output/experiments/experiments/)
  2. preprocessed-deepfake (preprocessed_DFDC02_16/, preprocessed_DFD01_16/, preprocessed_CelebDF_16/)
  3. preprocessed-t32 (preprocessed_DFDC02_32/, preprocessed_DFD01_32/, preprocessed_CelebDF_32/) — опционально

ВЫХОД (в /kaggle/working/):
  - cross_eval_DFD01.json    — A1/A2/A3/A4 cross-dataset DFDC02→DFD01
  - sanity_A1.json           — verify A1 → DFD01 ≈ 0.5531 (pipeline check)
  - per_domain_T32.json      — A1 @ T=32 разбивка по 3 датасетам
  - pr_curves_DFD01.png      — Precision-Recall кривые
  - tsne_dfd01.png           — t-SNE feature space A1 на DFD01
  - tsne_3datasets.png       — t-SNE A1 на 3 датасетах (показывает доменное разделение)
  - confusion_matrices.png   — 2x2 confusion для A1/A2/A3/A4 на DFD01
  - bootstrap_CI.json        — bootstrap доверительные интервалы AUC
"""

import sys, json, os, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === KAGGLE PATHS — ADJUST IF NEEDED ===
PROJECT_PATH = '/kaggle/input/project-code'  # uploaded project code
DATA_T16 = '/kaggle/input/preprocessed-deepfake'   # T=16 datasets
DATA_T32 = '/kaggle/input/preprocessed-t32'        # T=32 (optional)
CKPT_ROOT = '/kaggle/input/checkpoints'

OUT = '/kaggle/working'
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, PROJECT_PATH)
from config import Config
from dataset import DeepfakeVideoDataset, build_video_index
from models import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


CKPTS = {
    'A1_full_3DS_T32':       f'{CKPT_ROOT}/3ds_full_seed42_bs8_T32_adaptive/best_model.pt',
    'A1_full_3DS_T16':       f'{CKPT_ROOT}/3ds_full_seed42_bs8_T16_adaptive/best_model.pt',
    'A1_full_DFDC02_T16':    f'{CKPT_ROOT}/dfdc02_full_seed42_bs16_T16_adaptive/best_model.pt',
    'A2_spatial_DFDC02':     f'{CKPT_ROOT}/dfdc02_spatial_seed42_bs16_T16/best_model.pt',
    'A3_temporal_DFDC02':    f'{CKPT_ROOT}/dfdc02_temporal_seed42_bs16_T16/best_model.pt',
    'A4_sequential_DFDC02':  f'{CKPT_ROOT}/dfdc02_sequential_seed42_bs16_T16/best_model.pt',
}


# === HELPERS ===
def load_model(ckpt_path, target_root):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config()
    for k, v in ckpt['config'].items(): setattr(cfg, k, v)
    cfg.dataset_root = target_root
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def infer(model, loader, capture_embeddings=False):
    embeds = []
    handle = None
    if capture_embeddings and hasattr(model, 'fusion'):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            embeds.append(h.detach().cpu().numpy())
        handle = model.fusion.register_forward_hook(hook)
    
    probs, ys, alphas = [], [], []
    with torch.no_grad():
        for batch in loader:
            spatial = batch['spatial'].to(device, non_blocking=True)
            temporal = batch['temporal'].to(device, non_blocking=True)
            labels = batch['label']
            out = model(spatial_input=spatial, temporal_input=temporal)
            if isinstance(out, tuple):
                logits, alpha = out[0], out[1] if len(out) > 1 else None
                if alpha is not None:
                    alphas.extend(alpha.cpu().numpy().tolist())
            else:
                logits = out
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
            ys.extend(labels.cpu().numpy().tolist())
    
    if handle: handle.remove()
    result = {'y': np.array(ys), 'p': np.array(probs)}
    if alphas: result['alpha'] = np.array(alphas)
    if embeds: result['embeddings'] = np.concatenate(embeds, axis=0)
    return result


def metrics(y, p):
    m = {}
    if len(set(y)) > 1:
        m['auc'] = float(roc_auc_score(y, p))
        m['ap'] = float(average_precision_score(y, p))
    preds = (p >= 0.5).astype(int)
    m['acc'] = float(accuracy_score(y, preds))
    m['f1'] = float(f1_score(y, preds, zero_division=0))
    if len(set(y)) > 1:
        m['bal_acc'] = float(balanced_accuracy_score(y, preds))
    cm = confusion_matrix(y, preds, labels=[0, 1])
    m['confusion'] = cm.tolist()
    return m


def bootstrap_auc(y, p, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    aucs = np.array(aucs)
    return {
        'auc_mean': float(aucs.mean()),
        'auc_std': float(aucs.std()),
        'ci_low': float(np.percentile(aucs, 2.5)),
        'ci_high': float(np.percentile(aucs, 97.5)),
        'n_boot': len(aucs),
    }


# =============================================================
# EXP 1: Cross-eval DFDC02→DFD01 for A1/A2/A3/A4 + A1 sanity
# =============================================================
print('\n' + '='*60)
print('EXP 1: Cross-eval + A1 sanity on DFD01')
print('='*60)

cross_results = {}
predictions_storage = {}  # for PR curves + confusion matrices

dfd01_path = f'{DATA_T16}/preprocessed_DFD01_16'
for arch, ckpt_key in [
    ('A1_sanity', 'A1_full_DFDC02_T16'),
    ('A2', 'A2_spatial_DFDC02'),
    ('A3', 'A3_temporal_DFDC02'),
    ('A4', 'A4_sequential_DFDC02'),
]:
    t0 = time.time()
    print(f'\n→ {arch} ({ckpt_key})...')
    try:
        model, cfg = load_model(CKPTS[ckpt_key], dfd01_path)
        index = build_video_index(dfd01_path)
        dataset = DeepfakeVideoDataset(index, cfg, is_train=False)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
        capture = (arch == 'A1_sanity')  # only A1 gets embeddings
        r = infer(model, loader, capture_embeddings=capture)
        m = metrics(r['y'], r['p'])
        m['boot'] = bootstrap_auc(r['y'], r['p'])
        cross_results[arch] = m
        predictions_storage[arch] = {'y': r['y'], 'p': r['p']}
        if 'embeddings' in r:
            np.save(f'{OUT}/A1_embeddings_dfd01.npy', r['embeddings'])
            np.save(f'{OUT}/A1_labels_dfd01.npy', r['y'])
        elapsed = time.time() - t0
        print(f'   AUC={m.get("auc","N/A"):.4f}, AP={m.get("ap","N/A"):.4f}, Acc={m["acc"]:.4f}, F1={m["f1"]:.4f}, elapsed={elapsed:.0f}s')
        print(f'   Bootstrap 95% CI: [{m["boot"]["ci_low"]:.4f}, {m["boot"]["ci_high"]:.4f}]')
    except Exception as e:
        import traceback
        print(f'   FAILED: {e}\n{traceback.format_exc()}')
        cross_results[arch] = {'error': str(e)}

with open(f'{OUT}/cross_eval_DFD01.json', 'w') as f:
    json.dump(cross_results, f, indent=2)

# Sanity report
sanity_auc = cross_results.get('A1_sanity', {}).get('auc')
if sanity_auc:
    diff = abs(sanity_auc - 0.5531)
    print(f'\n*** SANITY: A1 → DFD01 AUC = {sanity_auc:.4f} (expected 0.5531, diff {diff:.4f})')
    print(f'    Pipeline OK: {diff < 0.02}')


# =============================================================
# EXP 2: Per-domain T=32 breakdown for A1 3DS
# =============================================================
print('\n' + '='*60)
print('EXP 2: Per-domain breakdown A1 @ T=32')
print('='*60)

per_domain = {}
for domain, dirname in [('DFDC02', 'preprocessed_DFDC02_32'),
                       ('DFD01', 'preprocessed_DFD01_32'),
                       ('CelebDF', 'preprocessed_CelebDF_32')]:
    path = f'{DATA_T32}/{dirname}'
    if not os.path.exists(path):
        print(f'  {domain}: T=32 data not found at {path} — skipping')
        per_domain[domain] = {'skipped': 'T=32 data not uploaded'}
        continue
    try:
        model, cfg = load_model(CKPTS['A1_full_3DS_T32'], path)
        index = build_video_index(path)
        dataset = DeepfakeVideoDataset(index, cfg, is_train=False)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        r = infer(model, loader)
        m = metrics(r['y'], r['p'])
        m['boot'] = bootstrap_auc(r['y'], r['p']) if len(set(r['y'])) > 1 else None
        per_domain[domain] = m
        print(f'  {domain}: AUC={m.get("auc","N/A"):.4f}, Acc={m["acc"]:.4f}, n={len(r["y"])}')
    except Exception as e:
        print(f'  {domain}: ERROR {e}')
        per_domain[domain] = {'error': str(e)}

with open(f'{OUT}/per_domain_T32.json', 'w') as f:
    json.dump(per_domain, f, indent=2)


# =============================================================
# EXP 3: PR curves + Confusion matrices (A1/A2/A3/A4 on DFD01)
# =============================================================
print('\n' + '='*60)
print('EXP 3: PR curves + Confusion matrices')
print('='*60)

# PR curves
fig, ax = plt.subplots(figsize=(8, 6))
for arch in ['A1_sanity', 'A2', 'A3', 'A4']:
    if arch in predictions_storage:
        d = predictions_storage[arch]
        y, p = d['y'], d['p']
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        label_name = 'A1 Full' if arch == 'A1_sanity' else arch
        ax.plot(rec, prec, label=f'{label_name} (AP={ap:.3f})', linewidth=2)

# Baseline: class prior
y_any = predictions_storage.get('A1_sanity', {}).get('y', np.array([]))
if len(y_any) > 0:
    baseline = (y_any == 1).mean()
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Random ({baseline:.3f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall кривые на DFD01 (cross-dataset, T=16)\nМодели обучены на DFDC02')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/pr_curves_DFD01.png', dpi=180, bbox_inches='tight')
plt.close()
print('  ✓ pr_curves_DFD01.png saved')

# Confusion matrices 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for i, arch in enumerate(['A1_sanity', 'A2', 'A3', 'A4']):
    ax = axes[i//2, i%2]
    if arch in predictions_storage:
        d = predictions_storage[arch]
        y, p = d['y'], d['p']
        preds = (p >= 0.5).astype(int)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real (pred)', 'Fake (pred)'])
        ax.set_yticklabels(['Real (true)', 'Fake (true)'])
        for ii in range(2):
            for jj in range(2):
                ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center', fontsize=16, fontweight='bold',
                        color='white' if cm[ii, jj] > cm.max()/2 else 'black')
        auc = cross_results.get(arch, {}).get('auc', 'N/A')
        auc_str = f'{auc:.3f}' if isinstance(auc, float) else auc
        label_name = 'A1 Full' if arch == 'A1_sanity' else arch
        ax.set_title(f'{label_name} (AUC={auc_str})', fontsize=11)
plt.suptitle('Confusion matrices: 4 архитектуры на DFD01 (cross-dataset)', fontsize=12, y=1.0)
plt.tight_layout()
plt.savefig(f'{OUT}/confusion_matrices.png', dpi=180, bbox_inches='tight')
plt.close()
print('  ✓ confusion_matrices.png saved')


# =============================================================
# EXP 4: t-SNE — A1 feature space on DFD01
# =============================================================
print('\n' + '='*60)
print('EXP 4: t-SNE on DFD01')
print('='*60)

emb = np.load(f'{OUT}/A1_embeddings_dfd01.npy') if os.path.exists(f'{OUT}/A1_embeddings_dfd01.npy') else None
y_emb = np.load(f'{OUT}/A1_labels_dfd01.npy') if os.path.exists(f'{OUT}/A1_labels_dfd01.npy') else None

if emb is not None:
    print(f'  Embeddings shape: {emb.shape}')
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1500, n_jobs=-1)
    emb_2d = tsne.fit_transform(emb)
    np.save(f'{OUT}/tsne_2d_dfd01.npy', emb_2d)
    
    p_dfd01 = predictions_storage['A1_sanity']['p']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    real_mask = y_emb == 0
    fake_mask = y_emb == 1
    axes[0].scatter(emb_2d[real_mask, 0], emb_2d[real_mask, 1], c='green', label=f'Real (n={real_mask.sum()})', alpha=0.6, s=20)
    axes[0].scatter(emb_2d[fake_mask, 0], emb_2d[fake_mask, 1], c='red', label=f'Fake (n={fake_mask.sum()})', alpha=0.6, s=20)
    axes[0].set_title('(а) По истинному классу')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    scatter = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=p_dfd01, cmap='RdYlGn_r', alpha=0.7, s=20, vmin=0, vmax=1)
    axes[1].set_title('(б) По предсказанной P(fake)')
    plt.colorbar(scatter, ax=axes[1], label='P(fake)')
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f't-SNE feature space модели A1 на DFD01 cross-dataset (n={len(y_emb)})', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/tsne_dfd01.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('  ✓ tsne_dfd01.png saved')


# =============================================================
# EXP 5: t-SNE — A1 3DS model on ALL 3 datasets (show domain separation)
# =============================================================
print('\n' + '='*60)
print('EXP 5: t-SNE on 3 datasets — domain separation visualization')
print('='*60)

all_emb, all_y, all_domain = [], [], []
domain_paths = [
    ('DFDC02', f'{DATA_T16}/preprocessed_DFDC02_16'),
    ('DFD01',  f'{DATA_T16}/preprocessed_DFD01_16'),
    ('CelebDF', f'{DATA_T16}/preprocessed_CelebDF_16'),
]

for domain, path in domain_paths:
    if not os.path.exists(path):
        print(f'  {domain}: path missing')
        continue
    try:
        model, cfg = load_model(CKPTS['A1_full_3DS_T16'], path)
        index = build_video_index(path)
        # Sample max 200 videos per domain for t-SNE speed
        if len(index) > 200:
            np.random.seed(42)
            idx = np.random.choice(len(index), 200, replace=False)
            index = [index[i] for i in idx]
        dataset = DeepfakeVideoDataset(index, cfg, is_train=False)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
        r = infer(model, loader, capture_embeddings=True)
        all_emb.append(r['embeddings'])
        all_y.extend(r['y'].tolist())
        all_domain.extend([domain] * len(r['y']))
        print(f'  {domain}: collected {len(r["y"])} embeddings')
    except Exception as e:
        print(f'  {domain}: error {e}')

if all_emb:
    all_emb = np.concatenate(all_emb, axis=0)
    all_y = np.array(all_y)
    all_domain = np.array(all_domain)
    print(f'  Total: {all_emb.shape}')
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1500, n_jobs=-1)
    emb_2d = tsne.fit_transform(all_emb)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors_dom = {'DFDC02': 'blue', 'DFD01': 'orange', 'CelebDF': 'purple'}
    for d in ['DFDC02', 'DFD01', 'CelebDF']:
        mask = all_domain == d
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=colors_dom[d], label=f'{d} (n={mask.sum()})', alpha=0.6, s=20)
    axes[0].set_title('(а) По датасету — видна доменная кластеризация')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    
    real_mask = all_y == 0
    fake_mask = all_y == 1
    axes[1].scatter(emb_2d[real_mask, 0], emb_2d[real_mask, 1], c='green', label=f'Real (n={real_mask.sum()})', alpha=0.6, s=20)
    axes[1].scatter(emb_2d[fake_mask, 0], emb_2d[fake_mask, 1], c='red', label=f'Fake (n={fake_mask.sum()})', alpha=0.6, s=20)
    axes[1].set_title('(б) По классу — real vs fake')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    
    plt.suptitle('t-SNE A1 dual-path (3DS T=16) — 3 датасета\nЕсли датасеты кластеризуются раздельно → модель учит ДОМЕН а не deepfake-инвариант', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUT}/tsne_3datasets.png', dpi=180, bbox_inches='tight')
    plt.close()
    print('  ✓ tsne_3datasets.png saved')


# =============================================================
# SUMMARY
# =============================================================
print('\n' + '='*60)
print('OUTPUTS in /kaggle/working/:')
print('='*60)
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f'  {f}: {size/1024:.1f} KB')

print('\nDone. Download all files for VKR integration.')
