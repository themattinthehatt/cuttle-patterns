"""Fish-free static map: compare median (M0), brightness-selected (M1), iterative masked (M2), window (W)."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image

S = Path(sys.argv[1]); RAW = Path('/media/mattw/CUTTLE/raw_data_samples')
BOX = json.load(open('/home/mattw/code/cuttle-patterns/scratch/photometry/boxes.json'))
POOL, W, MIN_VALID, FISH = 4, 15, 5, 0.9
def to_lin(u8):
    x = u8.astype(np.float32) / 255
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
def pool(a):
    h, w = (a.shape[0] // POOL) * POOL, (a.shape[1] // POOL) * POOL
    return a[:h, :w].reshape(h // POOL, POOL, w // POOL, POOL, 3).mean((1, 3))
def scalars(A, M):
    """per-frame per-channel scale: median of frame/map over pixels that look like floor."""
    r = A / np.maximum(M, 1e-4); s = np.median(r.reshape(len(A), -1, 3), 1)
    for _ in range(2):
        lr = r.mean(3) / s.mean(1)[:, None, None]; ok = np.abs(lr - 1) < .15
        s = np.stack([np.array([np.median(r[k][ok[k]][:, c]) if ok[k].sum() > 50 else s[k, c] for c in range(3)]) for k in range(len(A))])
    return s
def build(A):
    lum = A.mean(3); keep = lum >= np.percentile(lum, 40, axis=0)
    M1 = np.nanmedian(np.where(keep[..., None], A, np.nan), 0)
    M = M1
    for _ in range(2):
        s = scalars(A, M); fish = A.mean(3) < FISH * (s[:, None, None, :] * M).mean(3)
        B = np.where(fish[..., None], np.nan, A / s[:, None, None, :])
        M2 = np.nanmedian(B, 0); M2 = np.where((~fish).sum(0)[..., None] < MIN_VALID, M, M2); M = M2
    return M1, M2, s
def spread(r, m):
    v = r[m]; lum = v.mean(1); br = v[:, 2] / v[:, 0]
    return np.ptp(np.percentile(lum, [5, 95])) / np.median(lum), np.ptp(np.percentile(br, [5, 95]))

rows, contam, ex = [], [], {}
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    n = int(d.name.split('Tank')[1]); b = BOX[f'Day1_Tank{n}']['regions']['tank']
    fs = sorted((d / 'Downsampled_frames').glob('*.jpg'))
    A = np.stack([pool(to_lin(np.asarray(Image.open(f).convert('RGB')))[b[1]:b[3], b[0]:b[2]]) for f in fs]); K = len(A)
    best = (0, None)
    for k in range(2, K - 2, 3):
        tr = np.delete(np.arange(K), k); M0 = np.median(A[tr], 0); M1, M2, s_tr = build(A[tr])
        # window map from the 14 temporal neighbours, scalar-normalised and fish-masked via M2
        lo = min(max(k - W // 2, 0), K - W); idx = np.array([i for i in range(lo, lo + W) if i != k])
        s_w = scalars(A[idx], M2); fw = A[idx].mean(3) < FISH * (s_w[:, None, None, :] * M2).mean(3)
        Bw = np.where(fw[..., None], np.nan, A[idx] / s_w[:, None, None, :]); Wm = np.nanmedian(Bw, 0)
        Wm = np.where(((~fw).sum(0)[..., None] < MIN_VALID) | np.isnan(Wm), M2, Wm)
        s_k = scalars(A[k:k+1], M1)[0]; lr = A[k].mean(2) / (s_k.mean() * M1.mean(2)); F = np.abs(lr - 1) < .1
        if F.sum() < 200: continue
        for name, Mp in (('M0 median', M0), ('M1 bright-select', M1), ('M2 masked iter', M2), ('W window', Wm)):
            r = A[k] / np.maximum(Mp, 1e-4); r = r / np.median(r[F], 0)
            rows.append(dict(tank=n, map=name, **dict(zip(['lum', 'BR'], spread(r, F)))))
        C = M0.mean(2) < 0.92 * M1.mean(2)                       # where the median map is fish-darkened
        fish_on_C = (A[k].mean(2) < 0.7 * s_k.mean() * M1.mean(2)) & C
        if fish_on_C.sum() > 20:
            contam.append(dict(tank=n, area=C.mean(), lift=(M1.mean(2) / M0.mean(2))[fish_on_C].mean() - 1))
        if fish_on_C.sum() > best[0]: best = (fish_on_C.sum(), (A[k], M0, M2, s_k))
    ex[n] = best[1]; print('done', n, flush=True)

R = pd.DataFrame(rows)
print('\nheld-out floor flatness (p5-p95; lower is flatter); luminance relative, then B/R')
g = R.groupby(['tank', 'map'])[['lum', 'BR']].mean().unstack('map')
print(g['lum'].round(3).to_string()); print(); print(g['BR'].round(3).to_string())
c = pd.DataFrame(contam)
print('\nfish-darkened area of the median map (share of tank, mean over eval frames) and, when a fish sits on it,')
print('how much the median map lightens that fish vs the clean map (mean fractional lift):')
print(c.groupby('tank').agg(area=('area', 'mean'), n_frames_fish_on_area=('lift', 'size'), lift=('lift', 'mean')).round(3).to_string())

def enc(l):
    l = np.clip(l, 0, 1); return np.where(l <= .0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - .055)
show = [n for n in (2, 4, 6) if ex.get(n) is not None]
fig, ax = plt.subplots(len(show), 3, figsize=(12, 3.2 * len(show)))
for i, n in enumerate(show):
    a, M0, M2, s_k = ex[n]
    for j, (t, im) in enumerate((('raw (pooled)', a), ('÷ median map (M0)', a / M0 * .6), ('÷ clean map (M2)', a / M2 * .6))):
        ax[i, j].imshow(enc(im)); ax[i, j].axis('off'); ax[i, j].set_title(f'tank {n}: {t}', fontsize=9)
plt.tight_layout(); plt.savefig(S / 'clean_map.png', dpi=70)
