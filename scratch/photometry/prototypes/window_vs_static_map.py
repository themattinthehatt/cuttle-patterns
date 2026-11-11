"""Does the lighting *shape* drift in time? Windowed fish-masked maps vs the static map."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image

S = Path(sys.argv[1]); RAW = Path('/media/mattw/CUTTLE/raw_data_samples')
BOX = json.load(open('/home/mattw/code/cuttle-patterns/scratch/photometry/boxes.json'))
W, STEP, POOL, FISH_THR, MIN_VALID = 15, 5, 4, 0.85, 5
rng = np.random.default_rng(0)

def to_lin(u8):
    x = u8.astype(np.float32) / 255
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
def pool(a):
    h, w = (a.shape[0] // POOL) * POOL, (a.shape[1] // POOL) * POOL
    return a[:h, :w].reshape(h // POOL, POOL, w // POOL, POOL, 3).mean((1, 3))
def spread(r):
    """p5-p95 range of lum (relative) and of B/R over valid pixels of a ratio image."""
    v = r[np.isfinite(r).all(-1)]
    lum = v.mean(1); br = v[:, 2] / v[:, 0]
    return np.ptp(np.percentile(lum, [5, 95])) / np.median(lum), np.ptp(np.percentile(br, [5, 95]))
def masked_median(A, M, idx):
    """per-pixel median over frames idx, ignoring fish pixels; NaN where < MIN_VALID samples."""
    x = np.where(M[idx][..., None], np.nan, A[idx])
    out = np.nanmedian(x, 0); out[(~M[idx]).sum(0) < MIN_VALID] = np.nan
    return out
def norm(r):  # remove the overall per-channel scale (what the dial would do)
    return r / np.nanmedian(r.reshape(-1, 3), 0)

rows, ctrl_rows, loo_rows, maps = [], [], [], {}
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    n = int(d.name.split('Tank')[1]); b = BOX[f'Day1_Tank{n}']['regions']['tank']
    fs = sorted((d / 'Downsampled_frames').glob('*.jpg'))
    A = np.stack([pool(to_lin(np.asarray(Image.open(f).convert('RGB')))[b[1]:b[3], b[0]:b[2]]) for f in fs])
    K = len(A); bg = np.median(A, 0)
    M = A.mean(3) < FISH_THR * bg.mean(2)[None]       # fish (or shadow) pixels
    # (a) contiguous windows vs random frames: shape residual after removing the scalar
    for s in range(0, K - W + 1, STEP):
        idx = np.arange(s, s + W)
        wm = masked_median(A, M, idx); r = norm(wm / bg)
        rows.append(dict(tank=n, start=s, holes=np.isnan(wm).any(-1).mean(),
                         # fish baked into an *unmasked* window median
                         baked=(np.median(A[idx], 0).mean(2) < FISH_THR * bg.mean(2)).mean(),
                         **dict(zip(['lum', 'BR'], spread(r)))))
        if s in (0, (K - W) // 2 // STEP * STEP, (K - W) // STEP * STEP): maps[(n, s)] = r.mean(2)
        ridx = np.sort(rng.choice(K, W, replace=False))
        rr = norm(masked_median(A, M, ridx) / bg)
        ctrl_rows.append(dict(tank=n, **dict(zip(['lum', 'BR'], spread(rr)))))
    # (b) head to head on held-out frames: static map vs leave-one-out window map, scalar removed
    for k in range(2, K - 2, 3):
        lo = min(max(k - W // 2, 0), K - W); idx = np.array([i for i in range(lo, lo + W) if i != k])
        stat = np.median(np.delete(A, k, 0), 0)
        win = masked_median(A, M, idx)
        win = np.where(np.isnan(win), stat, win)       # fall back to the static map in holes
        ok = ~M[k]
        for name, mp in (('static', stat), ('window', win)):
            r = norm(A[k] / mp); r = np.where(ok[..., None], r, np.nan)
            loo_rows.append(dict(tank=n, map=name, **dict(zip(['lum', 'BR'], spread(r)))))
    print('done tank', n, flush=True)

R, C, L = pd.DataFrame(rows), pd.DataFrame(ctrl_rows), pd.DataFrame(loo_rows)
t = R.groupby('tank').agg(win_lum=('lum', 'mean'), win_lum_max=('lum', 'max'), win_BR=('BR', 'mean'), win_BR_max=('BR', 'max'),
                          holes_mean=('holes', 'mean'), holes_max=('holes', 'max'), baked_mean=('baked', 'mean'), baked_max=('baked', 'max'))
t = t.join(C.groupby('tank').agg(rand_lum=('lum', 'mean'), rand_BR=('BR', 'mean')))
print('\n(a) shape residual of a 15-sample map vs the static map (scalar removed); contiguous vs random frames')
print(t.round(3).to_string())
print('\n(b) held-out floor flatness of one frame, dividing by static map vs by 14-neighbour window map (lower = flatter)')
print(L.groupby(['tank', 'map'])[['lum', 'BR']].mean().unstack().round(3).to_string())

fig, ax = plt.subplots(3, 6, figsize=(24, 9))
for j, n in enumerate(sorted(R.tank.unique())):
    for i, key in enumerate([k for k in maps if k[0] == n]):
        im = ax[i, j].imshow(maps[key], vmin=.9, vmax=1.1, cmap='RdBu_r'); ax[i, j].axis('off')
        ax[i, j].set_title(f'tank {n}, samples {key[1]}-{key[1]+W-1}', fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6, label='window map / static map (scalar removed)')
plt.savefig(S / 'window_maps.png', dpi=60); plt.close()
