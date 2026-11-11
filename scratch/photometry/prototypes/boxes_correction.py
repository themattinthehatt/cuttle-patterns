"""Prototype v3: boxes-driven D (drain), floor (tank box), flat field with interior mask."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
from skimage.color import rgb2lab

S = Path(sys.argv[1])
RAW = Path('/media/mattw/CUTTLE/raw_data_samples')
BF = Path('/media/mattw/CUTTLE/results/.iter-1.0/beast_frames')
BOX = json.load(open('/home/mattw/code/cuttle-patterns/scratch/photometry/boxes.json'))
CL = pd.read_parquet('/media/mattw/CUTTLE/results/.iter-1.0/classifications/iter-1.1_classifier_d512.parquet')

def to_lin(u8):
    x = u8.astype(np.float32) / 255
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
def to_srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= .0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - .055)
def enc(l): return (to_srgb(l) * 255 + .5).astype(np.uint8)
def sl(b): return (slice(b[1], b[3]), slice(b[0], b[2]))

T = {}   # per-tank data
for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
    n = int(d.name.split('Tank')[1]); reg = BOX[f'Day1_Tank{n}']['regions']
    fs = sorted((d / 'Downsampled_frames').glob('*.jpg'))
    A = np.stack([to_lin(np.asarray(Image.open(f).convert('RGB'))) for f in fs])
    assert A.shape[2] == BOX[f'Day1_Tank{n}']['frame_size'][0]
    idx = np.array([int(f.stem) - 1 for f in fs])
    dr = reg['black_drain']; x0, y0, x1, y1 = dr
    sur = A[:, max(y0-15, 0):y1+15, max(x0-15, 0):x1+15].mean(3).reshape(len(fs), -1).std(1)
    ok = sur > 0.02                                   # person/blank covering the drain -> drop
    D_t = np.median(A[(slice(None), *sl(dr))].reshape(len(fs), -1, 3), 1)            # per-frame drain, linear
    bg = np.median(A, 0); tk = reg['tank']
    T[n] = dict(A=A, idx=idx, ok=ok, D_t=D_t, bg=bg, tank=tk, reg=reg, files=fs)
    T[n]['D_s'] = np.median(D_t[ok], 0)
    T[n]['D_edge'] = np.percentile(bg.reshape(-1, 3), .5, axis=0)
    print(f'tank {n}: {int((~ok).sum())} frames dropped as occluded: {idx[~ok].tolist()}')

# 1. D drift table + time series -------------------------------------------------------------
rows = []
for n, t in T.items():
    m = t['D_t'].mean(1); ok = t['ok']; h = len(m) // 2
    rows.append(dict(tank=n, D_session=t['D_s'].round(4).tolist(), D_edge=t['D_edge'].round(4).tolist(),
                     D_p5=np.percentile(m[ok], 5).round(4), D_p95=np.percentile(m[ok], 95).round(4),
                     first_half=m[:h][ok[:h]].mean().round(4), second_half=m[h:][ok[h:]].mean().round(4)))
print('\ndrain-derived D (linear):'); print(pd.DataFrame(rows).set_index('tank').to_string())

def floor_series(t, D):
    """per-frame floor median over fish-free tank-box pixels, after subtracting D (list per frame)."""
    out = []
    bgl = t['bg'].mean(2)[sl(t['tank'])]
    for k in range(len(t['A'])):
        x = t['A'][k][sl(t['tank'])]; m = np.abs(x.mean(2) / np.maximum(bgl, 1e-3) - 1) < .1
        out.append(np.median((x[m] - D[k]), 0))
    return np.array(out)
fig, ax = plt.subplots(1, 3, figsize=(18, 4.2)); cmap = plt.cm.tab10
stab = []
for n, t in T.items():
    ok = t['ok']; x = t['idx'][ok]
    raw_fl = floor_series(t, np.zeros_like(t['D_t'])); t['fl_raw'] = raw_fl
    ax[0].plot(x, t['D_t'].mean(1)[ok], c=cmap(n), label=f'tank {n}')
    ax[1].plot(x, raw_fl.mean(1)[ok], c=cmap(n)); ax[2].plot(x, (raw_fl[:, 2] / raw_fl[:, 0])[ok], c=cmap(n))
    r = np.corrcoef(t['D_t'].mean(1)[ok], raw_fl.mean(1)[ok])[0, 1]
    stab.append(dict(tank=n, floor_lum_mean=raw_fl.mean(1)[ok].mean().round(3), floor_lum_sd=raw_fl.mean(1)[ok].std().round(4),
                     floor_BR_sd=(raw_fl[:, 2] / raw_fl[:, 0])[ok].std().round(4), corr_D_floor=round(r, 2)))
ax[0].set_title('drain D (linear)'); ax[1].set_title('floor luminance (linear, raw)'); ax[2].set_title('floor B/R (raw)')
for a in ax: a.set_xlabel('video frame')
ax[0].legend(ncol=2, fontsize=8); plt.tight_layout(); plt.savefig(S / 'timeseries_v3.png', dpi=70); plt.close()
print('\nfloor stability over time (raw, fish-free tank-box pixels):'); print(pd.DataFrame(stab).set_index('tank').to_string())

# 2. correction modes -------------------------------------------------------------------------
def fit(Dsrc):
    fl = {n: np.maximum(np.median(T[n]['bg'][sl(T[n]['tank'])].reshape(-1, 3), 0) - Dsrc(n), 1e-3) for n in T}
    ref = np.median([f.mean() for f in fl.values()])
    return {n: ref / fl[n] for n in fl}      # chroma-neutral + exposure to common reference
G = {'A2': fit(lambda n: np.zeros(3)), 'A3d': fit(lambda n: T[n]['D_s']), 'A3e': fit(lambda n: T[n]['D_edge'])}
REF = {m: np.median([np.median(T[n]['bg'][sl(T[n]['tank'])].reshape(-1, 3), 0).mean() - (0 if m == 'A2' else (T[n]['D_s'] if m == 'A3d' else T[n]['D_edge']).mean()) for n in T]) for m in G}

def correct_lin(x, n, mode, k=None, bgs=None):
    """x: linear HxWx3; k: frame k (per-frame D) for the *t modes; bgs: bg override for held-out flat field."""
    t = T[n]; bg = t['bg'] if bgs is None else bgs
    if mode == 'raw': return x
    if mode in G:
        D = {'A2': 0, 'A3d': t['D_s'], 'A3e': t['D_edge']}[mode]
        return np.maximum(x - D, 0) * G[mode][n]
    if mode in ('B2s', 'B2t'):
        D = t['D_s'] if mode == 'B2s' else t['D_t'][k]
        den = np.maximum(bg - t['D_s'], 5e-3)
        r = np.maximum(x - D, 0) / den
        if mode == 'B2t':    # per-frame scalar gain: fish-free tank-box median of the ratio image
            sub = r[sl(t['tank'])]; m = np.abs(sub.mean(2) - 1) < .1
            r = r / np.median(sub[m], 0)
        return r * REF['A3d'] * 1.15
    raise ValueError(mode)

# 3. held-out flatness: fit bg on even frames, evaluate on odd frames (fish-free tank-box pixels)
flat = []
for n, t in T.items():
    bg_even = np.median(t['A'][::2], 0); ev = []
    for k in range(1, len(t['A']), 2):
        if not t['ok'][k]: continue
        # flat-field fits use the even-frame bg for the held-out test
        row = {'tank': n}
        for mode in ['raw', 'A3d', 'B2s', 'B2t']:
            y = correct_lin(t['A'][k], n, mode, k=k, bgs=bg_even)[sl(t['tank'])]
            ref = np.median(t['A'][::2][(slice(None), *sl(t['tank']))].mean(3), 0)       # fish-free mask from raw bg
            m = np.abs(t['A'][k][sl(t['tank'])].mean(2) / np.maximum(ref, 1e-3) - 1) < .1
            lum = y.mean(2)[m]; br = (y[..., 2] / np.maximum(y[..., 0], 1e-3))[m]
            row[f'{mode}_lum_range'] = np.ptp(np.percentile(lum, [5, 95])) / np.median(lum)
            row[f'{mode}_BR_range'] = np.ptp(np.percentile(br, [5, 95]))
        ev.append(row)
    flat.append(pd.DataFrame(ev).groupby('tank').mean())
flat = pd.concat(flat)
print('\nheld-out spatial spread of the floor within the tank box (p5-p95; lum relative to median) - lower is flatter')
print(flat[[c for c in flat.columns if 'lum' in c]].round(3).to_string())
print(flat[[c for c in flat.columns if 'BR' in c]].round(3).to_string())

# 4. frames figure (mid frame): raw / A3 (drain D) / B2 masked to tank box; early+late for tanks 4, 6
fig, ax = plt.subplots(5, 6, figsize=(24, 13))
for j, n in enumerate(sorted(T)):
    t = T[n]; k = 40; x = t['A'][k]; tk = t['tank']
    for i, mode in enumerate(['raw', 'A3d', 'B2s']):
        y = correct_lin(x, n, mode, k=k)
        if mode == 'B2s':
            m = np.zeros(y.shape[:2], bool); m[sl(tk)] = True; y = np.where(m[..., None], y, 0)
        ax[i, j].imshow(enc(y)); ax[i, j].axis('off')
    kk = [3, 90]
    for i, k2 in enumerate(kk):
        y = correct_lin(t['A'][k2], n, 'B2t', k=k2); m = np.zeros(y.shape[:2], bool); m[sl(tk)] = True
        ax[3 + i, j].imshow(enc(np.where(m[..., None], y, 0))); ax[3 + i, j].axis('off')
        ax[3 + i, j].set_title(f'B2t frame {t["idx"][k2]}' + ('' if t['ok'][k2] else ' (occluded D)'), fontsize=8)
    ax[0, j].set_title(f'Tank {n}')
for i, l in enumerate(['raw', 'A3: drain D + box-floor gain', 'B2: drain D + flat field (masked)', 'B2t early (per-frame D, gain)', 'B2t late']):
    ax[i, 0].text(-.02, .5, l, transform=ax[i, 0].transAxes, rotation=90, va='center', ha='right', fontsize=10)
plt.tight_layout(); plt.savefig(S / 'frames_v3.jpg', dpi=60); plt.close()

# 5. eta^2 on crops with session-constant corrections (per-frame D needs the crop<->frame map)
CLs = CL[CL.video_name.str.startswith('Day1_Tank') & (CL.confidence > .8)]
CLs = CLs.assign(tank=CLs.video_name.str.extract(r'Tank(\d)')[0].astype(int))
avail = [n for n in sorted(CLs.tank.unique()) if n in T]
def load(r): return np.asarray(Image.open(BF / r.video_name / f'img{int(r.frame_number):08d}.png').convert('RGB'))
rows = []
for r in CLs[CLs.tank.isin(avail)].groupby('video_name').sample(150, random_state=0, replace=True).drop_duplicates(['video_name', 'frame_number']).itertuples():
    u8 = load(r); fish = u8.sum(2) > 24
    if fish.mean() < .2: continue
    d = {'tank': r.tank, 'pattern': r.predicted_pattern}
    for m in ['raw', 'A2', 'A3e', 'A3d']:
        y = enc(correct_lin(to_lin(u8), r.tank, m)) if m != 'raw' else u8
        lab = rgb2lab(y / 255.)[fish]
        d[f'L_{m}'], d[f'a_{m}'], d[f'b_{m}'] = lab.mean(0)
        d[f'blk_{m}'] = np.percentile(lab[:, 0], 2); d[f'sat_{m}'] = np.hypot(lab[:, 1], lab[:, 2]).mean()
    rows.append(d)
df = pd.DataFrame(rows)
def eta(col):
    ssb = sst = 0
    for _, g in df.groupby('pattern'):
        if g.tank.nunique() < 2: continue
        mu = g[col].mean(); sst += ((g[col] - mu) ** 2).sum()
        ssb += g.groupby('tank')[col].apply(lambda x: len(x) * (x.mean() - mu) ** 2).sum()
    return ssb / sst
M = ['raw', 'A2', 'A3e', 'A3d']
print(f'\n{len(df)} crops; tank eta^2 within pattern (session-constant corrections)')
print(pd.DataFrame({m: {s: eta(f'{s}_{m}') for s in ['L', 'a', 'b', 'blk', 'sat']} for m in M}).round(3).to_string())
