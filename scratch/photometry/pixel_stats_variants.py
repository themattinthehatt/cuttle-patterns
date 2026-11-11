"""Compare nuisance structure (eta^2) of animal color statistics across correction variants.

Variants on the same sampled frames as `pixel_stats.py`: original, gain-only correction
('corr'), tone-curve correction ('curve'), each also with a per-session own-floor target
('*_own', brightness not equalized across sessions), and a per-tank mean/std color-normalization
baseline ('norm', applied per Lab channel to the original frames). Also reports an a* ceiling
estimate: eta^2 of fish identity within session (biological/behavioral, not lighting).

Example:
    python scratch/photometry/pixel_stats_variants.py out_dir
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pixel_stats import OUT, RES, eta_squared, lab_stats  # noqa: E402
from prototypes.apply_session import to_linear, to_srgb_u8  # noqa: E402

STATS = ['L', 'a', 'b', 'blk', 'sat']


def main(out_dir: Path) -> None:
    """Recompute stats for each variant on the frames in pixel_stats.parquet."""
    df = pd.read_parquet(out_dir / 'stats' / 'pixel_stats.parquet')
    cls = pd.read_parquet(RES / '.iter-1.0' / 'classifications' / 'iter-1.1_classifier_d512.parquet')
    params = {
        s: float(np.load(OUT / 'params' / f'{s}.npz')['floor_ref']) for s in df.session.unique()
    }
    glob = float(np.load(OUT / 'params' / f'{df.session.iloc[0]}.npz')['target'][0])
    print(f'global target {glob:.3f}')
    # re-pick the same frames: pixel_stats sampled with rng(0); recover from files instead
    rows = []
    for video, sub in df.groupby('video'):
        have = sorted(int(p.stem[3:]) for p in (OUT / 'beast_frames_curve' / video).glob('img*.png'))
        rows.append((video, have))
    # use all frames present in both dirs, subsample per video deterministically
    rng = np.random.default_rng(1)
    out = []
    for video, have in rows:
        sess = video.split('_Cuttle')[0]
        conf = cls[(cls.video_name == video) & (cls.confidence > 0.8)]
        conf = conf[conf.frame_number.isin(have)]
        if len(conf) == 0:
            continue
        pick = conf.iloc[rng.choice(len(conf), min(120, len(conf)), replace=False)]
        for r in pick.itertuples():
            name = f'img{int(r.frame_number):08d}.png'
            orig = cv2.imread(str(RES / '.iter-1.0' / 'beast_frames' / video / name))
            mask = orig.sum(2) > 24
            if mask.mean() < 0.2:
                continue
            row = {
                'video': video, 'session': sess, 'tank': int(video.split('Tank')[1][0]),
                'fish': video.split('_')[2], 'pattern': r.predicted_pattern,
            }
            imgs = {'orig': orig}
            for key, d in (('corr', 'beast_frames'), ('curve', 'beast_frames_curve')):
                imgs[key] = cv2.imread(str(OUT / d / video / name))
                lin = to_linear(imgs[key]) * (params[sess] / glob)
                imgs[f'{key}_own'] = to_srgb_u8(lin)
            for k, im in imgs.items():
                for stat, val in lab_stats(im, mask).items():
                    row[f'{stat}_{k}'] = val
            out.append(row)
    d = pd.DataFrame(out)
    # per-tank mean/std normalization of the original stats (tank = the simple baseline)
    for s in STATS:
        grp = d.groupby(['tank', 'pattern'])[f'{s}_orig']
        d[f'{s}_norm'] = (d[f'{s}_orig'] - grp.transform('mean')) / grp.transform('std')
    d.to_parquet(out_dir / 'stats' / 'pixel_stats_variants.parquet')
    print(f'{len(d)} frames')
    keys = ['orig', 'corr', 'corr_own', 'curve', 'curve_own']
    for group in ('session', 'tank'):
        ks = keys + ['norm'] if group == 'session' else keys
        t = pd.DataFrame({k: {s: eta_squared(d, f'{s}_{k}', group) for s in STATS} for k in ks})
        print(f'\neta^2 of {group} within pattern')
        print(t.round(3).to_string())
    # ceiling: variation explained by fish identity (video) vs by session
    for stat in ('a', 'L'):
        ev = eta_squared(d, f'{stat}_orig', 'video')
        es = eta_squared(d, f'{stat}_orig', 'session')
        print(f'{stat}: eta^2 video {ev:.3f}, session {es:.3f}')
    # fish-pair difference within session (leopard only): lighting is shared by the pair
    leo = d[d.pattern == 'Leopard']
    m = leo.groupby(['session', 'fish'])[['a_orig', 'a_curve', 'L_orig', 'L_curve']].mean()
    print('\nwithin-session difference between the two fish (Leopard, mean |Cuttle1-Cuttle2|):')
    diff = m.groupby('session').agg(lambda x: abs(x.iloc[0] - x.iloc[1]) if len(x) == 2 else np.nan)
    print(diff.mean().round(2).to_string())
    print(f'between-session std of session means: {leo.groupby("session")[["a_orig","a_curve","L_orig","L_curve"]].mean().std().round(2).to_dict()}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
