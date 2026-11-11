"""Step 3 gate: do pixel color statistics of the animal depend less on tank/session after correction?

For the iter-1.0 videos whose current crop video reproduces the iter-1.0 frames (re-export MAD
< 10), samples classifier-confident frames and computes animal-pixel Lab statistics for the
original frame, the raw re-export control (same warp, no correction) and the corrected frame.
Reports eta^2 of tank (and of session) within pattern class, the README's metric (lower = less
nuisance structure), plus a null control: shuffled tank labels within class.

Example:
    python scratch/photometry/pixel_stats.py out_dir
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab

RES = Path('/media/mattw/CUTTLE/results')
OUT = RES / '.iter-1.0-photometry'
N_PER_VIDEO = 120
CONF = 0.8
MAD_OK = 10.0


def lab_stats(img_bgr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Animal-pixel Lab means, 2nd-percentile L (black level) and mean chroma."""
    lab = rgb2lab(img_bgr[..., ::-1] / 255.0)[mask]
    return {
        'L': lab[:, 0].mean(), 'a': lab[:, 1].mean(), 'b': lab[:, 2].mean(),
        'blk': np.percentile(lab[:, 0], 2), 'sat': np.hypot(lab[:, 1], lab[:, 2]).mean(),
    }


def eta_squared(df: pd.DataFrame, col: str, group: str) -> float:
    """Between-`group` variance of `col` as a fraction of total, within pattern class."""
    ssb = sst = 0.0
    for _, sub in df.groupby('pattern'):
        if sub[group].nunique() < 2:
            continue
        mu = sub[col].mean()
        sst += ((sub[col] - mu) ** 2).sum()
        ssb += sub.groupby(group)[col].apply(lambda x: len(x) * (x.mean() - mu) ** 2).sum()
    return ssb / sst


def main(out_dir: Path) -> None:
    """Compute and print the eta^2 table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cls = pd.read_parquet(RES / '.iter-1.0' / 'classifications' / 'iter-1.1_classifier_d512.parquet')
    cls = cls[cls.confidence > CONF]
    good = []
    for f in sorted(out_dir.parent.glob('apply/apply_Day*.csv')):
        df = pd.read_csv(f)
        stat = df.groupby('fish_id').mad_orig_vs_rawreexport.mean()
        for fish_id, mad in stat.items():
            if mad < MAD_OK:
                good.append(f'{f.stem.removeprefix("apply_")}_{fish_id}_Crop')
    print(f'{len(good)} clean videos')
    rows = []
    rng = np.random.default_rng(0)
    for video in good:
        sub = cls[cls.video_name == video]
        have = {int(p.stem[3:]) for p in (OUT / 'beast_frames' / video).glob('img*.png')}
        sub = sub[sub.frame_number.isin(have)]
        if len(sub) == 0:
            continue
        pick = sub.iloc[rng.choice(len(sub), min(N_PER_VIDEO, len(sub)), replace=False)]
        for r in pick.itertuples():
            name = f'img{int(r.frame_number):08d}.png'
            imgs = {
                'orig': cv2.imread(str(RES / '.iter-1.0' / 'beast_frames' / video / name)),
                'raw': cv2.imread(str(OUT / 'beast_frames_rawreexport' / video / name)),
                'corr': cv2.imread(str(OUT / 'beast_frames' / video / name)),
            }
            mask = imgs['orig'].sum(2) > 24
            if mask.mean() < 0.2:
                continue
            row = {
                'video': video, 'session': video.split('_Cuttle')[0],
                'tank': int(video.split('Tank')[1][0]), 'pattern': r.predicted_pattern,
            }
            for k, im in imgs.items():
                for stat, val in lab_stats(im, mask).items():
                    row[f'{stat}_{k}'] = val
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / 'pixel_stats.parquet')
    print(f'{len(df)} frames; patterns: {df.pattern.value_counts().to_dict()}')
    shuf = df.copy()
    shuf['tank_null'] = shuf.groupby('pattern').tank.transform(lambda x: rng.permutation(x.values))
    for group in ('tank', 'session'):
        table = pd.DataFrame({
            k: {s: eta_squared(df, f'{s}_{k}', group) for s in ['L', 'a', 'b', 'blk', 'sat']}
            for k in ('orig', 'raw', 'corr')
        })
        print(f'\neta^2 of {group} within pattern (lower = less nuisance structure)')
        print(table.round(3).to_string())
    null = pd.Series({s: eta_squared(shuf, f'{s}_corr', 'tank_null') for s in ['L', 'a', 'b', 'blk', 'sat']})
    print('\nnull (shuffled tank labels within pattern), corrected stats:')
    print(null.round(3).to_string())


if __name__ == '__main__':
    main(Path(sys.argv[1]))
