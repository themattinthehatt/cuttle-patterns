"""Wrong-parameter and no-fog controls for the tone-curve correction (same frames per variant).

'wrong': each session corrected with the next tank's fitted parameters (map, fog, gains).
'nofog': Day1 Tank2/Tank6 corrected without fog subtraction (diagnostic for the a* residual).
All variants use the per-session own-floor target for brightness (`*_own` in the main script).

Example:
    python scratch/photometry/pixel_stats_controls.py scratchpad_dir
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
DIRS = {'orig': None, 'curve': 'beast_frames_curve', 'wrong': 'beast_frames_wrong',
        'nofog': 'beast_frames_nofog'}


def main(out_dir: Path) -> None:
    """Print eta^2 by session/tank for all variants, and a* for the no-fog sessions."""
    base = pd.read_parquet(out_dir / 'stats' / 'pixel_stats_variants.parquet')
    cls = pd.read_parquet(RES / '.iter-1.0' / 'classifications' / 'iter-1.1_classifier_d512.parquet')
    glob = float(np.load(OUT / 'params' / 'Day1_Tank2.npz')['target'][0])
    rng = np.random.default_rng(2)
    rows = []
    for video in base.video.unique():
        sess = video.split('_Cuttle')[0]
        ref = float(np.load(OUT / 'params' / f'{sess}.npz')['floor_ref'])
        have = sorted(int(p.stem[3:]) for p in (OUT / 'beast_frames_wrong' / video).glob('img*.png'))
        conf = cls[(cls.video_name == video) & (cls.confidence > 0.8) & cls.frame_number.isin(have)]
        if len(conf) == 0:
            continue
        for r in conf.iloc[rng.choice(len(conf), min(120, len(conf)), replace=False)].itertuples():
            name = f'img{int(r.frame_number):08d}.png'
            orig = cv2.imread(str(RES / '.iter-1.0' / 'beast_frames' / video / name))
            mask = orig.sum(2) > 24
            if mask.mean() < 0.2:
                continue
            row = {'video': video, 'session': sess, 'tank': int(video.split('Tank')[1][0]),
                   'pattern': r.predicted_pattern}
            for key, d in DIRS.items():
                if d is None:
                    img = orig
                else:
                    img = cv2.imread(str(OUT / d / video / name))
                    if img is None:
                        continue
                    # brightness: own-floor target; wrong run used the borrowed session's target
                    img = to_srgb_u8(to_linear(img) * (ref / glob))
                for stat, val in lab_stats(img, mask).items():
                    row[f'{stat}_{key}'] = val
            rows.append(row)
    d = pd.DataFrame(rows)
    print(f'{len(d)} frames')
    for group in ('session', 'tank'):
        t = pd.DataFrame({k: {s: eta_squared(d, f'{s}_{k}', group) for s in STATS}
                          for k in ('orig', 'curve', 'wrong')})
        print(f'\neta^2 of {group} within pattern')
        print(t.round(3).to_string())
    nf = d.dropna(subset=['a_nofog'])
    print('\nno-fog diagnostic (mean over Leopard frames):')
    print(nf[nf.pattern == 'Leopard'].groupby('session')[
        ['a_orig', 'a_curve', 'a_nofog', 'blk_orig', 'blk_curve', 'blk_nofog',
         'b_curve', 'b_nofog']].mean().round(2).to_string())


if __name__ == '__main__':
    main(Path(sys.argv[1]))
