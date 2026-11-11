"""Per-session outlier check: scene (full-frame samples, all days) and animal (iter-1.0 frames).

Scene stats come from `raw_data_samples` (linear light, central 25-75% window; the window was
checked by eye only for the 16 iter-1.0 sessions): floor luminance and chromaticity (per-pixel
brightest-60% median over frames), dark level (p0.5 over pixels and frames, a fog proxy), dark/floor
ratio, spatial floor non-uniformity and frame-to-frame brightness variation.

Animal stats come from the original iter-1.0 beast frames (all 32 videos, including the six whose
current crops differ): per-video medians of Lab L, a, b, black level (p2 L) and saturation, within
each of the two common pattern classes.

Example:
    python scratch/photometry/outlier_stats.py out_dir
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2lab

RES = Path('/media/mattw/CUTTLE/results')
SAMPLES = Path('/media/mattw/CUTTLE/raw_data_samples')
STEP = 5
CENTER = (0.25, 0.75)


def to_linear(img_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 to linear float32."""
    x = img_u8.astype(np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def scene_stats(session: str) -> dict:
    """Floor / dark-level statistics of one session's sample frames."""
    files = sorted(f for f in (SAMPLES / session).glob('*.png'))[::STEP]
    frames = []
    for f in files:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        win = img[int(CENTER[0] * h):int(CENTER[1] * h), int(CENTER[0] * w):int(CENTER[1] * w)]
        frames.append(to_linear(win[::2, ::2]))
    arr = np.stack(frames)  # (n, h, w, 3) BGR
    lum = arr.mean(3)
    keep = lum >= np.percentile(lum, 40, axis=0)
    floor_map = np.nanmedian(np.where(keep[..., None], arr, np.nan), 0)
    floor = np.median(floor_map.reshape(-1, 3), 0)
    dark = np.percentile(arr.reshape(-1, 3), 0.5, axis=0)
    frame_lum = np.median(lum.reshape(len(arr), -1), 1)
    fl = floor_map.mean(2)
    return {
        'session': session, 'n': len(arr), 'floor_lum': floor.mean(),
        'floor_B_R': floor[0] / floor[2], 'floor_G_R': floor[1] / floor[2],
        'dark_lum': dark.mean(), 'dark_over_floor': dark.mean() / floor.mean(),
        'floor_nonunif': np.ptp(np.percentile(fl, [5, 95])) / np.median(fl),
        'brightness_cv': frame_lum.std() / frame_lum.mean(),
    }


def animal_stats(out_dir: Path) -> pd.DataFrame:
    """Per-video animal Lab statistics on the original iter-1.0 frames."""
    cls = pd.read_parquet(RES / '.iter-1.0' / 'classifications' / 'iter-1.1_classifier_d512.parquet')
    cls = cls[cls.confidence > 0.8]
    rng = np.random.default_rng(0)
    rows = []
    for video, sub in cls.groupby('video_name'):
        for pattern in ('Leopard', 'Dark aggression'):
            s = sub[sub.predicted_pattern == pattern]
            if len(s) < 20:
                continue
            vals = []
            for r in s.iloc[rng.choice(len(s), min(120, len(s)), replace=False)].itertuples():
                img = cv2.imread(str(
                    RES / '.iter-1.0' / 'beast_frames' / video / f'img{int(r.frame_number):08d}.png'))
                mask = img.sum(2) > 24
                if mask.mean() < 0.2:
                    continue
                lab = rgb2lab(img[..., ::-1] / 255.0)[mask]
                vals.append([lab[:, 0].mean(), lab[:, 1].mean(), lab[:, 2].mean(),
                             np.percentile(lab[:, 0], 2), np.hypot(lab[:, 1], lab[:, 2]).mean()])
            if len(vals) >= 10:
                m = np.median(vals, 0)
                rows.append({'video': video, 'session': video.split('_Cuttle')[0],
                             'pattern': pattern, 'n': len(vals), 'L': m[0], 'a': m[1],
                             'b': m[2], 'blk': m[3], 'sat': m[4]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'animal_outliers.csv', index=False)
    return df


def main(out_dir: Path) -> None:
    """Compute and save both tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(p.name for p in SAMPLES.iterdir() if p.is_dir())
    with Pool(12) as pool:
        scene = pd.DataFrame(pool.map(scene_stats, sessions))
    scene.to_csv(out_dir / 'scene_outliers.csv', index=False)
    animal_stats(out_dir)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
