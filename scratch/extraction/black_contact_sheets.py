"""Contact sheets of rectangle frames binned by fraction of pure-black (background) pixels.

Black fraction = share of pixels whose max channel value is exactly 0 in the aligned rectangle
video (`results/rectangles/*.mp4`); per-video fractions are cached. Population: frames that
pass criteria 1-4 (eligibility parquets from `count_eligible.py`) and are below the 99th
motion-energy percentile of those frames (motion energy from `motion_contact_sheets.py`'s
cache; frame 0 excluded).

Writes one 5x10 sheet per video; rows are black-fraction bins >25%, 20-25%, 15-20%, 10-15%,
5-10%; 10 random frames per row; empty cells are padded with black.

Usage:
    python black_contact_sheets.py --out-dir <dir> --cache-dir <black cache> \
        --me-cache-dir <me cache> --elig-dir <dir> [--shard i --n-shards n]
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.config import load_config
from jump_contact_sheets import POSE_DIR, read_frames

_logger = logging.getLogger(__name__)

# (low, high) black fraction per row, top to bottom; high=inf for the first row
BINS = [(0.25, np.inf), (0.20, 0.25), (0.15, 0.20), (0.10, 0.15), (0.05, 0.10)]
N_COLS = 10
TILE_W, TILE_H = 256, 128
ME_PCT_CUT = 99.0


def black_fraction(video_path: Path, cache_path: Path) -> np.ndarray:
    """Compute (or load cached) per-frame fraction of pure-black pixels.

    Args:
        video_path: aligned rectangle video.
        cache_path: .npy cache location.

    Returns:
        float32 array of shape (n_frames,).
    """
    if cache_path.exists():
        return np.load(cache_path)
    cap = cv2.VideoCapture(str(video_path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(float((frame.max(axis=2) == 0).mean()))
    cap.release()
    arr = np.array(out, dtype=np.float32)
    np.save(cache_path, arr)
    return arr


def main() -> None:
    """Make black-fraction contact sheets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--me-cache-dir', type=Path, required=True)
    parser.add_argument('--elig-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rect_dir = load_config().results_dir / 'rectangles'

    pose_files = sorted(POSE_DIR.glob('*.csv'))
    for i in range(args.shard, len(pose_files), args.n_shards):
        stem = pose_files[i].stem
        rng = np.random.default_rng([args.seed, i])
        video_path = rect_dir / f'{stem}.mp4'
        black = black_fraction(video_path, args.cache_dir / f'{stem}_black.npy')
        me = np.load(args.me_cache_dir / f'{stem}_me.npy')
        elig = pd.read_parquet(args.elig_dir / f'{stem}.parquet')['eligible'].to_numpy().copy()
        elig[0] = False
        idx_elig = np.where(elig)[0]
        # motion-energy percentile among eligible frames; drop the top 1%
        cut = np.percentile(me[idx_elig], ME_PCT_CUT)
        use = elig & (me <= cut)

        picks, counts = [], []
        for lo, hi in BINS:
            cand = np.where(use & (black >= lo) & (black < hi))[0]
            counts.append(len(cand))
            picks.append(np.sort(rng.choice(cand, min(N_COLS, len(cand)), replace=False)))
        frames = read_frames(video_path, [int(j) for p in picks for j in p])
        rows = []
        for pick in picks:
            tiles = []
            for j in pick:
                img = cv2.resize(frames[int(j)], (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
                cv2.putText(img, f'{j} b={100 * black[j]:.0f}%', (4, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(img)
            tiles += [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (N_COLS - len(tiles))
            rows.append(np.hstack(tiles))
        out = args.out_dir / f'{stem}_black.jpg'
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        frac_gt25 = float((use & (black >= 0.25)).sum() / use.sum())
        _logger.info(f'{stem}: {use.sum()} frames; counts per row {counts}; '
                     f'>=25% black: {100 * frac_gt25:.2f}%')


if __name__ == '__main__':
    main()
