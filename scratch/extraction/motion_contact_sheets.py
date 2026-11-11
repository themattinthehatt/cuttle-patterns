"""Contact sheets of frames binned by pixel motion-energy percentile.

For each of N random videos (same selection as `jump_contact_sheets.py` for a given seed),
computes BEAST's motion energy (on the aligned rectangle videos, `results/rectangles`) (summed absolute difference of consecutive 32x32 frames, a
backward difference) over the whole video, then writes a 6x10 grid: rows are percentile bins
(99.75-100, 99.5-99.75, 99-99.5, 98-99, 95-98, 90-95) and columns are 10 random frames per bin.

Percentiles are taken over frames that pass the other filters (not blank, all keypoints
>= 0.9 likelihood, keypoint jump <= 25 px), since that is the population x will be applied
to. The rectangle-size filter is not applied here. Motion energy is cached per video.

Usage:
    python motion_contact_sheets.py --n-videos 5 --seed 0 --out-dir <dir> --cache-dir <dir>
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
from beast.video import compute_video_motion_energy

from cuttle_patterns.config import load_config
from cuttle_patterns.preprocessing.pose import load_pose_predictions
from jump_contact_sheets import (
    N_COLS,
    POSE_DIR,
    THUMB,
    compute_jumps,
    read_frames,
)

_logger = logging.getLogger(__name__)

JUMP_MAX = 25.0
RESIZE_DIMS = 32  # matches extract.DEFAULT_RESIZE_DIMS
# (low, high) percentile per row, top to bottom
BINS = [(99.75, 100.0), (99.5, 99.75), (99.0, 99.5), (98.0, 99.0), (95.0, 98.0), (90.0, 95.0)]


def get_motion_energy(video_path: Path, cache_path: Path) -> np.ndarray:
    """Compute (or load cached) motion energy for a video.

    Args:
        video_path: crop video.
        cache_path: .npy cache location.

    Returns:
        motion energy of shape (n_frames,); frame 0 is meaningless (diff against zeros).
    """
    if cache_path.exists():
        return np.load(cache_path)
    me = compute_video_motion_energy(video_path, resize_dims=RESIZE_DIMS)
    np.save(cache_path, me)
    return me


def main() -> None:
    """Make motion-energy contact sheets for random videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-videos', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--all', action='store_true', help='use every video')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = load_config().data_dir
    rng = np.random.default_rng(args.seed)
    pose_files = sorted(POSE_DIR.glob('*.csv'))
    chosen = (np.arange(len(pose_files)) if args.all
              else rng.choice(len(pose_files), args.n_videos, replace=False))
    chosen = chosen[args.shard::args.n_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for i in chosen:
        rng = np.random.default_rng([args.seed, int(i)])  # per-video, order-independent
        pose_path = pose_files[i]
        stem = pose_path.stem
        video_path = data_dir.parent / 'results' / 'rectangles' / f'{stem}.mp4'  # aligned 200x100
        pose_df = load_pose_predictions(pose_path)
        me = get_motion_energy(video_path, args.cache_dir / f'{stem}_me.npy')

        jump, ok = compute_jumps(pose_df)
        blank_path = data_dir / f'{stem.removesuffix("_Crop")}_black_frames.txt'
        is_blank = np.zeros(len(pose_df), dtype=bool)
        if blank_path.exists():
            idx_blank = np.array([int(v) for v in blank_path.read_text().split()], dtype=int)
            is_blank[idx_blank[idx_blank < len(pose_df)]] = True
        use = ok & ~is_blank & ~(jump > JUMP_MAX)
        use[0] = False  # frame 0's backward difference is meaningless
        idx_use = np.where(use)[0]
        pct = np.full(len(me), np.nan)
        # percentile rank among eligible frames
        order = np.argsort(me[idx_use])
        pct[idx_use[order]] = 100.0 * np.arange(len(idx_use)) / len(idx_use)

        picks, counts = [], []
        for lo, hi in BINS:
            cand = np.where((pct >= lo) & (pct <= hi if hi == 100.0 else pct < hi))[0]
            counts.append(len(cand))
            picks.append(np.sort(rng.choice(cand, min(N_COLS, len(cand)), replace=False)))
        frames = read_frames(video_path, [int(j) for p in picks for j in p])
        rows = []
        for pick in picks:
            tiles = []
            for j in pick:
                img = cv2.resize(frames[int(j)], (2 * 128, 128), interpolation=cv2.INTER_AREA)
                cv2.putText(img, f'{j} p={pct[j]:.2f}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(img)
            tiles += [np.zeros((128, 256, 3), np.uint8)] * (N_COLS - len(tiles))
            rows.append(np.hstack(tiles))
        out = args.out_dir / f'{stem}_motion.jpg'
        cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
        _logger.info(
            f'{stem}: {len(idx_use)} eligible; bin counts {counts}; '
            f'me at p95/99/99.75 = {np.percentile(me[idx_use], [95, 99, 99.75]).round(0)} -> {out}'
        )


if __name__ == '__main__':
    main()
