"""Count eligible frames per video under the eligibility criteria and save per-video tables.

Criteria (see README.md): not blank; all keypoints >= 0.9 likelihood; rectangle not too small;
no keypoint jump > 25 px. Writes one parquet per video (one boolean flag column per
criterion, True = frame fails that criterion, plus `eligible`) and a summary csv.

Usage:
    python count_eligible.py --out-dir <dir>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from cuttle_patterns.config import load_config
from cuttle_patterns.preprocessing.extract import compute_small_rectangle_mask
from cuttle_patterns.preprocessing.pose import load_pose_predictions
from jump_contact_sheets import POSE_DIR, compute_jumps

_logger = logging.getLogger(__name__)

JUMP_MAX = 25.0
TARGET_PER_VIDEO = 1000


def main() -> None:
    """Build per-video eligibility tables and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    rect_dir = config.results_dir / 'rectangles'
    rows = []
    for pose_path in sorted(POSE_DIR.glob('*.csv')):
        stem = pose_path.stem
        rect_path = rect_dir / f'{stem}.csv'
        blank_path = config.data_dir / f'{stem.removesuffix("_Crop")}_black_frames.txt'
        pose_df = load_pose_predictions(pose_path)
        n = len(pose_df)

        blank = np.zeros(n, dtype=bool)
        if blank_path.exists():
            idx = np.array([int(v) for v in blank_path.read_text().split()], dtype=int)
            blank[idx[idx < n]] = True
        else:
            _logger.warning(f'{stem}: no blank-frame file')
        _, ok = compute_jumps(pose_df)
        low_lik = ~ok
        jump, _ = compute_jumps(pose_df)
        big_jump = jump > JUMP_MAX  # nan (no valid step) compares False
        if rect_path.exists():
            try:
                small_rect = compute_small_rectangle_mask(rect_path, pose_df)
            except ValueError as e:
                _logger.warning(f'{stem}: {e}')
                small_rect = np.zeros(n, dtype=bool)
        else:
            _logger.warning(f'{stem}: no rectangle csv')
            small_rect = np.zeros(n, dtype=bool)

        flags = pd.DataFrame({
            'frame_idx': np.arange(n),
            'blank': blank,
            'low_likelihood': low_lik,
            'small_rect': small_rect,
            'big_jump': big_jump,
        })
        fail = flags[['blank', 'low_likelihood', 'small_rect', 'big_jump']].to_numpy()
        flags['eligible'] = ~fail.any(axis=1)
        flags.to_parquet(args.out_dir / f'{stem}.parquet')
        rows.append({
            'video': stem,
            'n_frames': n,
            'blank': blank.sum(),
            'low_likelihood': low_lik.sum(),
            'small_rect': small_rect.sum(),
            'big_jump': big_jump.sum(),
            'only_big_jump': (big_jump & ~(blank | low_lik | small_rect)).sum(),
            'eligible': int(flags['eligible'].sum()),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / 'summary.csv', index=False)
    summary['frac'] = summary['eligible'] / summary['n_frames']
    print(summary.drop(columns='video').describe().round(1).to_string())
    print(f'total eligible: {summary["eligible"].sum()} / {summary["n_frames"].sum()}')
    short = summary[summary['eligible'] < TARGET_PER_VIDEO]
    print(f'videos with < {TARGET_PER_VIDEO} eligible frames: {len(short)}')
    print(short[['video', 'n_frames', 'eligible']].to_string(index=False))
    print('lowest 10 by eligible:')
    print(summary.nsmallest(10, 'eligible')[['video', 'n_frames', 'eligible', 'frac']]
          .to_string(index=False))


if __name__ == '__main__':
    main()
