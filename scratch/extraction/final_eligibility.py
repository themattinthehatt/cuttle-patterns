"""Build the final per-video eligibility + label tables (criteria 1-6, see README.md).

Status: superseded by the `cuttle extract-*` pipeline (see README.md).

Combines, per video: the criteria 1-4 flags from `count_eligible.py`, the 99th-percentile
motion-energy gate (computed among frames passing 1-4, on the aligned rectangle video, frame 0
excluded), the >= 5% pure-black gate, and the classifier's per-frame predictions. Row i is
frame i, matching the per-video DINOv3 `.npy` files. Days listed in `--exclude-days` (default
day 1, per the photometry decision) are skipped.

Usage:
    python final_eligibility.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cuttle_patterns.config import load_config

_logger = logging.getLogger(__name__)

EXTRACTION_DIR = Path('/media/mattw/CUTTLE/results/.extraction')
CLASSIFIER_NAME = '2026-11-07_classifier_model'
ME_PERCENTILE = 99.0
BLACK_MAX = 0.05


def build_table(stem: str, classifications_dir: Path) -> pd.DataFrame:
    """Build one video's final table.

    Args:
        stem: video stem, e.g. `Day5_Tank3_Cuttle2_Resident_Crop`.
        classifications_dir: directory of per-video classifier parquets.

    Returns:
        frame with `frame_idx`, the individual criteria flags (True = fails), `eligible`,
        `motion_energy`, `black_frac`, `predicted_pattern` and `confidence`.

    Raises:
        ValueError: if the per-frame inputs disagree in length.
    """
    flags = pd.read_parquet(EXTRACTION_DIR / 'eligibility' / f'{stem}.parquet')
    me = np.load(EXTRACTION_DIR / 'me_cache_rect' / f'{stem}_me.npy')
    black = np.load(EXTRACTION_DIR / 'black_cache_rect' / f'{stem}_black.npy')
    clf = pd.read_parquet(classifications_dir / f'{stem}.parquet')
    n = len(flags)
    if not (len(me) == len(black) == len(clf) == n):
        raise ValueError(
            f'{stem}: length mismatch (flags {n}, me {len(me)}, black {len(black)}, '
            f'classifier {len(clf)})'
        )

    pass_1_4 = flags['eligible'].to_numpy().copy()
    pass_1_4[0] = False  # frame 0's backward-difference motion energy is meaningless
    me_cut = np.percentile(me[pass_1_4], ME_PERCENTILE)
    out = flags.drop(columns='eligible').copy()
    out['me_top'] = pass_1_4 & (me > me_cut)
    out['black_high'] = black >= BLACK_MAX
    out['frame0'] = np.arange(n) == 0
    out['eligible'] = pass_1_4 & ~out['me_top'].to_numpy() & ~out['black_high'].to_numpy()
    out['motion_energy'] = me.astype(np.float32)
    out['black_frac'] = black.astype(np.float32)
    out['predicted_pattern'] = clf['predicted_pattern'].to_numpy()
    out['confidence'] = clf['confidence'].to_numpy()
    return out


def main() -> None:
    """Write final tables for every video not on an excluded day."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=EXTRACTION_DIR / 'eligibility_final')
    parser.add_argument('--exclude-days', type=int, nargs='*', default=[1])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classifications_dir = load_config().results_dir / 'classifications' / CLASSIFIER_NAME
    stems = sorted(p.stem for p in (EXTRACTION_DIR / 'eligibility').glob('*.parquet'))
    stems = [s for s in stems if int(s.split('_')[0].removeprefix('Day')) not in args.exclude_days]
    n_eligible = 0
    for stem in tqdm(stems, unit='video'):
        table = build_table(stem, classifications_dir)
        table.to_parquet(args.out_dir / f'{stem}.parquet')
        n_eligible += int(table['eligible'].sum())
    _logger.info(f'{len(stems)} videos, {n_eligible} eligible frames -> {args.out_dir}')


if __name__ == '__main__':
    main()
