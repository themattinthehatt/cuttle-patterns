"""Break down which individuals, videos, and time spans make up a cluster.

For each requested cluster, prints:
- share of the cluster's frames per individual;
- per contributing video: frames in the cluster, share of the cluster, share of that
  video's own frames, frame-number range, and the number of temporal bouts — runs of
  cluster frames with no gap longer than `--bout-gap` frames between consecutive ones
  (training frames are subsampled from the raw video, so contiguity is gap-based).

Example:
    python scratch/data-analysis/cluster_composition.py \
        --model iter-1.1_msps-vae-mask_d16 --clusters kmeans_k32 --cluster-ids 11 21 28
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import get_model_dir  # noqa: E402
from cuttle_patterns import paths  # noqa: E402
from cuttle_patterns.metadata import attach_individual_column  # noqa: E402

FPS = 24


def count_bouts(frame_numbers: np.ndarray, gap: int) -> int:
    """Number of runs in sorted `frame_numbers` separated by gaps longer than `gap`."""
    if len(frame_numbers) == 0:
        return 0
    return int((np.diff(np.sort(frame_numbers)) > gap).sum() + 1)


def describe_cluster(df: pd.DataFrame, cluster: int, bout_gap: int, top: int) -> str:
    """Format a text breakdown of one cluster's composition.

    Args:
        df: per-frame table with `cluster`, `video_name`, `frame_number`, `individual`.
        cluster: cluster id to describe.
        bout_gap: max frame gap within a bout.
        top: max number of videos to list.

    Returns:
        multi-line report string.
    """
    sub = df[df['cluster'] == cluster]
    n = len(sub)
    frames_per_video = df['video_name'].value_counts()

    lines = [f'=== cluster {cluster}: {n:,} frames ===', 'individuals (share of cluster):']
    share_indiv = sub['individual'].value_counts(normalize=True)
    lines.append('  ' + ', '.join(
        f'{i} {s:.0%}' for i, s in share_indiv.items() if s >= 0.01
    ))

    lines.append(
        f'videos (top {top}): frames | % of cluster | % of video | frame range | bouts '
        f'(gap > {bout_gap} frames = {bout_gap / FPS:.0f}s)'
    )
    for video_name, group in sorted(
        sub.groupby('video_name'), key=lambda kv: -len(kv[1]),
    )[:top]:
        f = group['frame_number'].to_numpy()
        label = video_name.replace('_Crop', '')
        lines.append(
            f'  {label:<30} {group["individual"].iat[0]:<3} {len(f):>5} | '
            f'{len(f) / n:>4.0%} | {len(f) / frames_per_video[video_name]:>4.0%} | '
            f'{f.min():>6}-{f.max():<6} | {count_bouts(f, bout_gap):>3}'
        )
    n_rest = sub['video_name'].nunique() - top
    if n_rest > 0:
        rest = sub[~sub['video_name'].isin(
            sub['video_name'].value_counts().index[:top]
        )]
        lines.append(f'  ... {n_rest} more videos, {len(rest):,} frames ({len(rest) / n:.0%})')
    return '\n'.join(lines)


def main() -> None:
    """Parse args and print a breakdown per requested cluster."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='iter-1.1_msps-vae-mask_d16')
    parser.add_argument('--clusters', default='kmeans_k32')
    parser.add_argument('--cluster-ids', type=int, nargs='+', required=True)
    parser.add_argument('--bout-gap', type=int, default=1000)
    parser.add_argument('--top', type=int, default=6)
    args = parser.parse_args()

    model_dir = get_model_dir(args.model)
    df = attach_individual_column(
        pd.read_parquet(model_dir / paths.CLUSTERS_RELPATH / f'{args.clusters}.parquet'),
    )
    for cluster in args.cluster_ids:
        print(describe_cluster(df, cluster, args.bout_gap, args.top))
        print()


if __name__ == '__main__':
    main()
