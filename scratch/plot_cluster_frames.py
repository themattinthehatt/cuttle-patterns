"""Plot a grid of representative frames for each cluster in a `cuttle cluster` output.

For a given BEAST model and cluster run, loads the clustering parquet from
`results_dir/beast_models/{model_name}/clusters/{cluster_run}.parquet` and, for each
distinct `cluster` label, randomly samples `--n-frames` member frames, loads their
exported PNGs from `results_dir/beast_frames/{video_name}/img{frame_number:08d}.png`, and
plots them in a grid (default 4 columns x 3 rows). One figure is written per cluster to
`results_dir/beast_frames_qc/clusters/{model_name}_{cluster_run}/`.

Frame selection is uniform-random per cluster for now; a nearest-centroid or other more
"representative" selection strategy may replace this later.

Usage:
    python scratch/plot_cluster_frames.py \
        --model-name iter-1.1_msps-vae_d16 \
        --cluster-run kmeans_k16
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuttle_patterns import paths
from cuttle_patterns.config import load_config

DEFAULT_N_FRAMES = 12
DEFAULT_N_COLS = 4
DEFAULT_SEED = 0


def sample_cluster_frames(
    cluster_df: pd.DataFrame,
    n_frames: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Randomly sample member frames of a single cluster.

    Args:
        cluster_df: rows of a `cuttle cluster` output belonging to one cluster label.
        n_frames: number of frames to sample.
        rng: random number generator, for reproducibility.

    Returns:
        a random subset of `cluster_df` with at most `n_frames` rows; if `cluster_df` has
        fewer than `n_frames` rows, all of them are returned.
    """
    if len(cluster_df) <= n_frames:
        return cluster_df
    idxs = rng.choice(len(cluster_df), size=n_frames, replace=False)
    return cluster_df.iloc[idxs]


def build_frame_path(results_dir: Path, video_name: str, frame_number: int) -> Path:
    """Build the on-disk path to a single exported training frame.

    Args:
        results_dir: root results directory.
        video_name: the video directory name, e.g. `Day1_Tank2_Cuttle1_Resident_Crop`.
        frame_number: the frame's number, as stored in a cluster parquet (not
            zero-padded — the padding is reapplied here to match the on-disk filename
            `cuttle extract` writes).

    Returns:
        path to `results_dir/beast_frames/{video_name}/img{frame_number:08d}.png`.
    """
    return (
        results_dir / paths.BEAST_FRAMES_RELPATH / video_name / f'img{frame_number:08d}.png'
    )


def load_frame_image(frame_path: Path) -> np.ndarray | None:
    """Load a single frame image as RGB.

    Args:
        frame_path: path to a frame PNG.

    Returns:
        an (H, W, 3) RGB array, or None if the file is missing or unreadable.
    """
    if not frame_path.exists():
        print(f'  missing frame image: {frame_path}, skipping')
        return None
    image = cv2.imread(str(frame_path))
    if image is None:
        print(f'  could not read frame image: {frame_path}, skipping')
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot_cluster_grid(
    sample_df: pd.DataFrame,
    results_dir: Path,
    cluster_label: int,
    n_cols: int,
) -> plt.Figure:
    """Plot a cluster's sampled frames in a grid.

    Args:
        sample_df: rows sampled from one cluster, with columns `video_name` and
            `frame_number`.
        results_dir: root results directory, for locating frame images.
        cluster_label: the cluster's integer label, used in the figure title.
        n_cols: number of grid columns; the number of rows is derived from
            `len(sample_df)`.

    Returns:
        the assembled figure.
    """
    n_rows = int(np.ceil(len(sample_df) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)

    for ax, (_, row) in zip(axes.flat, sample_df.iterrows(), strict=False):
        frame_path = build_frame_path(results_dir, row['video_name'], row['frame_number'])
        image = load_frame_image(frame_path)
        if image is not None:
            ax.imshow(image)
        ax.set_title(f"{row['video_name']}\nframe {row['frame_number']}", fontsize=7)
        ax.axis('off')

    for ax in axes.flat[len(sample_df):]:
        ax.axis('off')

    fig.suptitle(f'cluster {cluster_label} (n={len(sample_df)} shown)')
    fig.tight_layout()
    return fig


def main() -> None:
    """Sample and plot representative frames for every cluster in a cluster run."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help='BEAST model directory name, e.g. iter-1.1_msps-vae_d16',
    )
    parser.add_argument(
        '--cluster-run',
        required=True,
        help='stem of the cluster parquet file, e.g. kmeans_k16',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='override the results directory from config',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=DEFAULT_N_FRAMES,
        help='number of frames to sample per cluster',
    )
    parser.add_argument(
        '--n-cols',
        type=int,
        default=DEFAULT_N_COLS,
        help='number of grid columns; rows are derived from --n-frames',
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir is not None else load_config().results_dir

    cluster_path = (
        results_dir
        / paths.BEAST_MODELS_RELPATH
        / args.model_name
        / paths.CLUSTERS_RELPATH
        / f'{args.cluster_run}.parquet'
    )
    if not cluster_path.exists():
        raise FileNotFoundError(f'no cluster file found at {cluster_path}')
    df = pd.read_parquet(cluster_path)

    output_dir = (
        results_dir
        / paths.CLUSTER_FRAME_GRIDS_RELPATH
        / f'{args.model_name}_{args.cluster_run}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cluster_width = max(2, len(str(int(df['cluster'].max()))))

    for cluster_label in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_label]
        sample_df = sample_cluster_frames(cluster_df, args.n_frames, rng)

        fig = plot_cluster_grid(sample_df, results_dir, cluster_label, args.n_cols)
        output_path = output_dir / f'cluster_{cluster_label:0{cluster_width}d}.png'
        if output_path.exists():
            print(f'warning: overwriting existing {output_path}')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f'cluster {cluster_label}: {len(cluster_df)} members, wrote {output_path}')


if __name__ == '__main__':
    main()
