"""Sample and plot representative frames for each cluster in a `cuttle cluster` output.

For a given BEAST model and cluster run, samples random member frames per distinct
`cluster` label and assembles them into a grid figure, for `cuttle clusterview`.

Frame selection is uniform-random per cluster for now; a nearest-centroid or other more
"representative" selection strategy may replace this later.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuttle_patterns import paths

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

    images = []
    titles = []
    for _, row in sample_df.iterrows():
        frame_path = build_frame_path(results_dir, row['video_name'], row['frame_number'])
        images.append(load_frame_image(frame_path))
        titles.append(f"{row['video_name']}\nframe {row['frame_number']}")

    # size each subplot's height to the frames' actual aspect ratio (BEAST crops are
    # wide rectangles, not square) so imshow doesn't letterbox and waste vertical space
    first_valid = next((image for image in images if image is not None), None)
    aspect = first_valid.shape[0] / first_valid.shape[1] if first_valid is not None else 0.5
    col_width = 2.5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_width * n_cols, col_width * aspect * n_rows + 0.4 * n_rows),
        squeeze=False,
    )

    for ax, image, title in zip(axes.flat, images, titles, strict=False):
        if image is not None:
            ax.imshow(image)
        ax.set_title(title, fontsize=6, pad=3)
        ax.axis('off')

    for ax in axes.flat[len(sample_df):]:
        ax.axis('off')

    fig.suptitle(f'cluster {cluster_label} (n={len(sample_df)} shown)', y=0.995, fontsize=10)
    fig.subplots_adjust(
        top=1 - 0.4 / (col_width * aspect * n_rows + 0.4 * n_rows),
        bottom=0.01, left=0.01, right=0.99, hspace=0.5, wspace=0.05,
    )
    return fig
