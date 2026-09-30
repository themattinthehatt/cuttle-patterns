"""Horizontal bar plot of how many videos (or individuals) each cluster draws frames from.

Each bar is one cluster, labeled by a thumbnail of the frame nearest its centroid (in the
space the clustering ran on — `z_u` for MSPS-VAE models) plus a small cluster ID. Bar length
is the *effective* number of videos, `exp(entropy)` of the cluster's per-video frame
distribution (dark bar), drawn over the raw number of videos with at least one frame in the
cluster (light track) — the raw count saturates near the total since a single stray frame
counts, so the effective count is the informative one: a cluster touching 30 videos but
dominated by 3 of them has a dark bar near 3. Clusters are sorted by effective count, and
each cluster's frame count is printed to the right of the axis.

`--group individual` counts individual animals (`R1`-`R6`, `I1`-`I6`) instead of videos and
writes `individuals_per_cluster.png`. Note individuals contribute unequal total frame counts
(each appears in a different number of videos), so an identity-free cluster would score
somewhat below the number of individuals, not exactly at it. A dotted line marks that
identity-free reference: `exp(entropy)` of the whole dataset's per-group frame distribution.

Example:
    python scratch/data-analysis/plot_videos_per_cluster.py \
        --model iter-1.1_msps-vae_d16 --clusters kmeans_k16
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    RESULTS_DIR,
    effective_count,
    get_analysis_dir,
    get_cluster_space,
    get_model_dir,
    load_model_latents,
    setup_logging,
)
from cuttle_patterns import paths  # noqa: E402
from cuttle_patterns.metadata import INDIVIDUAL_COLUMN  # noqa: E402

logger = logging.getLogger(__name__)

BAR_COLOR = '#2a78d6'
TRACK_COLOR = '#cde2fb'
TEXT_MUTED = '#6b6b6b'
THUMB_ZOOM = 0.42

# --group choices: (dataframe column, plural noun for labels/filenames)
GROUPS = {
    'video': ('video_name', 'videos'),
    'individual': (INDIVIDUAL_COLUMN, 'individuals'),
    'session': ('session', 'sessions'),
}


def summarize_clusters(
    df: pd.DataFrame,
    X: np.ndarray,
    group_col: str = 'video_name',
) -> pd.DataFrame:
    """Per-cluster group counts, effective group counts, and nearest-to-centroid frame.

    Args:
        df: per-frame table with `cluster`, `video_name`, `frame_number`, and `group_col`,
            row-aligned with `X`.
        X: clustering-space latents.
        group_col: column whose distinct values are counted per cluster.

    Returns:
        one row per cluster with `n_groups`, `n_groups_eff`, `n_frames`,
        `video_name_center`, `frame_number_center`.
    """
    rows = []
    for cluster, group in df.groupby('cluster'):
        n_groups = group[group_col].nunique()
        X_c = X[group.index]
        idx_center = group.index[np.argmin(np.linalg.norm(X_c - X_c.mean(axis=0), axis=1))]
        rows.append({
            'cluster': cluster,
            'n_groups': n_groups,
            'n_groups_eff': effective_count(group[group_col]),
            'n_frames': len(group),
            'video_name_center': df.at[idx_center, 'video_name'],
            'frame_number_center': df.at[idx_center, 'frame_number'],
        })
    return pd.DataFrame(rows)


def draw_cluster_column(
    ax: plt.Axes,
    summary: pd.DataFrame,
    n_rows: int,
    n_groups_total: int,
    noun: str,
    eff_reference: float,
) -> None:
    """Draw one column of cluster bars, top row first, with thumbnail y-labels.

    Args:
        ax: axes to draw into.
        summary: rows of `summarize_clusters` output for this column, top to bottom.
        n_rows: rows per column (keeps bar heights equal across columns).
        n_groups_total: total number of groups in the dataset (x-axis limit).
        noun: plural group name for labels, e.g. `videos`.
        eff_reference: effective count if the cluster mixed groups like the whole dataset (dotted line).
    """
    y = n_rows - 1 - np.arange(len(summary))
    ax.barh(
        y, summary['n_groups'], height=0.62, color=TRACK_COLOR, label=f'{noun} with ≥1 frame',
    )
    ax.barh(
        y, summary['n_groups_eff'], height=0.62, color=BAR_COLOR,
        label=f'effective {noun}, exp(entropy)',
    )

    frames_dir = RESULTS_DIR / paths.BEAST_FRAMES_RELPATH
    for yi, (_, r) in zip(y, summary.iterrows()):
        ax.annotate(
            f'{r["n_groups_eff"]:.1f}', (r['n_groups_eff'], yi), xytext=(-4, 0),
            textcoords='offset points', va='center', ha='right', fontsize=7, color='white',
        )
        ax.annotate(
            f'{r["n_frames"]:,} frames', (1, yi), xytext=(6, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points',
            va='center', ha='left', fontsize=7, color=TEXT_MUTED, annotation_clip=False,
        )
        # thumbnail + small cluster id in place of a y tick label
        path = frames_dir / r['video_name_center'] / f'img{r["frame_number_center"]:08d}.png'
        thumb = OffsetImage(np.asarray(Image.open(path).convert('RGB')), zoom=THUMB_ZOOM)
        ax.add_artist(AnnotationBbox(
            thumb, (0, yi), xybox=(-52, 0), xycoords=('axes fraction', 'data'),
            boxcoords='offset points', frameon=False, annotation_clip=False,
        ))
        ax.annotate(
            f'c{r["cluster"]:02d}', (0, yi), xytext=(-100, 0),
            xycoords=('axes fraction', 'data'), textcoords='offset points',
            va='center', ha='right', fontsize=7, color=TEXT_MUTED,
            annotation_clip=False,
        )

    ax.set_yticks([])
    ax.axvline(
        eff_reference, color='#1a1a1a', linewidth=1, linestyle=':', zorder=3,
        label=f'even-mix reference ({eff_reference:.1f})',
    )
    ax.set_xlim(0, n_groups_total)
    ax.set_ylim(-0.6, n_rows - 0.4)
    ax.set_xlabel(f'number of {noun}')
    ax.grid(axis='x', color='#e5e5e5', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#9a9a9a')


def plot_groups_per_cluster(
    summary: pd.DataFrame,
    n_groups_total: int,
    title: str,
    noun: str,
    eff_reference: float,
    n_columns: int = 1,
) -> plt.Figure:
    """Draw the horizontal bar chart, split into side-by-side columns of equal width.

    Clusters are sorted by effective group count, highest first, reading down the first
    column and continuing at the top of the next.

    Args:
        summary: output of `summarize_clusters`.
        n_groups_total: total number of groups in the dataset (x-axis limit).
        title: figure title.
        noun: plural group name for labels, e.g. `videos`.
        eff_reference: effective count if the cluster mixed groups like the whole dataset (dotted line).
        n_columns: number of side-by-side columns.

    Returns:
        the matplotlib figure.
    """
    summary = summary.sort_values('n_groups_eff', ascending=False).reset_index(drop=True)
    n_rows = int(np.ceil(len(summary) / n_columns))

    # fixed per-column geometry in inches, so each column matches the single-column plot
    col_w, margin_l, margin_r = 8.0, 1.75, 1.1
    margin_t, margin_b = 0.75, 0.55
    fig_w = col_w * n_columns
    fig_h = 0.62 * n_rows + margin_t + margin_b
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for idx_col in range(n_columns):
        ax = fig.add_axes([
            (idx_col * col_w + margin_l) / fig_w,
            margin_b / fig_h,
            (col_w - margin_l - margin_r) / fig_w,
            (fig_h - margin_t - margin_b) / fig_h,
        ])
        chunk = summary.iloc[idx_col * n_rows:(idx_col + 1) * n_rows]
        draw_cluster_column(ax, chunk, n_rows, n_groups_total, noun, eff_reference)
        axes.append(ax)

    axes[0].set_title(title, loc='left', fontsize=10, pad=24)
    axes[0].legend(
        loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7, frameon=False,
        borderaxespad=0.2,
    )
    return fig


def save_groups_per_cluster(
    df: pd.DataFrame,
    X: np.ndarray,
    group: str,
    title: str,
    out_path: Path,
    n_columns: int | None = None,
) -> pd.DataFrame:
    """Summarize, plot, and save a groups-per-cluster figure.

    Args:
        df: per-frame table with `cluster`, `video_name`, `frame_number`, and the column
            for `group`, row-aligned with `X` (index must be 0..n-1).
        X: clustering-space latents, for picking each cluster's thumbnail.
        group: a key of `GROUPS`.
        title: figure title.
        out_path: where to save the PNG.
        n_columns: number of side-by-side columns; default one per 16 clusters.

    Returns:
        the per-cluster summary table.
    """
    group_col, noun = GROUPS[group]
    summary = summarize_clusters(df, X, group_col)
    fig = plot_groups_per_cluster(
        summary,
        n_groups_total=df[group_col].nunique(),
        title=title,
        noun=noun,
        eff_reference=effective_count(df[group_col]),
        n_columns=n_columns or int(np.ceil(len(summary) / 16)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f'wrote {out_path}')
    return summary


def main() -> None:
    """Parse args, summarize clusters, and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='iter-1.1_msps-vae_d16')
    parser.add_argument('--clusters', default='kmeans_k16')
    parser.add_argument('--group', choices=sorted(GROUPS), default='video')
    parser.add_argument(
        '--n-columns', type=int, default=None, help='default: one column per 16 clusters',
    )
    args = parser.parse_args()
    setup_logging()

    model_dir = get_model_dir(args.model)
    clusters = pd.read_parquet(model_dir / paths.CLUSTERS_RELPATH / f'{args.clusters}.parquet')
    X, meta = load_model_latents(model_dir)
    X = get_cluster_space(X, model_dir)
    df = meta.merge(
        clusters[['video_name', 'frame_number', 'cluster']],
        on=['video_name', 'frame_number'],
        how='left',
        validate='one_to_one',
    )
    if df['cluster'].isna().any():
        raise ValueError('some latents have no cluster label')

    noun = GROUPS[args.group][1]
    summary = save_groups_per_cluster(
        df,
        X,
        args.group,
        title=f'{args.model} · {args.clusters}: {noun} per cluster',
        out_path=(
            get_analysis_dir(args.model) / paths.CLUSTERS_RELPATH / args.clusters
            / f'{noun}_per_cluster.png'
        ),
        n_columns=args.n_columns,
    )
    print(summary.sort_values('n_groups').to_string(index=False))


if __name__ == '__main__':
    main()
