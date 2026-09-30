"""Re-embed and re-cluster a model's latents within a single individual.

Asks whether, within one animal, the latent space still separates by session — and,
comparing a resident (same tank every day) to an intruder (different tank every day),
whether that separation tracks tank/camera changes or day-level differences.

For one model and one individual, this:
1. fits UMAP and k-means on that individual's frames only (in the space `cuttle cluster`
   uses — `z_u` for MSPS models);
2. plots the UMAP colored by session, by the within-individual clusters, and by the
   model's global clusters;
3. plots sessions-per-cluster (effective number of sessions per within-individual cluster,
   with a nearest-to-centroid thumbnail per cluster) and writes a `clusterview`-style frame
   grid per cluster;
4. computes session separability: k-NN session decoding accuracy under *temporally blocked*
   cross-validation (adjacent frames are near-duplicates, so random splits would leak), next
   to shuffled-split accuracy for reference, plus the frame-weighted mean effective number
   of sessions per cluster.

Outputs go to `beast_models/{model}/analysis/subsets/individual_{id}/`, laid out like a
model directory (see `common.py`). `--summarize` instead collects every subset's
`separability.json` across models into one table.

Example:
    python scratch/data-analysis/subset_analysis.py \
        --model iter-1.1_msps-vae-mask_d16 --individual I2
    python scratch/data-analysis/subset_analysis.py --summarize
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    RESULTS_DIR,
    SUBSETS_RELPATH,
    effective_count,
    get_cluster_space,
    get_model_dir,
    load_model_latents,
    setup_logging,
)
from cuttle_patterns import paths  # noqa: E402
from cuttle_patterns.cluster import build_cluster_dataframe, run_kmeans  # noqa: E402
from cuttle_patterns.cluster import hparams_to_str as cluster_hparams_to_str  # noqa: E402
from cuttle_patterns.reduce import (  # noqa: E402
    DEFAULT_METRIC,
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    build_umap_dataframe,
    run_umap,
)
from cuttle_patterns.reduce import hparams_to_str as umap_hparams_to_str  # noqa: E402
from cuttle_patterns.visualization.cluster_frames import (  # noqa: E402
    DEFAULT_N_COLS,
    DEFAULT_N_FRAMES,
    plot_cluster_grid,
    sample_cluster_frames,
)
from plot_videos_per_cluster import save_groups_per_cluster  # noqa: E402

logger = logging.getLogger(__name__)

# categorical slots 1-3 of the dataviz reference palette
SESSION_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
N_TIME_BLOCKS = 10
KNN_NEIGHBORS = 15
N_FOLDS = 5


def session_decoding_accuracy(
    X: np.ndarray,
    meta: pd.DataFrame,
    seed: int,
) -> dict[str, float]:
    """k-NN session decoding accuracy, temporally blocked and shuffled.

    Blocked CV splits each session's frames into `N_TIME_BLOCKS` contiguous time blocks
    and holds out whole blocks, so a test frame's near-duplicate temporal neighbors are
    never in the training set. Shuffled CV is reported for reference only — it is
    inflated by exactly that leakage.

    Args:
        X: latents for the subset, row-aligned with `meta`.
        meta: subset metadata with `session` and `frame_number` columns.
        seed: random seed for the shuffled split.

    Returns:
        dict with `blocked` and `shuffled` balanced accuracies and `chance`.
    """
    y = meta['session'].to_numpy()
    # time-block id per frame: rank within session, cut into equal-count blocks
    rank = meta.groupby('session')['frame_number'].rank(method='first') - 1
    size = meta.groupby('session')['frame_number'].transform('size')
    block = (rank * N_TIME_BLOCKS // size).astype(int)
    groups = meta['session'] + '_' + block.astype(str)

    def cv_accuracy(splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
        y_pred = np.empty_like(y)
        for idx_train, idx_test in splits:
            knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)
            knn.fit(X[idx_train], y[idx_train])
            y_pred[idx_test] = knn.predict(X[idx_test])
        return float(balanced_accuracy_score(y, y_pred))

    blocked = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    shuffled = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y))
    return {
        'blocked': cv_accuracy(blocked),
        'shuffled': cv_accuracy(shuffled),
        'chance': 1 / len(np.unique(y)),
    }


def scatter_categorical(
    ax: plt.Axes,
    xy: np.ndarray,
    labels: pd.Series,
    colors: list[str] | None = None,
    label_centroids: bool = False,
) -> None:
    """Scatter points colored by a categorical label.

    Args:
        ax: axes to draw into.
        xy: (n, 2) UMAP coordinates.
        labels: per-point category.
        colors: one color per sorted category; defaults to matplotlib's `tab20`.
        label_centroids: if True, write each category's name at its median position so
            identity doesn't rely on color alone (needed beyond ~8 categories).
    """
    categories = sorted(labels.unique())
    if colors is None:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(categories))]
    order = np.random.default_rng(0).permutation(len(xy))  # avoid last-drawn-on-top bias
    color_of = dict(zip(categories, colors))
    ax.scatter(
        xy[order, 0], xy[order, 1], s=2, linewidths=0, alpha=0.6,
        c=[color_of[v] for v in labels.to_numpy()[order]],
    )
    if label_centroids:
        for cat in categories:
            center = np.median(xy[labels.to_numpy() == cat], axis=0)
            ax.text(
                center[0], center[1], str(cat), fontsize=7, ha='center', va='center',
                bbox={'boxstyle': 'round,pad=0.15', 'fc': 'white', 'ec': 'none', 'alpha': 0.8},
            )
    else:
        for cat in categories:
            ax.scatter([], [], s=20, color=color_of[cat], label=str(cat))
        ax.legend(fontsize=7, frameon=False, markerscale=1, loc='best')
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def plot_umap_panels(df: pd.DataFrame, title: str, k: int) -> plt.Figure:
    """Three UMAP panels: by session, by within-individual cluster, by global cluster."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    xy = df[['umap_x', 'umap_y']].to_numpy()
    sessions = sorted(df['session'].unique())
    scatter_categorical(axes[0], xy, df['session'], colors=SESSION_COLORS[:len(sessions)])
    axes[0].set_title('session', loc='left', fontsize=10)
    scatter_categorical(axes[1], xy, df['cluster'], label_centroids=True)
    axes[1].set_title(f'within-individual k-means (k={k})', loc='left', fontsize=10)
    scatter_categorical(axes[2], xy, df['cluster_global'], label_centroids=True)
    axes[2].set_title('global k-means cluster (whole dataset)', loc='left', fontsize=10)
    fig.suptitle(title, x=0.01, ha='left', fontsize=11)
    fig.tight_layout()
    return fig


def run_subset(args: argparse.Namespace) -> dict:
    """Run the full per-individual analysis for one model; returns the separability dict."""
    model_dir = get_model_dir(args.model)
    subset_dir = model_dir / SUBSETS_RELPATH / f'individual_{args.individual}'
    figures_dir = subset_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    X_all, meta_all = load_model_latents(model_dir)
    X_all = get_cluster_space(X_all, model_dir)
    mask = (meta_all['individual'] == args.individual).to_numpy()
    if not mask.any():
        raise ValueError(f'no frames for individual {args.individual}')
    X = X_all[mask]
    meta = meta_all[mask].reset_index(drop=True)
    role = meta['role'].iat[0]
    logger.info(
        f'{args.model} / {args.individual} ({role}): {len(meta):,} frames, sessions '
        f'{sorted(meta["session"].unique())}'
    )

    # within-individual UMAP and k-means, saved like `cuttle reduce`/`cuttle cluster`
    umap_hparams = umap_hparams_to_str(args.umap_nn, args.umap_md, args.umap_metric)
    xy = run_umap(X, n_neighbors=args.umap_nn, min_dist=args.umap_md, metric=args.umap_metric)
    umap_df = build_umap_dataframe(meta, xy)
    (subset_dir / paths.REDUCE_RELPATH).mkdir(parents=True, exist_ok=True)
    umap_df.to_parquet(subset_dir / paths.REDUCE_RELPATH / f'umap_{umap_hparams}.parquet')

    cluster_name = f'kmeans_{cluster_hparams_to_str(args.k)}'
    labels = run_kmeans(X, n_clusters=args.k)
    cluster_df = build_cluster_dataframe(meta, labels)
    (subset_dir / paths.CLUSTERS_RELPATH).mkdir(parents=True, exist_ok=True)
    cluster_df.to_parquet(subset_dir / paths.CLUSTERS_RELPATH / f'{cluster_name}.parquet')

    df = meta.copy()
    df['cluster'] = labels
    df['umap_x'] = xy[:, 0]
    df['umap_y'] = xy[:, 1]
    global_clusters = pd.read_parquet(
        model_dir / paths.CLUSTERS_RELPATH / f'{args.global_clusters}.parquet',
        columns=['video_name', 'frame_number', 'cluster'],
    ).rename(columns={'cluster': 'cluster_global'})
    df = df.merge(global_clusters, on=['video_name', 'frame_number'], how='left')

    title = f'{args.model} · {args.individual} ({role}) · {len(df):,} frames'
    fig = plot_umap_panels(df, f'{title} · UMAP {umap_hparams}', args.k)
    fig.savefig(figures_dir / f'umap_{umap_hparams}_{cluster_name}.png', dpi=150)
    plt.close(fig)

    save_groups_per_cluster(
        df, X, 'session',
        title=f'{title}\n{cluster_name}: sessions per cluster',
        out_path=figures_dir / f'sessions_per_cluster_{cluster_name}.png',
    )
    pd.crosstab(df['cluster'], df['session']).to_csv(
        figures_dir / f'cluster_by_session_{cluster_name}.csv',
    )

    # clusterview-style grids
    grids_dir = subset_dir / paths.CLUSTERS_RELPATH / cluster_name
    grids_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for cluster, group in cluster_df.groupby('cluster'):
        sample = sample_cluster_frames(group, DEFAULT_N_FRAMES, rng)
        fig = plot_cluster_grid(sample, RESULTS_DIR, cluster, DEFAULT_N_COLS)
        fig.savefig(grids_dir / f'cluster_{cluster:02d}.png', dpi=150)
        plt.close(fig)

    # separability
    acc = session_decoding_accuracy(X, meta, args.seed)
    eff_per_cluster = df.groupby('cluster')['session'].apply(effective_count)
    weights = df['cluster'].value_counts(normalize=True)
    separability = {
        'model': args.model,
        'individual': args.individual,
        'role': role,
        'sessions': sorted(meta['session'].unique()),
        'n_frames': len(meta),
        'knn_session_acc_blocked': acc['blocked'],
        'knn_session_acc_shuffled': acc['shuffled'],
        'chance': acc['chance'],
        'eff_sessions_per_cluster': float((eff_per_cluster * weights).sum()),
        'eff_sessions_reference': effective_count(meta['session']),
        'cluster_name': cluster_name,
        'umap_hparams': umap_hparams,
    }
    with (figures_dir / 'separability.json').open('w') as f:
        json.dump(separability, f, indent=2)
    with (subset_dir / 'subset.json').open('w') as f:
        json.dump({
            'filter': {'individual': args.individual},
            'role': role,
            'n_frames': len(meta),
            'videos': sorted(meta['video_name'].unique()),
            'latent_space': 'cluster space (z_u for msps_vae models)',
            'global_clusters': args.global_clusters,
        }, f, indent=2)
    logger.info(json.dumps(separability, indent=2))
    return separability


def summarize() -> pd.DataFrame:
    """Collect every model's subset `separability.json` into one table."""
    rows = []
    pattern = f'*/{SUBSETS_RELPATH}/*/figures/separability.json'
    for path in sorted((RESULTS_DIR / paths.BEAST_MODELS_RELPATH).glob(pattern)):
        with path.open() as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)
    cols = [
        'model', 'individual', 'role', 'n_frames', 'chance', 'knn_session_acc_blocked',
        'knn_session_acc_shuffled', 'eff_sessions_per_cluster', 'eff_sessions_reference',
    ]
    return df[cols].sort_values(['individual', 'model'])


def main() -> None:
    """Parse args and run one subset analysis, or summarize all of them."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='iter-1.1_msps-vae-mask_d16')
    parser.add_argument('--individual', default='I2')
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--global-clusters', default='kmeans_k16')
    parser.add_argument('--umap-nn', type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument('--umap-md', type=float, default=DEFAULT_MIN_DIST)
    parser.add_argument('--umap-metric', default=DEFAULT_METRIC)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--summarize', action='store_true', help='print the cross-model table')
    args = parser.parse_args()
    setup_logging()
    pd.set_option('display.width', 200)

    if args.summarize:
        print(summarize().round(3).to_string(index=False))
    else:
        run_subset(args)


if __name__ == '__main__':
    main()
