"""Score a set of embedders and assemble the scoreboard.

`score_embedder`/`run_report` are the functions most callers want — see
`cuttle_patterns/eval/README.md` for a full walkthrough, or
`cuttle_patterns/eval/run_core.py` for a ready-made script that wires these up against
real `results_dir` models.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cuttle_patterns.eval.clustering import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_SEEDS,
    prepare_for_clustering,
    run_kmeans_multiseed,
)
from cuttle_patterns.eval.load_embeddings import EmbedderSpec, load_embedder_matrix
from cuttle_patterns.eval.metrics import (
    DEFAULT_CV_SPLITS,
    DEFAULT_KNN_K,
    ami_cluster_vs_class,
    ami_cluster_vs_video_within_class,
    cross_video_knn_accuracy,
    effective_videos_per_cluster,
    linear_probe_accuracy,
)

SCOREBOARD_COLUMNS = [
    'embedder',
    'ami_cluster_class_mean',
    'ami_cluster_class_std',
    'ami_cluster_video_within_class_mean',
    'ami_cluster_video_within_class_std',
    'effective_videos_per_cluster_mean',
    'effective_videos_per_cluster_std',
    'cross_video_knn_accuracy',
    'linear_probe_accuracy',
    'variance_retained',
]


def score_embedder(
    spec: EmbedderSpec,
    manifest: pd.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    knn_k: int = DEFAULT_KNN_K,
    cv_splits: int = DEFAULT_CV_SPLITS,
) -> dict:
    """Compute the five core metrics for one embedder.

    Args:
        spec: which model/subspace to score.
        manifest: eval manifest from `eval.build_manifest.build_eval_manifest`,
            providing `predicted_pattern` (pattern proxy) and `video_name` (identity)
            per frame.
        n_components: PCA target dimension for clustering (see
            `eval.clustering.prepare_for_clustering`).
        n_clusters: k-means k.
        seeds: k-means seeds to average clustering-based metrics over.
        knn_k: `eval.metrics.cross_video_knn_accuracy`'s k.
        cv_splits: `eval.metrics.linear_probe_accuracy`'s number of `GroupKFold` folds.

    Returns:
        dict of scoreboard values for this embedder, keyed as in `SCOREBOARD_COLUMNS`.
    """
    X = load_embedder_matrix(spec, manifest)
    class_labels = manifest['predicted_pattern'].to_numpy()
    video_labels = manifest['video_name'].to_numpy()

    X_reduced, variance_retained = prepare_for_clustering(X, n_components=n_components)
    cluster_labels_per_seed = run_kmeans_multiseed(X_reduced, n_clusters=n_clusters, seeds=seeds)

    ami_class = [ami_cluster_vs_class(labels, class_labels) for labels in cluster_labels_per_seed]
    ami_video = [
        ami_cluster_vs_video_within_class(labels, video_labels, class_labels)
        for labels in cluster_labels_per_seed
    ]
    eff_videos = [
        effective_videos_per_cluster(labels, video_labels) for labels in cluster_labels_per_seed
    ]

    return {
        'embedder': spec.id,
        'ami_cluster_class_mean': float(np.mean(ami_class)),
        'ami_cluster_class_std': float(np.std(ami_class)),
        'ami_cluster_video_within_class_mean': float(np.mean(ami_video)),
        'ami_cluster_video_within_class_std': float(np.std(ami_video)),
        'effective_videos_per_cluster_mean': float(np.mean(eff_videos)),
        'effective_videos_per_cluster_std': float(np.std(eff_videos)),
        'cross_video_knn_accuracy': cross_video_knn_accuracy(
            X, class_labels, video_labels, k=knn_k,
        ),
        'linear_probe_accuracy': linear_probe_accuracy(
            X, class_labels, video_labels, n_splits=cv_splits,
        ),
        'variance_retained': variance_retained,
    }


def run_report(specs: list[EmbedderSpec], manifest: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Score every embedder in `specs` and assemble the scoreboard.

    Args:
        specs: embedders to score.
        manifest: eval manifest from `eval.build_manifest.build_eval_manifest`.
        **kwargs: forwarded to `score_embedder` (e.g. `n_clusters`, `knn_k`).

    Returns:
        one row per embedder, columns as in `SCOREBOARD_COLUMNS`.
    """
    rows = [score_embedder(spec, manifest, **kwargs) for spec in specs]
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def _to_markdown_table(frame: pd.DataFrame) -> str:
    """Format a DataFrame as a markdown table without depending on `tabulate`."""
    header = '| ' + ' | '.join(frame.columns) + ' |'
    separator = '|' + '|'.join(['---'] * len(frame.columns)) + '|'
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = [f'{v:.3f}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def write_report(scoreboard: pd.DataFrame, out_dir: Path) -> None:
    """Write the scoreboard as both markdown and JSON.

    Args:
        scoreboard: output of `run_report`.
        out_dir: directory to write `scoreboard.md`/`metrics.json` into; created if
            missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(
        json.dumps(scoreboard.to_dict(orient='records'), indent=2)
    )
    (out_dir / 'scoreboard.md').write_text(_to_markdown_table(scoreboard))
