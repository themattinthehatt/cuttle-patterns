"""Assign a discrete cluster label to every frame (Phase 6).

Clusters the raw per-frame BEAST embeddings `cuttle_patterns.embeddings.load_latents`
returns — not the 2D UMAP projection from `cuttle_patterns.reduce` — since it's more
principled to cluster in the space the embedding model actually produces.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_RANDOM_STATE = 42

CLUSTER_OUTPUT_COLUMNS = ['cluster', 'day', 'tank', 'role', 'frame_number', 'video_name']


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Cluster embeddings with k-means.

    Args:
        X: embeddings, shape (n_frames, latent_dim).
        n_clusters: number of clusters.
        random_state: KMeans' `random_state`, for reproducibility.

    Returns:
        int array of shape (n_frames,), a cluster label per row of X.
    """
    return KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X)


def hparams_to_str(n_clusters: int) -> str:
    """Build the `{hparams}` filename suffix for a given `n_clusters`.

    Args:
        n_clusters: number of clusters.

    Returns:
        string like `k10`.
    """
    return f'k{n_clusters}'


def build_cluster_dataframe(meta: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels to per-frame metadata.

    Args:
        meta: per-frame metadata from `cuttle_patterns.embeddings.load_latents`, with
            columns `video_name`, `day`, `tank`, `role`, `frame_number`.
        labels: array of shape (len(meta),), row-aligned with meta.

    Returns:
        DataFrame with columns `cluster`, `day`, `tank`, `role`, `frame_number`,
        `video_name`.
    """
    out = meta.copy()
    out['cluster'] = labels
    return out[CLUSTER_OUTPUT_COLUMNS]
