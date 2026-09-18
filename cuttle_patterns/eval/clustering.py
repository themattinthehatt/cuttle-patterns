"""Shared clustering preprocessing for the eval harness.

Standardizes embeddings of different dimensionality onto equal footing before k-means,
per `docs/eval_plan.md`'s "Clustering conventions": L2-normalize, then PCA (no
whitening).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from cuttle_patterns.cluster import run_kmeans

DEFAULT_N_COMPONENTS = 64
DEFAULT_N_CLUSTERS = 16
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def prepare_for_clustering(
    X: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> tuple[np.ndarray, float]:
    """L2-normalize then PCA-reduce embeddings for clustering.

    Args:
        X: embeddings, shape (n_frames, latent_dim).
        n_components: target PCA dimension; capped at `X`'s native dimension.

    Returns:
        (X_reduced, variance_retained): `X_reduced` has shape
        (n_frames, min(n_components, latent_dim)); `variance_retained` is the summed
        explained-variance ratio of the kept components.
    """
    X_norm = normalize(X)
    n_components = min(n_components, X_norm.shape[1])
    pca = PCA(n_components=n_components, whiten=False)
    X_reduced = pca.fit_transform(X_norm)
    variance_retained = float(pca.explained_variance_ratio_.sum())
    return X_reduced, variance_retained


def run_kmeans_multiseed(
    X: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[np.ndarray]:
    """Run k-means at several seeds, for mean/std reporting across seeds.

    Args:
        X: embeddings to cluster, shape (n_frames, dim) — typically
            `prepare_for_clustering`'s output.
        n_clusters: number of clusters (k).
        seeds: random seeds, one k-means run each.

    Returns:
        list of int label arrays, each shape (n_frames,), one per seed.
    """
    return [run_kmeans(X, n_clusters=n_clusters, random_state=seed) for seed in seeds]
