"""The five core eval metrics: pattern agreement vs. identity organization.

Each function is a pure function on arrays, so it's unit-testable on synthetic data
independent of the rest of the harness. This is deliberately a small starting set —
see `docs/eval_plan.md`'s "Metrics" section for the larger list deferred to v2.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

DEFAULT_KNN_K = 20
DEFAULT_KNN_SEARCH_MULTIPLIER = 10
DEFAULT_CV_SPLITS = 5


def ami_cluster_vs_class(cluster_labels: np.ndarray, class_labels: np.ndarray) -> float:
    """Adjusted mutual information between cluster assignment and classifier class.

    High values mean clusters agree with the classifier's coarse pattern-type proxy.

    Args:
        cluster_labels: k-means cluster id per frame, shape (n_frames,).
        class_labels: classifier `predicted_pattern` per frame, shape (n_frames,).

    Returns:
        AMI score in [0, 1] (can dip slightly negative for near-random labelings).
    """
    return float(adjusted_mutual_info_score(class_labels, cluster_labels))


def ami_cluster_vs_video_within_class(
    cluster_labels: np.ndarray,
    video_labels: np.ndarray,
    class_labels: np.ndarray,
) -> float:
    """AMI between cluster and video identity, computed within each classifier class.

    Pattern and video are correlated (some sessions are mostly one pattern), so this
    conditions on class before measuring identity structure, per the eval plan's
    "condition on pattern when measuring identity" principle.

    Args:
        cluster_labels: k-means cluster id per frame, shape (n_frames,).
        video_labels: `video_name` per frame, shape (n_frames,).
        class_labels: classifier `predicted_pattern` per frame, shape (n_frames,).

    Returns:
        frame-weighted average of per-class AMI(cluster, video); classes with fewer
        than two distinct videos are skipped (AMI is undefined for a single group).

    Raises:
        ValueError: if no classifier class has frames from at least two distinct
            videos.
    """
    class_labels = np.asarray(class_labels)
    cluster_labels = np.asarray(cluster_labels)
    video_labels = np.asarray(video_labels)

    scores = []
    weights = []
    for cls in np.unique(class_labels):
        mask = class_labels == cls
        if len(np.unique(video_labels[mask])) < 2:
            continue
        scores.append(adjusted_mutual_info_score(video_labels[mask], cluster_labels[mask]))
        weights.append(mask.sum())

    if not scores:
        raise ValueError('no classifier class has frames from at least two distinct videos')
    return float(np.average(scores, weights=weights))


def effective_videos_per_cluster(cluster_labels: np.ndarray, video_labels: np.ndarray) -> float:
    """Frame-weighted average effective number of videos per cluster.

    For each cluster, `exp(entropy(video distribution))` is the "effective" video
    count if frames were spread evenly across the videos present in that cluster (1.0
    if every frame in the cluster is the same video).

    Args:
        cluster_labels: k-means cluster id per frame, shape (n_frames,).
        video_labels: `video_name` per frame, shape (n_frames,).

    Returns:
        frame-weighted mean of the per-cluster effective-video-count.
    """
    frame = pd.DataFrame({'cluster': cluster_labels, 'video': video_labels})
    scores = []
    weights = []
    for _, group in frame.groupby('cluster'):
        counts = group['video'].value_counts(normalize=True).to_numpy()
        entropy = -np.sum(counts * np.log(counts))
        scores.append(np.exp(entropy))
        weights.append(len(group))
    return float(np.average(scores, weights=weights))


def cross_video_knn_accuracy(
    X: np.ndarray,
    class_labels: np.ndarray,
    video_labels: np.ndarray,
    k: int = DEFAULT_KNN_K,
    search_multiplier: int = DEFAULT_KNN_SEARCH_MULTIPLIER,
) -> float:
    """Majority-vote k-NN class accuracy, neighbors restricted to other videos.

    Neighbors are drawn from each frame's `k * search_multiplier` nearest points
    overall, then filtered down to at most `k` from a different video — a deliberate
    approximation (vs. an exact "k nearest excluding same video" search over the whole
    dataset) to keep this cheap. An identity-dominated embedding can legitimately have
    fewer than `k` cross-video candidates in that window (the vote just uses whatever
    it finds), or even none — a real, expected outcome for e.g. an MSPS-VAE `z_b`
    subspace, or a plain AE where video identity dominates the embedding. When that
    happens, this falls back to a single full-dataset search for that one frame only,
    rather than failing the whole metric on the exact embedders this harness exists to
    flag as identity-dominated.

    Args:
        X: embeddings, shape (n_frames, dim).
        class_labels: classifier `predicted_pattern` per frame, shape (n_frames,).
        video_labels: `video_name` per frame, shape (n_frames,).
        k: number of cross-video neighbors to vote over.
        search_multiplier: size of the initial (all-video) neighbor search, as a
            multiple of `k`.

    Returns:
        fraction of frames whose majority cross-video-neighbor class matches their own
        predicted class.

    Raises:
        ValueError: if some frame has zero cross-video neighbors anywhere in the
            dataset (i.e. its video is the only one present).
    """
    class_labels = np.asarray(class_labels)
    video_labels = np.asarray(video_labels)
    n = X.shape[0]

    search_k = min(n, k * search_multiplier)
    neighbors = NearestNeighbors(n_neighbors=search_k).fit(X)
    _, indices = neighbors.kneighbors(X)

    predictions = np.empty(n, dtype=class_labels.dtype)
    for i in range(n):
        candidates = indices[i]
        cross_video = candidates[video_labels[candidates] != video_labels[i]][:k]
        if len(cross_video) == 0:
            # local window is entirely the same video -- widen to the full dataset
            # for this one frame instead of failing the whole metric.
            _, wide_indices = neighbors.kneighbors(X[i : i + 1], n_neighbors=n)
            wide_candidates = wide_indices[0]
            cross_video = wide_candidates[video_labels[wide_candidates] != video_labels[i]][:k]
        if len(cross_video) == 0:
            raise ValueError(
                f'frame {i} (video {video_labels[i]!r}) has no cross-video neighbors '
                f'anywhere in the dataset -- that video may be the only one present'
            )
        neighbor_classes = class_labels[cross_video]
        values, counts = np.unique(neighbor_classes, return_counts=True)
        predictions[i] = values[np.argmax(counts)]

    return float(accuracy_score(class_labels, predictions))


def linear_probe_accuracy(
    X: np.ndarray,
    class_labels: np.ndarray,
    video_labels: np.ndarray,
    n_splits: int = DEFAULT_CV_SPLITS,
) -> float:
    """Video-grouped cross-validated linear-probe accuracy for classifier class.

    Args:
        X: embeddings, shape (n_frames, dim).
        class_labels: classifier `predicted_pattern` per frame, shape (n_frames,).
        video_labels: `video_name` per frame, shape (n_frames,) — the CV group, so no
            fold can be satisfied by recognizing the session.
        n_splits: number of `GroupKFold` folds.

    Returns:
        accuracy over the concatenated out-of-fold predictions.
    """
    class_labels = np.asarray(class_labels)
    video_labels = np.asarray(video_labels)

    group_kfold = GroupKFold(n_splits=n_splits)
    predictions = np.empty_like(class_labels)
    for train_idx, test_idx in group_kfold.split(X, class_labels, groups=video_labels):
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X[train_idx], class_labels[train_idx])
        predictions[test_idx] = classifier.predict(X[test_idx])

    return float(accuracy_score(class_labels, predictions))
