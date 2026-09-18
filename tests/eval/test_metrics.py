"""Tests for eval.metrics."""

import numpy as np
import pytest

from cuttle_patterns.eval.metrics import (
    ami_cluster_vs_class,
    ami_cluster_vs_video_within_class,
    cross_video_knn_accuracy,
    effective_videos_per_cluster,
    linear_probe_accuracy,
)


class TestAmiClusterVsClass:
    """Test the function ami_cluster_vs_class."""

    def test_perfect_agreement_scores_near_one(self):
        # Arrange
        class_labels = np.array(['a', 'a', 'b', 'b'] * 5)
        cluster_labels = class_labels.copy()

        # Act
        score = ami_cluster_vs_class(cluster_labels, class_labels)

        # Assert
        assert score > 0.99

    def test_random_labels_score_near_zero(self):
        # Arrange
        rng = np.random.default_rng(0)
        class_labels = rng.integers(0, 4, size=200)
        cluster_labels = rng.integers(0, 4, size=200)

        # Act
        score = ami_cluster_vs_class(cluster_labels, class_labels)

        # Assert
        assert score < 0.1


class TestAmiClusterVsVideoWithinClass:
    """Test the function ami_cluster_vs_video_within_class."""

    def test_identity_dominated_clusters_score_high(self):
        # Arrange: two classes, each split across two videos; clusters track video.
        class_labels = np.array(['a'] * 20 + ['b'] * 20)
        video_labels = np.array((['v1'] * 10 + ['v2'] * 10) * 2)
        cluster_labels = video_labels.copy()

        # Act
        score = ami_cluster_vs_video_within_class(cluster_labels, video_labels, class_labels)

        # Assert
        assert score > 0.99

    def test_raises_when_no_class_has_two_videos(self):
        # Arrange
        class_labels = np.array(['a'] * 10)
        video_labels = np.array(['v1'] * 10)
        cluster_labels = np.zeros(10, dtype=int)

        # Act / Assert
        with pytest.raises(ValueError, match='at least two distinct videos'):
            ami_cluster_vs_video_within_class(cluster_labels, video_labels, class_labels)


class TestEffectiveVideosPerCluster:
    """Test the function effective_videos_per_cluster."""

    def test_single_video_clusters_score_one(self):
        # Arrange
        cluster_labels = np.array([0, 0, 1, 1])
        video_labels = np.array(['v1', 'v1', 'v2', 'v2'])

        # Act
        score = effective_videos_per_cluster(cluster_labels, video_labels)

        # Assert
        assert score == pytest.approx(1.0)

    def test_evenly_mixed_cluster_scores_above_one(self):
        # Arrange
        cluster_labels = np.zeros(4, dtype=int)
        video_labels = np.array(['v1', 'v1', 'v2', 'v2'])

        # Act
        score = effective_videos_per_cluster(cluster_labels, video_labels)

        # Assert
        assert score == pytest.approx(2.0)


class TestCrossVideoKnnAccuracy:
    """Test the function cross_video_knn_accuracy."""

    def test_recovers_class_from_cross_video_structure(self):
        # Arrange: two tight class blobs, each split across two videos.
        rng = np.random.default_rng(0)
        blob_a = rng.normal(loc=0.0, scale=0.01, size=(20, 2))
        blob_b = rng.normal(loc=10.0, scale=0.01, size=(20, 2))
        X = np.concatenate([blob_a, blob_b])
        class_labels = np.array(['a'] * 20 + ['b'] * 20)
        video_labels = np.array((['v1'] * 10 + ['v2'] * 10) * 2)

        # Act
        accuracy = cross_video_knn_accuracy(X, class_labels, video_labels, k=5)

        # Assert
        assert accuracy == pytest.approx(1.0)

    def test_falls_back_to_full_dataset_when_local_window_is_same_video(self):
        # Arrange: 9 tightly clustered v1 points, 1 far-away v2 point. A narrow local
        # window (search_multiplier=1, k=5 -> search_k=5) finds no cross-video
        # neighbor for the v1 points, forcing the full-dataset fallback.
        rng = np.random.default_rng(0)
        cluster_v1 = rng.normal(loc=0.0, scale=0.01, size=(9, 2))
        point_v2 = np.array([[100.0, 100.0]])
        X = np.concatenate([cluster_v1, point_v2])
        class_labels = np.array(['a'] * 10)
        video_labels = np.array(['v1'] * 9 + ['v2'])

        # Act
        accuracy = cross_video_knn_accuracy(
            X, class_labels, video_labels, k=5, search_multiplier=1,
        )

        # Assert
        assert 0.0 <= accuracy <= 1.0

    def test_raises_when_only_one_video_present(self):
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 2))
        class_labels = np.array(['a'] * 10)
        video_labels = np.array(['v1'] * 10)

        # Act / Assert
        with pytest.raises(ValueError, match='no cross-video neighbors'):
            cross_video_knn_accuracy(X, class_labels, video_labels, k=5, search_multiplier=1)


class TestLinearProbeAccuracy:
    """Test the function linear_probe_accuracy."""

    def test_separable_classes_score_high(self):
        # Arrange: two well-separated blobs, 3 videos per class.
        rng = np.random.default_rng(0)
        blob_a = rng.normal(loc=0.0, scale=0.1, size=(30, 2))
        blob_b = rng.normal(loc=10.0, scale=0.1, size=(30, 2))
        X = np.concatenate([blob_a, blob_b])
        class_labels = np.array(['a'] * 30 + ['b'] * 30)
        video_labels = np.array(
            [f'v{i % 3}' for i in range(30)] + [f'v{3 + i % 3}' for i in range(30)]
        )

        # Act
        accuracy = linear_probe_accuracy(X, class_labels, video_labels, n_splits=3)

        # Assert
        assert accuracy > 0.9
