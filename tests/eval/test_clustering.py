"""Tests for eval.clustering."""

import numpy as np

from cuttle_patterns.eval.clustering import prepare_for_clustering, run_kmeans_multiseed


class TestPrepareForClustering:
    """Test the function prepare_for_clustering."""

    def test_output_shape_and_variance_in_range(self):
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 10))

        # Act
        X_reduced, variance_retained = prepare_for_clustering(X, n_components=5)

        # Assert
        assert X_reduced.shape == (50, 5)
        assert 0.0 <= variance_retained <= 1.0

    def test_caps_n_components_at_native_dim(self):
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))

        # Act
        X_reduced, _ = prepare_for_clustering(X, n_components=64)

        # Assert
        assert X_reduced.shape[1] == 3


class TestRunKmeansMultiseed:
    """Test the function run_kmeans_multiseed."""

    def test_returns_one_label_array_per_seed(self):
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))

        # Act
        labels_per_seed = run_kmeans_multiseed(X, n_clusters=3, seeds=(0, 1))

        # Assert
        assert len(labels_per_seed) == 2
        for labels in labels_per_seed:
            assert labels.shape == (30,)
            assert len(np.unique(labels)) <= 3
