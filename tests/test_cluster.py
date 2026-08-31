"""Tests for cuttle_patterns.cluster."""

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.cluster import build_cluster_dataframe, hparams_to_str, run_kmeans


class TestRunKmeans:
    """Test the function run_kmeans."""

    def test_run_kmeans_separates_distinct_blobs(self):
        # Arrange
        rng = np.random.default_rng(0)
        blob_a = rng.normal(loc=0.0, scale=0.01, size=(10, 3))
        blob_b = rng.normal(loc=100.0, scale=0.01, size=(10, 3))
        X = np.concatenate([blob_a, blob_b])

        # Act
        labels = run_kmeans(X, n_clusters=2, random_state=0)

        # Assert
        assert len(np.unique(labels)) == 2
        assert len(np.unique(labels[:10])) == 1
        assert len(np.unique(labels[10:])) == 1
        assert labels[0] != labels[-1]


class TestHparamsToStr:
    """Test the function hparams_to_str."""

    def test_hparams_to_str_formats_value(self):
        # Act
        result = hparams_to_str(n_clusters=10)

        # Assert
        assert result == 'k10'


class TestBuildClusterDataframe:
    """Test the function build_cluster_dataframe."""

    def test_build_cluster_dataframe_columns(self):
        # Arrange
        meta = pd.DataFrame({
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'],
            'day': [1],
            'tank': [2],
            'role': ['Resident'],
            'frame_number': [7],
        })
        labels = np.array([3])

        # Act
        df = build_cluster_dataframe(meta, labels)

        # Assert
        assert list(df.columns) == [
            'cluster', 'day', 'tank', 'role', 'frame_number', 'video_name',
        ]
        assert df['cluster'].iloc[0] == pytest.approx(3)
