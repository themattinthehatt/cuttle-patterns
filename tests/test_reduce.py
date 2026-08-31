"""Tests for cuttle_patterns.reduce."""

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.reduce import build_umap_dataframe, hparams_to_str, run_umap


class TestRunUmap:
    """Test the function run_umap."""

    def test_run_umap_output_shape(self):
        # Arrange
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 5))

        # Act
        umap_xy = run_umap(X, n_neighbors=5, min_dist=0.1, metric='euclidean', random_state=0)

        # Assert
        assert umap_xy.shape == (30, 2)


class TestHparamsToStr:
    """Test the function hparams_to_str."""

    def test_hparams_to_str_formats_values(self):
        # Act
        result = hparams_to_str(n_neighbors=15, min_dist=0.1)

        # Assert
        assert result == 'nn15_md0.1'


class TestBuildUmapDataframe:
    """Test the function build_umap_dataframe."""

    def test_build_umap_dataframe_columns(self):
        # Arrange
        meta = pd.DataFrame({
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'],
            'day': [1],
            'tank': [2],
            'role': ['Resident'],
            'frame_number': [7],
        })
        umap_xy = np.array([[0.5, -0.5]])

        # Act
        df = build_umap_dataframe(meta, umap_xy)

        # Assert
        assert list(df.columns) == [
            'umap_x', 'umap_y', 'day', 'tank', 'role', 'frame_number', 'video_name',
        ]
        assert df['umap_x'].iloc[0] == pytest.approx(0.5)
        assert df['umap_y'].iloc[0] == pytest.approx(-0.5)
