"""Tests for cuttle_patterns.dashboard.data."""

from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.dashboard import data


def _make_reduce_df(n_frames: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        'umap_x': [float(i) for i in range(n_frames)],
        'umap_y': [float(-i) for i in range(n_frames)],
        'day': [1] * n_frames,
        'tank': [2] * n_frames,
        'role': ['Resident'] * n_frames,
        'frame_number': list(range(n_frames)),
        'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'] * n_frames,
    })


def _write_parquet(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


class TestListModelNames:
    """Test the function list_model_names."""

    def test_list_model_names_returns_sorted_subdirs(self, tmp_path: Path):
        # Arrange
        models_dir = tmp_path / 'beast_models'
        (models_dir / 'b_model').mkdir(parents=True)
        (models_dir / 'a_model').mkdir(parents=True)
        (models_dir / 'not_a_dir.txt').parent.mkdir(parents=True, exist_ok=True)
        (models_dir / 'not_a_dir.txt').touch()

        # Act
        result = data.list_model_names(tmp_path)

        # Assert
        assert result == ['a_model', 'b_model']

    def test_list_model_names_missing_dir(self, tmp_path: Path):
        # Act & Assert
        assert data.list_model_names(tmp_path) == []


class TestListReducePaths:
    """Test the function list_reduce_paths."""

    def test_list_reduce_paths_returns_sorted_parquet_files(self, tmp_path: Path):
        # Arrange
        reduce_dir = tmp_path / 'reduce'
        _write_parquet(reduce_dir / 'umap_nn15_md0.1.parquet', _make_reduce_df())
        _write_parquet(reduce_dir / 'umap_nn5_md0.5.parquet', _make_reduce_df())

        # Act
        result = data.list_reduce_paths(tmp_path)

        # Assert
        assert [p.name for p in result] == ['umap_nn15_md0.1.parquet', 'umap_nn5_md0.5.parquet']

    def test_list_reduce_paths_missing_dir(self, tmp_path: Path):
        # Act & Assert
        assert data.list_reduce_paths(tmp_path) == []


class TestListClusterPaths:
    """Test the function list_cluster_paths."""

    def test_list_cluster_paths_returns_sorted_parquet_files(self, tmp_path: Path):
        # Arrange
        clusters_dir = tmp_path / 'clusters'
        _write_parquet(clusters_dir / 'kmeans_k10.parquet', _make_reduce_df())
        _write_parquet(clusters_dir / 'kmeans_k5.parquet', _make_reduce_df())

        # Act
        result = data.list_cluster_paths(tmp_path)

        # Assert
        assert [p.name for p in result] == ['kmeans_k10.parquet', 'kmeans_k5.parquet']

    def test_list_cluster_paths_missing_dir(self, tmp_path: Path):
        # Act & Assert
        assert data.list_cluster_paths(tmp_path) == []


class TestBuildImageRelpath:
    """Test the function build_image_relpath."""

    def test_build_image_relpath_zero_pads_frame_number(self):
        # Act
        result = data.build_image_relpath('Day1_Tank2_Cuttle1_Resident_Crop', 42)

        # Assert
        assert result == 'Day1_Tank2_Cuttle1_Resident_Crop/img00000042.png'


class TestLoadReduceDataframe:
    """Test the function load_reduce_dataframe."""

    def test_load_reduce_dataframe_adds_image_relpath(self, tmp_path: Path):
        # Arrange
        reduce_path = _write_parquet(tmp_path / 'umap_nn15_md0.1.parquet', _make_reduce_df(3))

        # Act
        df = data.load_reduce_dataframe(reduce_path)

        # Assert
        assert list(df['image_relpath']) == [
            'Day1_Tank2_Cuttle1_Resident_Crop/img00000000.png',
            'Day1_Tank2_Cuttle1_Resident_Crop/img00000001.png',
            'Day1_Tank2_Cuttle1_Resident_Crop/img00000002.png',
        ]


class TestAttachClusterColumn:
    """Test the function attach_cluster_column."""

    def test_attach_cluster_column_merges_and_renames(self, tmp_path: Path):
        # Arrange
        df = _make_reduce_df(3)
        cluster_df = pd.DataFrame({
            'cluster': [0, 1, 0],
            'day': [1, 1, 1],
            'tank': [2, 2, 2],
            'role': ['Resident'] * 3,
            'frame_number': [0, 1, 2],
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'] * 3,
        })
        cluster_path = _write_parquet(tmp_path / 'kmeans_k10.parquet', cluster_df)

        # Act
        result = data.attach_cluster_column(df, cluster_path)

        # Assert
        assert 'cluster' not in result.columns
        assert list(result['kmeans_k10']) == [0, 1, 0]
        assert list(df.columns) == list(_make_reduce_df(3).columns)  # original untouched

    def test_attach_cluster_column_raises_on_missing_from_cluster(self, tmp_path: Path):
        # Arrange
        df = _make_reduce_df(3)
        cluster_df = pd.DataFrame({
            'cluster': [0, 1],
            'day': [1, 1],
            'tank': [2, 2],
            'role': ['Resident'] * 2,
            'frame_number': [0, 1],
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'] * 2,
        })
        cluster_path = _write_parquet(tmp_path / 'kmeans_k10.parquet', cluster_df)

        # Act & Assert
        with pytest.raises(ValueError, match='does not match'):
            data.attach_cluster_column(df, cluster_path)

    def test_attach_cluster_column_raises_on_extra_in_cluster(self, tmp_path: Path):
        # Arrange
        df = _make_reduce_df(2)
        cluster_df = pd.DataFrame({
            'cluster': [0, 1, 0],
            'day': [1, 1, 1],
            'tank': [2, 2, 2],
            'role': ['Resident'] * 3,
            'frame_number': [0, 1, 2],
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'] * 3,
        })
        cluster_path = _write_parquet(tmp_path / 'kmeans_k10.parquet', cluster_df)

        # Act & Assert
        with pytest.raises(ValueError, match='does not match'):
            data.attach_cluster_column(df, cluster_path)


class TestColorableColumns:
    """Test the function colorable_columns."""

    def test_colorable_columns_excludes_umap_and_image_path(self):
        # Arrange
        df = _make_reduce_df(2)
        df['image_relpath'] = ['a', 'b']
        df['kmeans_k10'] = [0, 1]

        # Act
        result = data.colorable_columns(df)

        # Assert
        assert 'umap_x' not in result
        assert 'umap_y' not in result
        assert 'image_relpath' not in result
        assert 'kmeans_k10' in result
        assert 'role' in result


class TestIsCategoricalColumn:
    """Test the function is_categorical_column."""

    def test_is_categorical_column_true_for_string_dtype(self):
        # Arrange
        df = pd.DataFrame({'role': ['Resident', 'Intruder']})

        # Act & Assert
        assert data.is_categorical_column(df, 'role') is True

    def test_is_categorical_column_true_for_low_cardinality_numeric(self):
        # Arrange
        df = pd.DataFrame({'cluster': [0, 1, 2, 0, 1]})

        # Act & Assert
        assert data.is_categorical_column(df, 'cluster', max_categories=20) is True

    def test_is_categorical_column_false_for_high_cardinality_numeric(self):
        # Arrange
        df = pd.DataFrame({'frame_number': list(range(50))})

        # Act & Assert
        assert data.is_categorical_column(df, 'frame_number', max_categories=20) is False
