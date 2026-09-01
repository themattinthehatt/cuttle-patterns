"""Tests for cuttle_patterns.embeddings."""

from pathlib import Path

import numpy as np
import pytest

from cuttle_patterns.embeddings import (
    load_latents,
    parse_video_name,
    select_cluster_latents,
    split_latent_spaces,
)


def _write_latent(path: Path, vector: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vector)


class TestParseVideoName:
    """Test the function parse_video_name."""

    def test_parse_video_name_success(self):
        # Act
        result = parse_video_name('Day1_Tank2_Cuttle1_Resident_Crop')

        # Assert
        assert result == {'day': 1, 'tank': 2, 'role': 'Resident'}

    def test_parse_video_name_lowercase_crop(self):
        # Act
        result = parse_video_name('Day3_Tank10_Cuttle2_Intruder_crop')

        # Assert
        assert result == {'day': 3, 'tank': 10, 'role': 'Intruder'}

    def test_parse_video_name_malformed(self):
        # Act & Assert
        with pytest.raises(ValueError, match='does not match expected pattern'):
            parse_video_name('not_a_video_name')


class TestLoadLatents:
    """Test the function load_latents."""

    def test_load_latents_success(self, tmp_path: Path):
        # Arrange
        latents_dir = tmp_path / 'latents'
        _write_latent(
            latents_dir / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000002.npy',
            np.array([1.0, 2.0], dtype=np.float32),
        )
        _write_latent(
            latents_dir / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000001.npy',
            np.array([3.0, 4.0], dtype=np.float32),
        )
        _write_latent(
            latents_dir / 'Day1_Tank2_Cuttle2_Intruder_Crop' / 'img00000005.npy',
            np.array([5.0, 6.0], dtype=np.float32),
        )

        # Act
        X, meta = load_latents(latents_dir)

        # Assert
        assert X.shape == (3, 2)
        assert list(meta.columns) == ['video_name', 'day', 'tank', 'role', 'frame_number']
        assert meta['frame_number'].tolist() == [1, 2, 5]
        assert meta['video_name'].tolist() == [
            'Day1_Tank2_Cuttle1_Resident_Crop',
            'Day1_Tank2_Cuttle1_Resident_Crop',
            'Day1_Tank2_Cuttle2_Intruder_Crop',
        ]
        assert meta['role'].tolist() == ['Resident', 'Resident', 'Intruder']
        np.testing.assert_array_equal(X[0], [3.0, 4.0])
        np.testing.assert_array_equal(X[1], [1.0, 2.0])
        np.testing.assert_array_equal(X[2], [5.0, 6.0])

    def test_load_latents_missing_dir(self, tmp_path: Path):
        # Arrange
        latents_dir = tmp_path / 'does_not_exist'

        # Act & Assert
        with pytest.raises(FileNotFoundError, match='does not exist'):
            load_latents(latents_dir)

    def test_load_latents_empty_dir(self, tmp_path: Path):
        # Arrange
        latents_dir = tmp_path / 'latents'
        latents_dir.mkdir()

        # Act & Assert
        with pytest.raises(ValueError, match='no .npy latent files found'):
            load_latents(latents_dir)


class TestSplitLatentSpaces:
    """Test the function split_latent_spaces."""

    def test_split_latent_spaces_non_msps_vae_returns_full_array(self, tmp_path: Path):
        # Arrange
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        (model_dir / 'config.yaml').write_text('model:\n  model_class: resnet_ae\n')
        X = np.arange(12).reshape(3, 4)

        # Act
        subspaces = split_latent_spaces(X, model_dir)

        # Assert
        assert list(subspaces.keys()) == ['all']
        np.testing.assert_array_equal(subspaces['all'], X)

    def test_split_latent_spaces_msps_vae_splits_columns(self, tmp_path: Path):
        # Arrange
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        (model_dir / 'config.yaml').write_text(
            'model:\n'
            '  model_class: msps_vae\n'
            '  model_params:\n'
            '    num_latents_unsupervised: 2\n'
        )
        X = np.arange(12).reshape(3, 4)

        # Act
        subspaces = split_latent_spaces(X, model_dir)

        # Assert
        assert set(subspaces.keys()) == {'unsupervised', 'background'}
        np.testing.assert_array_equal(subspaces['unsupervised'], X[:, :2])
        np.testing.assert_array_equal(subspaces['background'], X[:, 2:])

    def test_split_latent_spaces_missing_config(self, tmp_path: Path):
        # Arrange
        model_dir = tmp_path / 'model'
        model_dir.mkdir()
        X = np.arange(12).reshape(3, 4)

        # Act & Assert
        with pytest.raises(FileNotFoundError, match='no config.yaml found'):
            split_latent_spaces(X, model_dir)


class TestSelectClusterLatents:
    """Test the function select_cluster_latents."""

    def test_select_cluster_latents_single_space(self):
        # Arrange
        X = np.arange(6).reshape(3, 2)

        # Act
        result = select_cluster_latents({'all': X})

        # Assert
        np.testing.assert_array_equal(result, X)

    def test_select_cluster_latents_two_spaces_picks_unsupervised(self):
        # Arrange
        X_u = np.arange(6).reshape(3, 2)
        X_b = np.arange(6, 12).reshape(3, 2)

        # Act
        result = select_cluster_latents({'unsupervised': X_u, 'background': X_b})

        # Assert
        np.testing.assert_array_equal(result, X_u)
