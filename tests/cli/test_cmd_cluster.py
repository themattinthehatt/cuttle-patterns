"""Tests for cuttle_patterns.cli.cmd_cluster."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cuttle_patterns.cli.cmd_cluster import cmd_cluster


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        model_name='resnet-ae-v1',
        predictions_name='beast_frames',
        method='kmeans',
        n_clusters=2,
        random_state=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_latents(
    results_dir: Path,
    model_name: str = 'resnet-ae-v1',
    n_frames: int = 5,
    model_class: str = 'resnet_ae',
    num_latents_unsupervised: int | None = None,
) -> Path:
    model_dir = results_dir / 'beast_models' / model_name
    latents_dir = (
        model_dir / 'image_predictions' / 'beast_frames' / 'latents'
        / 'Day1_Tank2_Cuttle1_Resident_Crop'
    )
    latents_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        np.save(latents_dir / f'img{i:08d}.npy', rng.normal(size=4).astype(np.float32))

    config = {'model': {'model_class': model_class}}
    if model_class == 'msps_vae':
        config['model']['model_params'] = {
            'num_latents_unsupervised': num_latents_unsupervised,
        }
    with (model_dir / 'config.yaml').open('w') as f:
        yaml.safe_dump(config, f)

    return latents_dir


class TestCmdCluster:
    """Test the function cmd_cluster."""

    def test_cmd_cluster_missing_latents(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_cluster(args)
        assert exc_info.value.code == 1
        assert 'does not exist' in capsys.readouterr().out

    def test_cmd_cluster_writes_parquet(self, tmp_path: Path):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_latents(results_dir, n_frames=5)
        args = _make_args(results_dir=results_dir, n_clusters=2)

        # Act
        cmd_cluster(args)

        # Assert
        output_path = (
            results_dir / 'beast_models' / 'resnet-ae-v1' / 'clusters' / 'kmeans_k2.parquet'
        )
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 5
        assert list(df.columns) == [
            'cluster', 'day', 'tank', 'role', 'frame_number', 'video_name',
        ]
        assert df['cluster'].nunique() <= 2

    def test_cmd_cluster_overwrites_existing_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_latents(results_dir, n_frames=5)
        args = _make_args(results_dir=results_dir, n_clusters=2)
        cmd_cluster(args)
        capsys.readouterr()

        # Act
        cmd_cluster(args)

        # Assert
        assert 'overwriting existing' in capsys.readouterr().out

    def test_cmd_cluster_msps_vae_clusters_unsupervised_only(self, tmp_path: Path):
        # Arrange: dims 0-1 (unsupervised) group frames {0,1} vs {2,3}; dims 2-3
        # (background), at a much larger scale, instead group {0,2} vs {1,3} — so
        # clustering on the full vector would recover the background grouping instead
        results_dir = tmp_path / 'results'
        latents_dir = (
            results_dir / 'beast_models' / 'msps-vae-v1' / 'image_predictions'
            / 'beast_frames' / 'latents' / 'Day1_Tank2_Cuttle1_Resident_Crop'
        )
        latents_dir.mkdir(parents=True)
        vectors = [
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.1, 0.1, 1000.0, 1000.0], dtype=np.float32),
            np.array([10.0, 10.0, 0.1, 0.1], dtype=np.float32),
            np.array([10.1, 10.1, 1000.1, 1000.1], dtype=np.float32),
        ]
        for i, vector in enumerate(vectors):
            np.save(latents_dir / f'img{i:08d}.npy', vector)

        model_dir = results_dir / 'beast_models' / 'msps-vae-v1'
        config = {
            'model': {
                'model_class': 'msps_vae',
                'model_params': {'num_latents_unsupervised': 2},
            },
        }
        with (model_dir / 'config.yaml').open('w') as f:
            yaml.safe_dump(config, f)

        args = _make_args(results_dir=results_dir, model_name='msps-vae-v1', n_clusters=2)

        # Act
        cmd_cluster(args)

        # Assert
        output_path = model_dir / 'clusters' / 'kmeans_k2.parquet'
        df = pd.read_parquet(output_path).sort_values('frame_number')
        labels = df['cluster'].tolist()
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
