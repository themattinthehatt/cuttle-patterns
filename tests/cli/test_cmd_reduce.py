"""Tests for cuttle_patterns.cli.cmd_reduce."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cuttle_patterns.cli.cmd_reduce import cmd_reduce


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        model_name='resnet-ae-v1',
        predictions_name='beast_frames',
        n_neighbors=2,
        min_dist=0.1,
        metric='euclidean',
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


class TestCmdReduce:
    """Test the function cmd_reduce."""

    def test_cmd_reduce_missing_latents(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_reduce(args)
        assert exc_info.value.code == 1
        assert 'does not exist' in capsys.readouterr().out

    def test_cmd_reduce_writes_parquet(self, tmp_path: Path):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_latents(results_dir, n_frames=5)
        args = _make_args(results_dir=results_dir, n_neighbors=2)

        # Act
        cmd_reduce(args)

        # Assert
        output_path = (
            results_dir / 'beast_models' / 'resnet-ae-v1' / 'reduce' / 'umap_nn2_md0.1.parquet'
        )
        assert output_path.exists()
        df = pd.read_parquet(output_path)
        assert len(df) == 5
        assert list(df.columns) == [
            'umap_x', 'umap_y', 'day', 'tank', 'role', 'frame_number', 'video_name',
        ]

    def test_cmd_reduce_overwrites_existing_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_latents(results_dir, n_frames=5)
        args = _make_args(results_dir=results_dir, n_neighbors=2)
        cmd_reduce(args)
        capsys.readouterr()

        # Act
        cmd_reduce(args)

        # Assert
        assert 'overwriting existing' in capsys.readouterr().out

    def test_cmd_reduce_msps_vae_writes_two_reductions(self, tmp_path: Path):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_latents(
            results_dir,
            model_name='msps-vae-v1',
            n_frames=6,
            model_class='msps_vae',
            num_latents_unsupervised=2,
        )
        args = _make_args(results_dir=results_dir, model_name='msps-vae-v1', n_neighbors=2)

        # Act
        cmd_reduce(args)

        # Assert
        reduce_dir = results_dir / 'beast_models' / 'msps-vae-v1' / 'reduce'
        unsupervised_path = reduce_dir / 'umap_nn2_md0.1_unsupervised.parquet'
        background_path = reduce_dir / 'umap_nn2_md0.1_background.parquet'
        assert unsupervised_path.exists()
        assert background_path.exists()
        assert len(pd.read_parquet(unsupervised_path)) == 6
        assert len(pd.read_parquet(background_path)) == 6
