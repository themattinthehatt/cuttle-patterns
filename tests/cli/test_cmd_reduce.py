"""Tests for cuttle_patterns.cli.cmd_reduce."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
) -> Path:
    latents_dir = (
        results_dir / 'beast_models' / model_name / 'image_predictions' / 'beast_frames'
        / 'latents' / 'Day1_Tank2_Cuttle1_Resident_Crop'
    )
    latents_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        np.save(latents_dir / f'img{i:08d}.npy', rng.normal(size=4).astype(np.float32))
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
