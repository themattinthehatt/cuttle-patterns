"""Tests for cuttle_patterns.cli.cmd_clusterview."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.cli.cmd_clusterview import cmd_clusterview


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        model_name='resnet-ae-v1',
        cluster_run='kmeans_k2',
        n_frames=2,
        n_cols=2,
        seed=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_cluster_run(
    results_dir: Path,
    model_name: str = 'resnet-ae-v1',
    cluster_run: str = 'kmeans_k2',
    with_frame_images: bool = True,
) -> Path:
    rows = [
        {'cluster': 0, 'video_name': 'vidA', 'frame_number': 0},
        {'cluster': 0, 'video_name': 'vidA', 'frame_number': 1},
        {'cluster': 1, 'video_name': 'vidB', 'frame_number': 0},
    ]
    df = pd.DataFrame(rows)

    cluster_dir = results_dir / 'beast_models' / model_name / 'clusters'
    cluster_dir.mkdir(parents=True)
    cluster_path = cluster_dir / f'{cluster_run}.parquet'
    df.to_parquet(cluster_path, index=False)

    if with_frame_images:
        for row in rows:
            frame_dir = results_dir / 'beast_frames' / row['video_name']
            frame_dir.mkdir(parents=True, exist_ok=True)
            image_path = frame_dir / f"img{row['frame_number']:08d}.png"
            cv2.imwrite(str(image_path), np.zeros((10, 20, 3), dtype=np.uint8))

    return cluster_path


class TestCmdClusterview:
    """Test the function cmd_clusterview."""

    def test_cmd_clusterview_missing_cluster_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_clusterview(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_cmd_clusterview_writes_one_png_per_cluster(self, tmp_path: Path):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_cluster_run(results_dir)
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_clusterview(args)

        # Assert
        output_dir = results_dir / 'beast_models' / 'resnet-ae-v1' / 'clusters' / 'kmeans_k2'
        assert sorted(p.name for p in output_dir.glob('*.png')) == [
            'cluster_00.png', 'cluster_01.png',
        ]

    def test_cmd_clusterview_uses_results_dir_override_over_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange: point config at a directory with no cluster run, but pass a
        # --results-dir override that does have one; the override should win
        config_results_dir = tmp_path / 'from_config'
        override_results_dir = tmp_path / 'from_override'
        _make_cluster_run(override_results_dir)

        def _fail_if_called():
            raise AssertionError('load_config should not be called when --results-dir is set')

        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_clusterview.load_config',
            lambda: _fail_if_called(),
        )
        args = _make_args(results_dir=override_results_dir)

        # Act
        cmd_clusterview(args)

        # Assert
        output_dir = (
            override_results_dir / 'beast_models' / 'resnet-ae-v1' / 'clusters' / 'kmeans_k2'
        )
        assert len(list(output_dir.glob('*.png'))) == 2
        assert not config_results_dir.exists()
