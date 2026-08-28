"""Tests for cuttle_patterns.cli.cmd_inscribe."""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.cli.cmd_inscribe import cmd_inscribe
from cuttle_patterns.cli.main import main


def _raise_file_not_found() -> None:
    raise FileNotFoundError('no config file found at ~/.cuttle-patterns/config.yaml')


def _blob_frame(height: int = 60, width: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 15), (100, 45), (100, 100, 100), -1)
    return frame


def _write_pose_csv(path: Path, n_frames: int, tail: tuple, neck: tuple) -> Path:
    columns = pd.MultiIndex.from_tuples(
        [
            ('m', 'neck', 'x'), ('m', 'neck', 'y'), ('m', 'neck', 'likelihood'),
            ('m', 'tail', 'x'), ('m', 'tail', 'y'), ('m', 'tail', 'likelihood'),
        ],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = [[neck[0], neck[1], 0.99, tail[0], tail[1], 0.99]] * n_frames
    pd.DataFrame(data, columns=columns).to_csv(path)
    return path


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        data_dir=None,
        results_dir=None,
        output_dir=None,
        video_path=None,
        skip_existing=False,
        pose_dir=None,
        pose_path=None,
        thresh=0,
        aspect=2.0,
        canonical_height=20,
        smoothing_window=9,
        smoothing_sigma=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdInscribe:
    """Test the function cmd_inscribe."""

    def test_cmd_inscribe_missing_config_and_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_inscribe.load_config', _raise_file_not_found,
        )
        args = _make_args()

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_inscribe(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_cmd_inscribe_with_overrides(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        make_custom_video(
            data_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.mp4',
            [_blob_frame(), _blob_frame(), _blob_frame()],
        )
        args = _make_args(data_dir=data_dir, results_dir=results_dir)

        # Act
        cmd_inscribe(args)

        # Assert
        output_dir = results_dir / 'rectangles'
        assert (output_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.mp4').exists()
        assert (output_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.csv').exists()
        out = capsys.readouterr().out
        assert 'processing' in out
        assert 'no pose predictions' in out

    def test_cmd_inscribe_uses_matching_pose_file(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange: a pose CSV at the default results_dir/pose/{video_name}.csv location
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        make_custom_video(
            data_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.mp4',
            [_blob_frame(), _blob_frame(), _blob_frame()],
        )
        pose_dir = results_dir / 'pose'
        pose_dir.mkdir(parents=True)
        _write_pose_csv(
            pose_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.csv', n_frames=3,
            tail=(25.0, 30.0), neck=(95.0, 30.0),
        )
        args = _make_args(data_dir=data_dir, results_dir=results_dir)

        # Act
        cmd_inscribe(args)

        # Assert
        output_dir = results_dir / 'rectangles'
        assert (output_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.mp4').exists()
        out = capsys.readouterr().out
        assert 'no pose predictions' not in out

    def test_cmd_inscribe_skip_existing(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange: output mp4/csv already present for the one video in data_dir
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        make_custom_video(
            data_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.mp4',
            [_blob_frame(), _blob_frame(), _blob_frame()],
        )
        output_dir = results_dir / 'rectangles'
        output_dir.mkdir(parents=True)
        video_out_path = output_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.mp4'
        csv_out_path = output_dir / 'Day1_Tank2_Cuttle1_Resident_Crop.csv'
        video_out_path.write_bytes(b'existing')
        csv_out_path.write_text('existing')
        args = _make_args(data_dir=data_dir, results_dir=results_dir, skip_existing=True)

        # Act
        cmd_inscribe(args)

        # Assert: neither file was touched, and inscribe never ran for this video
        assert video_out_path.read_bytes() == b'existing'
        assert csv_out_path.read_text() == 'existing'
        out = capsys.readouterr().out
        assert 'skipping' in out
        assert 'processing' not in out

    def test_cmd_inscribe_smoothing_flags_are_mutually_exclusive(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'cuttle', 'inscribe',
                '--smoothing-window', '9',
                '--smoothing-sigma', '2.0',
            ],
        )

        # Act & Assert: argparse rejects both flags together at parse time
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
