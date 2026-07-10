"""Tests for cuttle_patterns.cli.cmd_overlay."""

import argparse
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

from cuttle_patterns.cli.cmd_overlay import cmd_overlay
from cuttle_patterns.preprocessing.align import align_video


def _raise_file_not_found() -> None:
    raise FileNotFoundError('no config file found at ~/.cuttle-patterns/config.yaml')


def _blob_frame(height: int = 60, width: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 15), (100, 45), (100, 100, 100), -1)
    return frame


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        data_dir=None,
        results_dir=None,
        output_dir=None,
        video_path=None,
        thresh=0,
        aspect=2.0,
        canonical_height=20,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdOverlay:
    """Test the function cmd_overlay."""

    def test_cmd_overlay_missing_config_and_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_overlay.load_config', _raise_file_not_found,
        )
        args = _make_args()

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_overlay(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_cmd_overlay_runs_inscribe_when_csv_missing(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange: no {video_name}.csv exists yet under results_dir/rectangles
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        make_custom_video(
            data_dir / 'session-01_cuttle-01.mp4',
            [_blob_frame(), _blob_frame(), _blob_frame()],
        )
        args = _make_args(data_dir=data_dir, results_dir=results_dir)

        # Act
        cmd_overlay(args)

        # Assert
        output_dir = results_dir / 'rectangles'
        out = capsys.readouterr().out
        assert 'not found, running inscribe' in out
        assert (output_dir / 'session-01_cuttle-01.csv').exists()
        assert (output_dir / 'session-01_cuttle-01_overlay.mp4').exists()

    def test_cmd_overlay_reuses_existing_csv(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange: pre-generate {video_name}.csv via align_video, before calling overlay
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        video_path = make_custom_video(
            data_dir / 'session-01_cuttle-01.mp4',
            [_blob_frame(), _blob_frame(), _blob_frame()],
        )
        output_dir = results_dir / 'rectangles'
        align_video(video_path, output_dir, canonical_height=20)
        args = _make_args(data_dir=data_dir, results_dir=results_dir)

        # Act
        cmd_overlay(args)

        # Assert
        out = capsys.readouterr().out
        assert 'not found, running inscribe' not in out
        assert (output_dir / 'session-01_cuttle-01_overlay.mp4').exists()
