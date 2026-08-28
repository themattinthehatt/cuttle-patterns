"""Tests for cuttle_patterns.cli.cmd_extract."""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.cli.cmd_extract import cmd_extract
from cuttle_patterns.preprocessing.align import CORNER_COLUMNS


def _raise_file_not_found() -> None:
    raise FileNotFoundError('no config file found at ~/.cuttle-patterns/config.yaml')


def _write_pose_csv(path: Path, n_frames: int, likelihood: float = 0.99) -> Path:
    columns = pd.MultiIndex.from_tuples(
        [
            ('m', 'neck', 'x'), ('m', 'neck', 'y'), ('m', 'neck', 'likelihood'),
            ('m', 'tail', 'x'), ('m', 'tail', 'y'), ('m', 'tail', 'likelihood'),
        ],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = [[0.0, 0.0, likelihood, 0.0, 0.0, likelihood]] * n_frames
    pd.DataFrame(data, columns=columns).to_csv(path)
    return path


def _write_rect_csv(
    path: Path, n_frames: int, long_edge: float = 50.0, short_edge: float = 10.0,
) -> Path:
    # a single uniform (non-degenerate) rectangle repeated for every frame; paired with
    # _write_pose_csv's degenerate tail==neck==(0,0) keypoints, so it never trips the
    # small-rectangle filter unless a test wants it to
    corners = {
        'corner_tl_x': 0.0, 'corner_tl_y': 0.0,
        'corner_tr_x': long_edge, 'corner_tr_y': 0.0,
        'corner_br_x': long_edge, 'corner_br_y': short_edge,
        'corner_bl_x': 0.0, 'corner_bl_y': short_edge,
    }
    df = pd.DataFrame([corners] * n_frames, columns=CORNER_COLUMNS)
    df.insert(0, 'frame_idx', range(n_frames))
    df['is_interpolated'] = False
    df.to_csv(path, index=False)
    return path


def _make_ramp_video(make_custom_video: Callable, path: Path, n_frames: int = 8) -> Path:
    intensities = [(idx * 37) % 256 for idx in range(n_frames)]
    frames = [np.full((32, 32, 3), val, dtype=np.uint8) for val in intensities]
    return make_custom_video(path, frames)


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        data_dir=None,
        results_dir=None,
        input_dir=None,
        output_dir=None,
        pose_dir=None,
        pose_path=None,
        video_path=None,
        skip_existing=False,
        frames_per_video=3,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestCmdExtract:
    """Test the function cmd_extract."""

    def test_cmd_extract_missing_config_and_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract.load_config', _raise_file_not_found,
        )
        args = _make_args()

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_cmd_extract_with_overrides(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        input_dir = results_dir / 'rectangles'
        pose_dir = results_dir / 'pose'
        data_dir.mkdir()
        input_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        stem = 'Day1_Tank2_Cuttle1_Resident_Crop'
        _make_ramp_video(make_custom_video, input_dir / f'{stem}.mp4', n_frames=8)
        _write_pose_csv(pose_dir / f'{stem}.csv', n_frames=8)
        _write_rect_csv(input_dir / f'{stem}.csv', n_frames=8)
        (data_dir / 'Day1_Tank2_Cuttle1_Resident_black_frames.txt').write_text('3\n')

        args = _make_args(data_dir=data_dir, results_dir=results_dir, pose_dir=pose_dir)

        # Act
        cmd_extract(args)

        # Assert
        save_dir = results_dir / 'beast_frames' / stem
        assert (save_dir / 'selected_frames.csv').exists()
        manifest_path = results_dir / 'manifests' / 'extract.parquet'
        assert manifest_path.exists()
        manifest = pd.read_parquet(manifest_path)
        assert list(manifest.columns) == ['session_id', 'fish_id', 'frame_idx', 'image_path']
        assert (manifest['session_id'] == 'Day1_Tank2').all()
        assert (manifest['fish_id'] == 'Cuttle1_Resident').all()
        # frame 3 was flagged blank, so it and both its neighbors (2, 4) never show up
        # as selected anchors
        assert not set(manifest['frame_idx'].to_numpy()) & {2, 3, 4}
        out = capsys.readouterr().out
        assert 'processing' in out
        assert 'Manifest written to' in out

    def test_cmd_extract_excludes_overlay_files(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        input_dir = results_dir / 'rectangles'
        pose_dir = results_dir / 'pose'
        data_dir.mkdir()
        input_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        stem = 'Day1_Tank2_Cuttle1_Resident_Crop'
        _make_ramp_video(make_custom_video, input_dir / f'{stem}.mp4', n_frames=8)
        _make_ramp_video(make_custom_video, input_dir / f'{stem}_overlay.mp4', n_frames=8)
        _write_pose_csv(pose_dir / f'{stem}.csv', n_frames=8)
        _write_rect_csv(input_dir / f'{stem}.csv', n_frames=8)

        args = _make_args(data_dir=data_dir, results_dir=results_dir, pose_dir=pose_dir)

        # Act
        cmd_extract(args)

        # Assert: only the non-overlay video was processed
        assert (results_dir / 'beast_frames' / stem).exists()
        assert not (results_dir / 'beast_frames' / f'{stem}_overlay').exists()
        assert capsys.readouterr().out.count('processing') == 1

    def test_cmd_extract_skip_existing(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        input_dir = results_dir / 'rectangles'
        pose_dir = results_dir / 'pose'
        data_dir.mkdir()
        input_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        stem = 'Day1_Tank2_Cuttle1_Resident_Crop'
        _make_ramp_video(make_custom_video, input_dir / f'{stem}.mp4', n_frames=8)
        _write_pose_csv(pose_dir / f'{stem}.csv', n_frames=8)

        save_dir = results_dir / 'beast_frames' / stem
        save_dir.mkdir(parents=True)
        (save_dir / 'selected_frames.csv').write_text('existing')

        args = _make_args(
            data_dir=data_dir, results_dir=results_dir, pose_dir=pose_dir, skip_existing=True,
        )

        # Act
        cmd_extract(args)

        # Assert
        assert (save_dir / 'selected_frames.csv').read_text() == 'existing'
        out = capsys.readouterr().out
        assert 'skipping' in out
        assert 'processing' not in out

    def test_cmd_extract_skips_video_missing_pose_file(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        input_dir = results_dir / 'rectangles'
        pose_dir = results_dir / 'pose'
        data_dir.mkdir()
        input_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        stem = 'Day1_Tank2_Cuttle1_Resident_Crop'
        _make_ramp_video(make_custom_video, input_dir / f'{stem}.mp4', n_frames=8)

        args = _make_args(data_dir=data_dir, results_dir=results_dir, pose_dir=pose_dir)

        # Act
        cmd_extract(args)

        # Assert
        assert not (results_dir / 'beast_frames' / stem).exists()
        assert not (results_dir / 'manifests' / 'extract.parquet').exists()
        out = capsys.readouterr().out
        assert 'no pose predictions' in out
        assert 'No frames extracted.' in out

    def test_cmd_extract_missing_blank_frames_file_warns_and_continues(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange: no black_frames.txt in data_dir at all
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        input_dir = results_dir / 'rectangles'
        pose_dir = results_dir / 'pose'
        data_dir.mkdir()
        input_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        stem = 'Day1_Tank2_Cuttle1_Resident_Crop'
        _make_ramp_video(make_custom_video, input_dir / f'{stem}.mp4', n_frames=8)
        _write_pose_csv(pose_dir / f'{stem}.csv', n_frames=8)
        _write_rect_csv(input_dir / f'{stem}.csv', n_frames=8)

        args = _make_args(data_dir=data_dir, results_dir=results_dir, pose_dir=pose_dir)

        # Act
        cmd_extract(args)

        # Assert
        assert (results_dir / 'beast_frames' / stem / 'selected_frames.csv').exists()
        out = capsys.readouterr().out
        assert 'no blank-frames file found' in out

    def test_cmd_extract_skips_video_missing_rect_csv(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange: pose predictions exist, but no {video_name}.csv in input_dir
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        input_dir = results_dir / 'rectangles'
        pose_dir = results_dir / 'pose'
        data_dir.mkdir()
        input_dir.mkdir(parents=True)
        pose_dir.mkdir(parents=True)

        stem = 'Day1_Tank2_Cuttle1_Resident_Crop'
        _make_ramp_video(make_custom_video, input_dir / f'{stem}.mp4', n_frames=8)
        _write_pose_csv(pose_dir / f'{stem}.csv', n_frames=8)

        args = _make_args(data_dir=data_dir, results_dir=results_dir, pose_dir=pose_dir)

        # Act
        cmd_extract(args)

        # Assert
        assert not (results_dir / 'beast_frames' / stem).exists()
        assert not (results_dir / 'manifests' / 'extract.parquet').exists()
        out = capsys.readouterr().out
        assert 'no rectangle geometry' in out
        assert 'No frames extracted.' in out
