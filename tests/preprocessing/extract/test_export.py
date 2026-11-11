"""Tests for cuttle_patterns.preprocessing.extract.export."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.extract.export import (
    build_extraction_manifest,
    export_video_frames,
    frame_filename,
    read_frames_sequential,
    run_export,
)

STEM = 'Day2_Tank1_Cuttle1_Resident_Crop'


class TestFrameFilename:
    """Test the function frame_filename."""

    def test_frame_filename_zero_pads_to_eight_digits(self):
        assert frame_filename(1234) == 'img00001234.png'


class TestReadFramesSequential:
    """Test the function read_frames_sequential."""

    def test_read_frames_sequential_returns_requested_frames(
        self, tmp_path: Path, make_custom_video: Callable, constant_frames: Callable,
    ):
        # Arrange: brightness identifies the frame
        video = make_custom_video(tmp_path / 'v.mp4', constant_frames([10, 60, 110, 160, 210]))

        # Act
        frames = read_frames_sequential(video, np.array([3, 1]))

        # Assert
        assert sorted(frames) == [1, 3]
        assert frames[1].mean() == pytest.approx(60, abs=6)
        assert frames[3].mean() == pytest.approx(160, abs=6)

    def test_read_frames_sequential_skips_frames_past_the_end(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Act
        video = make_video(tmp_path / 'v.mp4', n_frames=3)
        frames = read_frames_sequential(video, np.array([1, 9]))

        # Assert
        assert sorted(frames) == [1]

    def test_read_frames_sequential_raises_on_missing_video(self, tmp_path: Path):
        with pytest.raises(OSError, match='could not open'):
            read_frames_sequential(tmp_path / 'missing.mp4', np.array([0]))


class TestExportVideoFrames:
    """Test the function export_video_frames."""

    def test_export_video_frames_writes_only_selected_frames_and_list(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / f'{STEM}.mp4', n_frames=6)

        # Act
        save_dir = export_video_frames(video, tmp_path / 'out', np.array([4, 1]))

        # Assert: no context frames, just the two anchors and the csv
        assert sorted(p.name for p in save_dir.iterdir()) == [
            'img00000001.png', 'img00000004.png', 'selected_frames.csv',
        ]
        assert (save_dir / 'selected_frames.csv').read_text().split() == [
            'img00000001.png', 'img00000004.png',
        ]
        assert cv2.imread(str(save_dir / 'img00000001.png')).shape == (32, 32, 3)

    def test_export_video_frames_raises_when_frames_cannot_be_read(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / f'{STEM}.mp4', n_frames=3)

        # Act & Assert
        with pytest.raises(ValueError, match='could not read frames'):
            export_video_frames(video, tmp_path / 'out', np.array([1, 8]))


class TestBuildExtractionManifest:
    """Test the function build_extraction_manifest."""

    def test_build_extraction_manifest_columns_and_rows(self):
        # Act
        manifest = build_extraction_manifest([
            {'session_id': 'Day2_Tank1', 'fish_id': 'Cuttle1_Resident', 'frame_idx': 3,
             'image_path': '/x/img00000003.png'},
        ])

        # Assert
        assert list(manifest.columns) == ['session_id', 'fish_id', 'frame_idx', 'image_path']
        assert len(manifest) == 1

    def test_build_extraction_manifest_empty_rows_keeps_columns(self):
        assert list(build_extraction_manifest([]).columns) == [
            'session_id', 'fish_id', 'frame_idx', 'image_path',
        ]


class TestRunExport:
    """Test the function run_export."""

    def _selection(self, frames: list[int], video: str = STEM) -> pd.DataFrame:
        return pd.DataFrame({'video': video, 'frame_idx': frames})

    def test_run_export_writes_frames_and_returns_manifest(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        (tmp_path / 'rect').mkdir()
        make_video(tmp_path / 'rect' / f'{STEM}.mp4', n_frames=6)

        # Act
        manifest = run_export(self._selection([4, 1]), tmp_path / 'rect', tmp_path / 'out')

        # Assert
        assert list(manifest['frame_idx']) == [1, 4]
        assert set(manifest['session_id']) == {'Day2_Tank1'}
        assert set(manifest['fish_id']) == {'Cuttle1_Resident'}
        assert all(Path(p).exists() for p in manifest['image_path'])

    def test_run_export_skip_existing_does_not_rewrite_but_still_lists_frames(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        (tmp_path / 'rect').mkdir()
        make_video(tmp_path / 'rect' / f'{STEM}.mp4', n_frames=6)
        run_export(self._selection([1, 4]), tmp_path / 'rect', tmp_path / 'out')
        marker = tmp_path / 'out' / STEM / 'img00000001.png'
        marker.write_bytes(b'sentinel')

        # Act
        manifest = run_export(self._selection([1, 4]), tmp_path / 'rect', tmp_path / 'out',
                              skip_existing=True)

        # Assert
        assert marker.read_bytes() == b'sentinel'
        assert len(manifest) == 2

    def test_run_export_skips_videos_with_unexpected_names(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Act
        manifest = run_export(self._selection([1], video='weird_name'), tmp_path,
                              tmp_path / 'out')

        # Assert
        assert manifest.empty
