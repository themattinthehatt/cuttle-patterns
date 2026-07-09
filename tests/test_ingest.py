"""Tests for cuttle_patterns.ingest."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.ingest import (
    build_manifest,
    find_raw_videos,
    read_blank_frame_indices,
    read_video_info,
)


class TestFindRawVideos:
    """Test the function find_raw_videos."""

    def test_find_raw_videos_success(self, tmp_path: Path, make_video: Callable):
        # Arrange
        make_video(tmp_path / 'session-01_cuttle-01.mp4')
        make_video(tmp_path / 'session-01_cuttle-02.mp4')
        (tmp_path / 'unrelated.mp4').write_bytes(b'')

        # Act
        videos = find_raw_videos(tmp_path)

        # Assert
        assert [v.name for v in videos] == [
            'session-01_cuttle-01.mp4',
            'session-01_cuttle-02.mp4',
        ]

    def test_find_raw_videos_missing_dir(self, tmp_path: Path):
        # Arrange
        missing_dir = tmp_path / 'does_not_exist'

        # Act & Assert
        with pytest.raises(FileNotFoundError, match='does not exist'):
            find_raw_videos(missing_dir)


class TestReadVideoInfo:
    """Test the function read_video_info."""

    def test_read_video_info_success(self, tmp_path: Path, make_video: Callable):
        # Arrange
        video_path = make_video(
            tmp_path / 'video.mp4', n_frames=7, fps=15.0, width=48, height=32,
        )

        # Act
        info = read_video_info(video_path)

        # Assert
        assert info.n_frames == 7
        assert info.fps == pytest.approx(15.0)
        assert info.width == 48
        assert info.height == 32

    def test_read_video_info_missing_file(self, tmp_path: Path):
        # Arrange
        video_path = tmp_path / 'does_not_exist.mp4'

        # Act & Assert
        with pytest.raises(OSError, match='could not open video file'):
            read_video_info(video_path)


class TestReadBlankFrameIndices:
    """Test the function read_blank_frame_indices."""

    def test_read_blank_frame_indices_success(self, tmp_path: Path):
        # Arrange
        blank_frames_path = tmp_path / 'blank.txt'
        blank_frames_path.write_text('5\n3\n10\n')

        # Act
        indices = read_blank_frame_indices(blank_frames_path)

        # Assert
        assert indices == [3, 5, 10]

    def test_read_blank_frame_indices_empty_lines_ignored(self, tmp_path: Path):
        # Arrange
        blank_frames_path = tmp_path / 'blank.txt'
        blank_frames_path.write_text('5\n\n3\n')

        # Act
        indices = read_blank_frame_indices(blank_frames_path)

        # Assert
        assert indices == [3, 5]


class TestBuildManifest:
    """Test the function build_manifest."""

    def test_build_manifest_success(self, tmp_path: Path, make_video: Callable):
        # Arrange
        make_video(tmp_path / 'session-01_cuttle-01.mp4', n_frames=10)
        (tmp_path / 'session-01_cuttle-01.txt').write_text('1\n2\n3\n')
        make_video(tmp_path / 'session-01_cuttle-02.mp4', n_frames=10)
        (tmp_path / 'session-01_cuttle-02.txt').write_text('4\n')

        # Act
        manifest = build_manifest(tmp_path)

        # Assert
        assert len(manifest) == 2
        assert set(manifest['session_id']) == {1}
        assert set(manifest['fish_id']) == {1, 2}
        assert manifest.loc[manifest['fish_id'] == 1, 'n_blank_frames'].item() == 3
        assert manifest.loc[manifest['fish_id'] == 2, 'n_blank_frames'].item() == 1

    def test_build_manifest_missing_blank_frames_file(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        make_video(tmp_path / 'session-01_cuttle-01.mp4', n_frames=10)

        # Act
        manifest = build_manifest(tmp_path)

        # Assert
        assert len(manifest) == 1
        assert pd.isna(manifest.loc[0, 'n_blank_frames'])
        assert manifest.loc[0, 'blank_frames_path'] is None

    def test_build_manifest_no_videos(self, tmp_path: Path):
        # Act
        manifest = build_manifest(tmp_path)

        # Assert
        assert manifest.empty
