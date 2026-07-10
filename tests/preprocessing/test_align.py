"""Tests for cuttle_patterns.preprocessing.align."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.align import (
    align_video,
    compute_corner_trajectory,
    interpolate_corners,
    warp_to_canonical,
)


def _blob_frame(height: int = 60, width: int = 120) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 15), (100, 45), (100, 100, 100), -1)
    return frame


def _blank_frame(height: int = 60, width: int = 120) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _write_pose_csv(path: Path, n_frames: int, tail: tuple, neck: tuple) -> Path:
    # same tail/neck point repeated for every frame, at full likelihood
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


class TestComputeCornerTrajectory:
    """Test the function compute_corner_trajectory."""

    def test_compute_corner_trajectory_blank_frame_is_nan(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
    ):
        # Arrange: a genuinely blank frame (every pixel 0) sandwiched between two
        # frames with a body present
        video_path = make_custom_video(
            tmp_path / 'video.mp4',
            [_blob_frame(), _blank_frame(), _blob_frame()],
        )

        # Act
        corners = compute_corner_trajectory(video_path)

        # Assert
        assert corners.shape == (3, 4, 2)
        assert not np.isnan(corners[0]).any()
        assert np.isnan(corners[1]).all()
        assert not np.isnan(corners[2]).any()

    def test_compute_corner_trajectory_uses_pose_when_given(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
    ):
        # Arrange
        video_path = make_custom_video(tmp_path / 'video.mp4', [_blob_frame()])
        tail_xy = np.array([[25.0, 30.0]])
        neck_xy = np.array([[95.0, 30.0]])

        # Act
        corners = compute_corner_trajectory(video_path, tail_xy=tail_xy, neck_xy=neck_xy)

        # Assert
        assert corners.shape == (1, 4, 2)
        assert not np.isnan(corners).any()


class TestInterpolateCorners:
    """Test the function interpolate_corners."""

    def test_interpolate_corners_fills_interior_gap(self):
        # Arrange
        corners = np.full((3, 4, 2), np.nan)
        corners[0] = 0.0
        corners[2] = 10.0

        # Act
        filled, is_interpolated = interpolate_corners(corners)

        # Assert
        assert not np.isnan(filled).any()
        assert filled[1] == pytest.approx(5.0)
        assert list(is_interpolated) == [False, True, False]

    def test_interpolate_corners_flat_fills_edges(self):
        # Arrange: no valid frame at the start or end
        corners = np.full((3, 4, 2), np.nan)
        corners[1] = 5.0

        # Act
        filled, is_interpolated = interpolate_corners(corners)

        # Assert
        assert filled[0] == pytest.approx(5.0)
        assert filled[2] == pytest.approx(5.0)
        assert list(is_interpolated) == [True, False, True]

    def test_interpolate_corners_no_valid_frames_raises(self):
        # Arrange
        corners = np.full((3, 4, 2), np.nan)

        # Act & Assert
        with pytest.raises(ValueError, match='no frame'):
            interpolate_corners(corners)


class TestWarpToCanonical:
    """Test the function warp_to_canonical."""

    def test_warp_to_canonical_output_shape(self):
        # Arrange
        frame = _blob_frame()
        corners = np.array([[20, 15], [100, 15], [100, 45], [20, 45]], dtype=np.float64)

        # Act
        warped = warp_to_canonical(frame, corners, canonical_size=(80, 40))

        # Assert: shape matches (height, width) from canonical_size, and the source
        # rectangle was uniformly filled, so the canonical crop should mostly be too
        assert warped.shape == (40, 80, 3)
        assert (warped > 0).mean() > 0.9


class TestAlignVideo:
    """Test the function align_video."""

    def test_align_video_writes_video_and_csv(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
    ):
        # Arrange
        video_path = make_custom_video(
            tmp_path / 'session-01_cuttle-01.mp4',
            [_blob_frame(), _blank_frame(), _blob_frame()],
        )
        output_dir = tmp_path / 'out'

        # Act
        video_out_path, csv_out_path = align_video(video_path, output_dir, canonical_height=20)

        # Assert
        assert video_out_path == output_dir / 'session-01_cuttle-01.mp4'
        assert csv_out_path == output_dir / 'session-01_cuttle-01.csv'
        assert video_out_path.exists()
        assert csv_out_path.exists()

        df = pd.read_csv(csv_out_path)
        assert len(df) == 3
        assert list(df['is_interpolated']) == [False, True, False]
        assert not df.drop(columns=['frame_idx', 'is_interpolated']).isna().any().any()

        cap = cv2.VideoCapture(str(video_out_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 40
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 20
        cap.release()

    def test_align_video_with_pose_path(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
    ):
        # Arrange
        video_path = make_custom_video(
            tmp_path / 'session-01_cuttle-01.mp4',
            [_blob_frame(), _blob_frame(), _blob_frame()],
        )
        pose_path = _write_pose_csv(
            tmp_path / 'pose.csv', n_frames=3, tail=(25.0, 30.0), neck=(95.0, 30.0),
        )
        output_dir = tmp_path / 'out'

        # Act
        video_out_path, csv_out_path = align_video(
            video_path, output_dir, canonical_height=20, pose_path=pose_path,
        )

        # Assert
        assert video_out_path.exists()
        assert csv_out_path.exists()

        df = pd.read_csv(csv_out_path)
        assert len(df) == 3
        assert not df['is_interpolated'].any()

        cap = cv2.VideoCapture(str(video_out_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        cap.release()
