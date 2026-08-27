"""Tests for cuttle_patterns.preprocessing.overlay."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.preprocessing.align import CORNER_COLUMNS
from cuttle_patterns.preprocessing.overlay import (
    DETECTED_COLOR_BGR,
    INTERPOLATED_COLOR_BGR,
    KEYPOINT_COLOR_BGR,
    create_overlay_video,
)


def _plain_frame(height: int = 40, width: int = 40) -> np.ndarray:
    return np.full((height, width, 3), 50, dtype=np.uint8)


def _write_pose_csv(
    path: Path, n_frames: int, tail: tuple, tail_likelihood: float, neck: tuple,
    neck_likelihood: float,
) -> Path:
    columns = pd.MultiIndex.from_tuples(
        [
            ('m', 'neck', 'x'), ('m', 'neck', 'y'), ('m', 'neck', 'likelihood'),
            ('m', 'tail', 'x'), ('m', 'tail', 'y'), ('m', 'tail', 'likelihood'),
        ],
        names=['scorer', 'bodyparts', 'coords'],
    )
    row = [neck[0], neck[1], neck_likelihood, tail[0], tail[1], tail_likelihood]
    pd.DataFrame([row] * n_frames, columns=columns).to_csv(path)
    return path


class TestCreateOverlayVideo:
    """Test the function create_overlay_video."""

    def test_create_overlay_video_colors_by_interpolated_flag(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
    ):
        # Arrange: a 2-frame video and a matching geometry CSV, one detected frame and
        # one interpolated frame, both using the same small square so the drawn edge
        # color at a known pixel is directly comparable
        video_path = make_custom_video(
            tmp_path / 'video.mp4', [_plain_frame(), _plain_frame()],
        )
        square = [5, 5, 15, 5, 15, 15, 5, 15]
        df = pd.DataFrame([square, square], columns=CORNER_COLUMNS)
        df.insert(0, 'frame_idx', [0, 1])
        df['is_interpolated'] = [False, True]
        csv_path = tmp_path / 'video.csv'
        df.to_csv(csv_path, index=False)
        output_path = tmp_path / 'out' / 'video_overlay.mp4'

        # Act: pin crf to near-lossless so this assertion isn't coupled to whatever the
        # module's default compression level happens to be (the default, crf=28, is
        # deliberately lossy for file size and its exact pixel drift is also sensitive
        # to the local libx264 build, which caused this test to flake across machines)
        result_path = create_overlay_video(video_path, csv_path, output_path, crf=0)

        # Assert
        assert result_path == output_path
        assert output_path.exists()

        cap = cv2.VideoCapture(str(output_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        ok0, frame0 = cap.read()
        ok1, frame1 = cap.read()
        cap.release()

        assert ok0 and ok1
        # left edge of the square, drawn on both frames (loose tolerance: yuv420p
        # chroma subsampling still perturbs exact pixel values slightly even losslessly)
        assert np.allclose(frame0[10, 5], DETECTED_COLOR_BGR, atol=15)
        assert np.allclose(frame1[10, 5], INTERPOLATED_COLOR_BGR, atol=15)

    def test_create_overlay_video_draws_keypoints_above_likelihood_thresh(
        self,
        tmp_path: Path,
        make_custom_video: Callable,
    ):
        # Arrange: one frame, a rectangle away from both keypoints, tail above the
        # likelihood threshold (should be drawn) and neck below it (should not)
        video_path = make_custom_video(tmp_path / 'video.mp4', [_plain_frame()])
        square = [0, 0, 5, 0, 5, 5, 0, 5]
        df = pd.DataFrame([square], columns=CORNER_COLUMNS)
        df.insert(0, 'frame_idx', [0])
        df['is_interpolated'] = [False]
        csv_path = tmp_path / 'video.csv'
        df.to_csv(csv_path, index=False)

        pose_path = _write_pose_csv(
            tmp_path / 'video_pose.csv', n_frames=1,
            tail=(30, 20), tail_likelihood=0.99,
            neck=(20, 30), neck_likelihood=0.5,
        )
        output_path = tmp_path / 'out' / 'video_overlay.mp4'

        # Act
        create_overlay_video(video_path, csv_path, output_path, crf=0, pose_path=pose_path)

        # Assert
        cap = cv2.VideoCapture(str(output_path))
        ok, frame = cap.read()
        cap.release()

        assert ok
        assert np.allclose(frame[20, 30], KEYPOINT_COLOR_BGR, atol=15)
        assert not np.allclose(frame[30, 20], KEYPOINT_COLOR_BGR, atol=15)
