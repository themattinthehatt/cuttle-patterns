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
    create_overlay_video,
)


def _plain_frame(height: int = 40, width: int = 40) -> np.ndarray:
    return np.full((height, width, 3), 50, dtype=np.uint8)


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

        # Act
        result_path = create_overlay_video(video_path, csv_path, output_path)

        # Assert
        assert result_path == output_path
        assert output_path.exists()

        cap = cv2.VideoCapture(str(output_path))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        ok0, frame0 = cap.read()
        ok1, frame1 = cap.read()
        cap.release()

        assert ok0 and ok1
        # left edge of the square, drawn on both frames (loose tolerance: mp4
        # compression perturbs exact pixel values slightly)
        assert np.allclose(frame0[10, 5], DETECTED_COLOR_BGR, atol=15)
        assert np.allclose(frame1[10, 5], INTERPOLATED_COLOR_BGR, atol=15)
