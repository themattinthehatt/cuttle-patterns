"""Shared fixtures for cuttle_patterns tests."""

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video() -> Callable[..., Path]:
    """Return a factory that writes a small synthetic mp4 file.

    Returns:
        a callable taking (path, n_frames, fps, width, height) and writing a video to
        path, returning that same path.
    """

    def _make_video(
        path: Path,
        n_frames: int = 5,
        fps: float = 10.0,
        width: int = 32,
        height: int = 32,
    ) -> Path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for _ in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _make_video
