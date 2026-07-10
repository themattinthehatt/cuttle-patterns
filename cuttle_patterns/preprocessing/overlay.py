"""Draw inscribed rectangles on raw video frames, from a saved rectangle-geometry CSV.

Reads the CSV written by `cuttle_patterns.preprocessing.align.align_video` and draws
each frame's (interpolated) rectangle on top of the corresponding raw frame, so
inscription quality can be reviewed without re-running detection. Frames that were
originally interpolated (no body detected) are drawn in a different color, so
low-confidence stretches are visually distinguishable from directly-detected ones.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.preprocessing.align import CORNER_COLUMNS

DETECTED_COLOR_BGR = (0, 255, 0)
INTERPOLATED_COLOR_BGR = (0, 165, 255)


def create_overlay_video(video_path: Path, csv_path: Path, output_path: Path) -> Path:
    """Draw each frame's rectangle (from a geometry CSV) on top of the raw video.

    Args:
        video_path: path to the raw video.
        csv_path: path to the rectangle-geometry CSV written by `align_video`.
        output_path: path to write the overlay mp4 to.

    Returns:
        output_path.

    Raises:
        OSError: if the video file cannot be opened.
    """
    df = pd.read_csv(csv_path)
    corners = df[CORNER_COLUMNS].to_numpy().reshape(-1, 4, 2)
    is_interpolated = df['is_interpolated'].to_numpy()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        try:
            for idx in range(len(df)):
                ok, frame = cap.read()
                if not ok:
                    break
                color = INTERPOLATED_COLOR_BGR if is_interpolated[idx] else DETECTED_COLOR_BGR
                poly = corners[idx].astype(np.int32)
                cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()

    return output_path
