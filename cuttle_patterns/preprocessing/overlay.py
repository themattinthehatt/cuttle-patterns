"""Draw inscribed rectangles on raw video frames, from a saved rectangle-geometry CSV.

Reads the CSV written by `cuttle_patterns.preprocessing.align.align_video` and draws
each frame's (interpolated) rectangle on top of the corresponding raw frame, so
inscription quality can be reviewed without re-running detection. Frames that were
originally interpolated (no body detected) are drawn in a different color, so
low-confidence stretches are visually distinguishable from directly-detected ones.
"""

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from cuttle_patterns.preprocessing.align import CORNER_COLUMNS

DETECTED_COLOR_BGR = (0, 255, 0)
INTERPOLATED_COLOR_BGR = (0, 165, 255)

# overlay videos are full raw-resolution QC output, potentially thousands of frames;
# cv2.VideoWriter's ffmpeg backend has no H.264 encoder available in some environments
# (falls back to the far less efficient mp4v/MPEG-4 Part 2 codec), so frames are piped to
# the system ffmpeg binary instead, encoding with libx264 for a real (typically 5-10x)
# size reduction at visually-lossless quality
DEFAULT_CRF = 23


def _open_ffmpeg_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
) -> subprocess.Popen:
    """Launch an ffmpeg subprocess that reads raw BGR24 frames on stdin and writes H.264.

    Args:
        output_path: path to write the encoded mp4 to.
        width: frame width in pixels.
        height: frame height in pixels.
        fps: output frame rate.
        crf: x264 constant rate factor (lower = higher quality/larger file).

    Returns:
        the running subprocess, with `stdin` open for raw frame bytes.

    Raises:
        OSError: if the ffmpeg binary is not found on PATH.
    """
    if shutil.which('ffmpeg') is None:
        raise OSError('ffmpeg not found on PATH; required to write overlay videos')

    command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf), '-pix_fmt', 'yuv420p',
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def create_overlay_video(
    video_path: Path,
    csv_path: Path,
    output_path: Path,
    crf: int = DEFAULT_CRF,
) -> Path:
    """Draw each frame's rectangle (from a geometry CSV) on top of the raw video.

    Args:
        video_path: path to the raw video.
        csv_path: path to the rectangle-geometry CSV written by `align_video`.
        output_path: path to write the overlay mp4 to.
        crf: x264 constant rate factor passed to ffmpeg; lower means higher quality and a
            larger file.

    Returns:
        output_path.

    Raises:
        OSError: if the video file, or the ffmpeg binary, cannot be found.
        RuntimeError: if ffmpeg exits with an error.
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

        writer = _open_ffmpeg_writer(output_path, width, height, fps, crf)
        try:
            for idx in tqdm(range(len(df)), desc=video_path.name):
                ok, frame = cap.read()
                if not ok:
                    break
                color = INTERPOLATED_COLOR_BGR if is_interpolated[idx] else DETECTED_COLOR_BGR
                poly = corners[idx].astype(np.int32)
                cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)
                writer.stdin.write(frame.tobytes())
        finally:
            writer.stdin.close()
            writer.wait()
    finally:
        cap.release()

    if writer.returncode != 0:
        raise RuntimeError(f'ffmpeg exited with code {writer.returncode} writing {output_path}')

    return output_path
