"""Build an egocentrically-aligned crop video and rectangle-geometry CSV for a raw video.

Orchestrates `cuttle_patterns.preprocessing.inscribe` over every frame of a video: detect
a rectangle per frame, linearly interpolate over frames with no detected body (including
genuinely blank frames, where every pixel is 0), warp each frame's rectangle into a fixed
canonical size, and write both the resulting video and a CSV of the (interpolated)
rectangle corners plus which frames were interpolated.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.preprocessing.inscribe import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_THRESHOLD,
    inscribe_rectangle,
)

DEFAULT_CANONICAL_HEIGHT = 100

CORNER_COLUMNS = [
    'corner_tl_x', 'corner_tl_y',
    'corner_tr_x', 'corner_tr_y',
    'corner_br_x', 'corner_br_y',
    'corner_bl_x', 'corner_bl_y',
]


def compute_corner_trajectory(
    video_path: Path,
    thresh: int = DEFAULT_THRESHOLD,
    aspect: float = DEFAULT_ASPECT_RATIO,
) -> np.ndarray:
    """Run rectangle inscription on every frame of a video.

    Args:
        video_path: path to the raw video.
        thresh: passed through to `inscribe_rectangle`.
        aspect: passed through to `inscribe_rectangle`.

    Returns:
        (n_frames, 4, 2) array of rectangle corners. Frames with no detected body
        (including genuinely blank frames, where every pixel is 0) are all-NaN.

    Raises:
        OSError: if the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    rows = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = inscribe_rectangle(frame, thresh=thresh, aspect=aspect)
            rows.append(result.corners if result is not None else np.full((4, 2), np.nan))
    finally:
        cap.release()

    return np.stack(rows, axis=0)


def interpolate_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate rectangle corners over frames with no detected body.

    Frames before the first, or after the last, detected body are filled with the
    nearest valid frame's corners (flat extrapolation) — there's no trend to extrapolate
    for a bounding box.

    Args:
        corners: (n_frames, 4, 2) array, NaN where no body was detected.

    Returns:
        (interpolated corners, is_interpolated), where is_interpolated is a boolean
        array of shape (n_frames,), True for frames that were NaN before interpolation.

    Raises:
        ValueError: if no frame has a detected body.
    """
    n_frames = corners.shape[0]
    flat = corners.reshape(n_frames, 8)
    is_valid = ~np.isnan(flat).any(axis=1)

    if not is_valid.any():
        raise ValueError('no frame in this video has a detected body; cannot interpolate')

    frame_indices = np.arange(n_frames)
    interpolated = flat.copy()
    for col in range(flat.shape[1]):
        interpolated[:, col] = np.interp(
            frame_indices, frame_indices[is_valid], flat[is_valid, col],
        )

    return interpolated.reshape(n_frames, 4, 2), ~is_valid


def warp_to_canonical(
    frame: np.ndarray,
    corners: np.ndarray,
    canonical_size: tuple[int, int],
) -> np.ndarray:
    """Warp a frame's inscribed rectangle into a fixed-size, upright canonical crop.

    Args:
        frame: decoded video frame (BGR).
        corners: (4, 2) rectangle corners, clockwise from top-left.
        canonical_size: (width, height) of the output crop.

    Returns:
        the warped crop, shaped `(height, width, ...)` to match `canonical_size`.
    """
    width, height = canonical_size
    src = corners[[0, 1, 3]].astype(np.float32)
    dst = np.array([[0, 0], [width, 0], [0, height]], dtype=np.float32)
    m = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(frame, m, (width, height))


def align_video(
    video_path: Path,
    output_dir: Path,
    thresh: int = DEFAULT_THRESHOLD,
    aspect: float = DEFAULT_ASPECT_RATIO,
    canonical_height: int = DEFAULT_CANONICAL_HEIGHT,
) -> tuple[Path, Path]:
    """Build an aligned crop video and rectangle-geometry CSV for one raw video.

    Two passes over the video: the first computes and interpolates the rectangle
    trajectory frame-by-frame, without buffering raw frames; the second re-reads the
    video to warp each frame's rectangle into the canonical crop and write it out.

    Args:
        video_path: path to the raw video.
        output_dir: directory to write `{video_path.stem}.mp4` / `.csv` into.
        thresh: passed through to `inscribe_rectangle`.
        aspect: passed through to `inscribe_rectangle`.
        canonical_height: output crop height in pixels; width is `round(aspect * height)`.

    Returns:
        (video_out_path, csv_out_path).

    Raises:
        OSError: if the video file cannot be opened.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = output_dir / f'{video_path.stem}.mp4'
    csv_out_path = output_dir / f'{video_path.stem}.csv'

    corners = compute_corner_trajectory(video_path, thresh=thresh, aspect=aspect)
    corners, is_interpolated = interpolate_corners(corners)

    canonical_size = (round(aspect * canonical_height), canonical_height)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, canonical_size)
        try:
            for idx in range(corners.shape[0]):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(warp_to_canonical(frame, corners[idx], canonical_size))
        finally:
            writer.release()
    finally:
        cap.release()

    df = pd.DataFrame(corners.reshape(-1, 8), columns=CORNER_COLUMNS)
    df.insert(0, 'frame_idx', np.arange(corners.shape[0]))
    df['is_interpolated'] = is_interpolated
    df.to_csv(csv_out_path, index=False)

    return video_out_path, csv_out_path
