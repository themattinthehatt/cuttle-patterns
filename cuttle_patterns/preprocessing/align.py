"""Build an egocentrically-aligned crop video and rectangle-geometry CSV for a raw video.

Orchestrates `cuttle_patterns.preprocessing.inscribe` over every frame of a video: detect
a rectangle per frame, linearly interpolate over frames with no detected body (including
genuinely blank frames, where every pixel is 0), warp each frame's rectangle into a fixed
canonical size, and write both the resulting video and a CSV of the (interpolated)
rectangle corners plus which frames were interpolated. If a pose CSV is supplied (see
`cuttle_patterns.preprocessing.pose`), rectangle inscription switches to the pose-informed
Phase 2b path instead of Phase 2a's PCA-based one.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from cuttle_patterns.preprocessing.inscribe import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_THRESHOLD,
    inscribe_rectangle,
)
from cuttle_patterns.preprocessing.pose import interpolate_pose, load_pose_predictions

DEFAULT_CANONICAL_HEIGHT = 100
DEFAULT_SMOOTHING_WINDOW = 9
DEFAULT_SMOOTHING_SIGMA = 2.0

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
    tail_xy: np.ndarray | None = None,
    neck_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Run rectangle inscription on every frame of a video.

    Args:
        video_path: path to the raw video.
        thresh: passed through to `inscribe_rectangle`.
        aspect: passed through to `inscribe_rectangle`.
        tail_xy: optional (n_frames, 2) per-frame tail keypoints; if given along with
            `neck_xy`, switches `inscribe_rectangle` to the pose-informed Phase 2b path.
        neck_xy: optional (n_frames, 2) per-frame neck keypoints; if given along with
            `tail_xy`, switches `inscribe_rectangle` to the pose-informed Phase 2b path.

    Returns:
        (n_frames, 4, 2) array of rectangle corners. Frames with no detected body
        (including genuinely blank frames, where every pixel is 0) are all-NaN.

    Raises:
        OSError: if the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'computing rectangle trajectory for {video_path.name}...')
    rows = []
    try:
        idx = 0
        with tqdm(desc=video_path.name, total=n_frames) as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                tail = tuple(tail_xy[idx]) if tail_xy is not None else None
                neck = tuple(neck_xy[idx]) if neck_xy is not None else None
                result = inscribe_rectangle(
                    frame, thresh=thresh, aspect=aspect, tail=tail, neck=neck,
                )
                rows.append(result.corners if result is not None else np.full((4, 2), np.nan))
                idx += 1
                pbar.update(1)
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


def smooth_corners(
    corners: np.ndarray,
    window: int = DEFAULT_SMOOTHING_WINDOW,
) -> np.ndarray:
    """Temporally smooth a rectangle-corner trajectory with a centered rolling median.

    Rapid oscillations (e.g. fin beats) can make frame-to-frame rectangle geometry
    jitter even when the body's actual pose barely changes. A rolling median damps that
    without getting dragged around by a one- or two-frame outlier the way a moving
    average would, at the cost of some lag on genuinely fast movements if `window` is
    too wide.

    Args:
        corners: (n_frames, 4, 2) array of rectangle corners, no NaNs — run
            `interpolate_corners` first if there are undetected frames.
        window: number of frames in the rolling median window; a window of 1 is a no-op.

    Returns:
        (n_frames, 4, 2) array of smoothed corners.
    """
    n_frames = corners.shape[0]
    flat = corners.reshape(n_frames, 8)
    smoothed = pd.DataFrame(flat).rolling(window, center=True, min_periods=1).median()

    return smoothed.to_numpy().reshape(n_frames, 4, 2)


def smooth_corners_gaussian(
    corners: np.ndarray,
    sigma: float = DEFAULT_SMOOTHING_SIGMA,
) -> np.ndarray:
    """Temporally smooth a rectangle-corner trajectory with a Gaussian filter.

    Alternative to `smooth_corners`'s rolling median, better suited to continuous,
    quasi-periodic jitter (e.g. fin beats) that a median doesn't fully flatten, since it
    blends the whole window rather than snapping to one of the observed values. Trade-off:
    less robust to a single genuinely-bad frame, which gets blended in rather than
    rejected outright.

    Args:
        corners: (n_frames, 4, 2) array of rectangle corners, no NaNs — run
            `interpolate_corners` first if there are undetected frames.
        sigma: standard deviation, in frames, of the Gaussian kernel.

    Returns:
        (n_frames, 4, 2) array of smoothed corners.
    """
    n_frames = corners.shape[0]
    flat = corners.reshape(n_frames, 8)
    smoothed = gaussian_filter1d(flat, sigma=sigma, axis=0, mode='nearest')

    return smoothed.reshape(n_frames, 4, 2)


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
    pose_path: Path | None = None,
    smoothing_window: int | None = None,
    smoothing_sigma: float | None = None,
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
        pose_path: optional path to a per-frame tail/neck pose CSV (see
            `cuttle_patterns.preprocessing.pose.load_pose_predictions`); if given, uses
            the pose-informed Phase 2b rectangle-inscription path instead of Phase 2a's
            PCA-based one.
        smoothing_window: passed through to `smooth_corners`. Mutually exclusive with
            `smoothing_sigma`; if both are None (the default), falls back to
            `smooth_corners` with `DEFAULT_SMOOTHING_WINDOW`.
        smoothing_sigma: passed through to `smooth_corners_gaussian`. Mutually exclusive
            with `smoothing_window`; if given, smooths with a Gaussian filter instead of
            the rolling median.

    Returns:
        (video_out_path, csv_out_path).

    Raises:
        OSError: if the video file cannot be opened.
        ValueError: if both `smoothing_window` and `smoothing_sigma` are given.
    """
    if smoothing_window is not None and smoothing_sigma is not None:
        raise ValueError('smoothing_window and smoothing_sigma are mutually exclusive')
    output_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = output_dir / f'{video_path.stem}.mp4'
    csv_out_path = output_dir / f'{video_path.stem}.csv'

    if pose_path is not None:
        tail_xy, neck_xy, is_interpolated_pose = interpolate_pose(
            load_pose_predictions(pose_path),
        )
        corners = compute_corner_trajectory(
            video_path, thresh=thresh, aspect=aspect, tail_xy=tail_xy, neck_xy=neck_xy,
        )
    else:
        is_interpolated_pose = None
        corners = compute_corner_trajectory(video_path, thresh=thresh, aspect=aspect)

    corners, is_interpolated = interpolate_corners(corners)
    if is_interpolated_pose is not None:
        is_interpolated = is_interpolated | is_interpolated_pose

    if smoothing_sigma is not None:
        corners = smooth_corners_gaussian(corners, sigma=smoothing_sigma)
    else:
        window = smoothing_window if smoothing_window is not None else DEFAULT_SMOOTHING_WINDOW
        corners = smooth_corners(corners, window=window)

    canonical_size = (round(aspect * canonical_height), canonical_height)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    print(f'warping {video_path.name} into canonical crops...')
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, canonical_size)
        try:
            for idx in tqdm(range(corners.shape[0]), desc=video_path.name):
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
