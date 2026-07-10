"""Load per-frame tail/neck keypoint predictions for pose-informed rectangle inscription.

The pose-estimation pipeline (outside this codebase) writes one CSV per video, in the
standard multi-header format used by pose-estimation tools (Lightning Pose/DeepLabCut-style):
three header rows (scorer, bodyparts, coords) followed by one data row per frame. See
`docs/PHASES.md` (Phase 2b) for how tail/neck keypoints feed into rectangle inscription.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_LIKELIHOOD_THRESH = 0.9

KEYPOINTS = ('tail', 'neck')


def load_pose_predictions(csv_path: Path) -> pd.DataFrame:
    """Load tail/neck keypoint predictions from a pose-estimation CSV.

    Args:
        csv_path: path to a pose CSV in the standard multi-header format (three header
            rows: scorer, bodyparts, coords), containing at least 'tail' and 'neck'
            bodyparts. The 'scorer' label itself is ignored, so this works regardless of
            which pose model/run produced the file.

    Returns:
        a tidy frame indexed by frame_idx with columns tail_x, tail_y, tail_likelihood,
        neck_x, neck_y, neck_likelihood.
    """
    raw = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    raw.columns = raw.columns.droplevel('scorer')
    raw.index.name = 'frame_idx'

    columns = {}
    for keypoint in KEYPOINTS:
        subset = raw.xs(keypoint, axis=1, level='bodyparts')
        columns[f'{keypoint}_x'] = subset['x']
        columns[f'{keypoint}_y'] = subset['y']
        columns[f'{keypoint}_likelihood'] = subset['likelihood']

    return pd.DataFrame(columns, index=raw.index)


def interpolate_pose(
    pose_df: pd.DataFrame,
    likelihood_thresh: float = DEFAULT_LIKELIHOOD_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate tail/neck keypoints over low-confidence frames.

    A frame is considered invalid (and therefore interpolated) if either keypoint's
    likelihood is below `likelihood_thresh`. Frames before the first, or after the last,
    valid frame are filled with the nearest valid frame's keypoints (flat extrapolation)
    — same convention as `align.interpolate_corners`.

    Args:
        pose_df: tidy pose frame from `load_pose_predictions`, indexed by frame_idx, one
            row per video frame.
        likelihood_thresh: minimum per-keypoint likelihood to trust a frame's prediction.

    Returns:
        (tail_xy, neck_xy, is_interpolated): `tail_xy`/`neck_xy` are (len(pose_df), 2)
        arrays, `is_interpolated` is a boolean array of shape (len(pose_df),), True for
        frames that were filled in rather than directly trusted.

    Raises:
        ValueError: if no frame has a valid prediction for both keypoints.
    """
    is_valid = (
        (pose_df['tail_likelihood'] >= likelihood_thresh)
        & (pose_df['neck_likelihood'] >= likelihood_thresh)
    ).to_numpy()

    if not is_valid.any():
        raise ValueError('no frame has a valid tail/neck prediction; cannot interpolate')

    frame_indices = np.arange(len(pose_df))
    columns = ['tail_x', 'tail_y', 'neck_x', 'neck_y']
    flat = pose_df[columns].to_numpy()
    interpolated = flat.copy()
    for col in range(flat.shape[1]):
        interpolated[:, col] = np.interp(
            frame_indices, frame_indices[is_valid], flat[is_valid, col],
        )

    tail_xy = interpolated[:, 0:2]
    neck_xy = interpolated[:, 2:4]

    return tail_xy, neck_xy, ~is_valid
