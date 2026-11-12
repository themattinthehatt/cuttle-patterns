"""Per-frame eligibility for frame selection (`cuttle extract-eligibility`).

A frame is eligible only if all of the following hold (thresholds are adjustable):

1. not blank (the collaborators' `_black_frames.txt`);
2. every tail/neck keypoint has likelihood >= `likelihood_thresh`;
3. the `cuttle inscribe` rectangle's long edge is >= `small_rect_ratio_thresh` of the neck-tail
   distance (a body clearly longer than the box inscribed around it is a sizing failure);
4. no keypoint jump: each keypoint moves <= `jump_max` px between t-1 and t and between t and
   t+1 (pose coordinates are in the 512x512 crop frame). Only steps whose two endpoints are both
   high-confidence count; the missing neighbor of the first/last frame is skipped;
5. motion energy (summed absolute difference of consecutive 32x32 frames of the aligned
   rectangle video) at or below the `me_percentile` percentile of the frames passing 1-4,
   which drops alignment glitches and heavy blur; frame 0 (no previous frame) is never eligible;
6. fewer than `black_max` of the rectangle frame's pixels are pure black (max channel == 0),
   i.e. background from a rectangle that extends beyond the animal.

Each criterion is stored as its own boolean column (True = the frame fails it), so rejected
frames can be inspected and thresholds re-tuned without recomputing the pixel statistics.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from cuttle_patterns.ingest import FILENAME_PATTERN, read_blank_frame_indices
from cuttle_patterns.preprocessing import pose
from cuttle_patterns.preprocessing.align import CORNER_COLUMNS
from cuttle_patterns.preprocessing.extract.common import (
    progress_path,
    save_parquet_atomic,
    shard_items,
    write_progress,
)

_logger = logging.getLogger(__name__)

DEFAULT_SMALL_RECT_RATIO_THRESH = 0.5
DEFAULT_JUMP_MAX = 25.0
DEFAULT_ME_PERCENTILE = 99.0
DEFAULT_BLACK_MAX = 0.05
ME_RESIZE_DIMS = 32

TABLE_COLUMNS = [
    'frame_idx', 'blank', 'low_likelihood', 'small_rect', 'big_jump', 'frame0', 'me_top',
    'black_high', 'eligible', 'motion_energy', 'black_frac',
]


def compute_small_rectangle_mask(
    rect_csv_path: Path,
    pose_df: pd.DataFrame,
    ratio_thresh: float = DEFAULT_SMALL_RECT_RATIO_THRESH,
    likelihood_thresh: float = pose.DEFAULT_LIKELIHOOD_THRESH,
) -> np.ndarray:
    """Flag frames where the rectangle's long edge is too short relative to the body.

    Args:
        rect_csv_path: path to `cuttle inscribe`'s per-frame corner geometry CSV.
        pose_df: tidy pose frame from `pose.load_pose_predictions`, one row per video
            frame, same length as `rect_csv_path`.
        ratio_thresh: a frame is flagged if the rectangle's long edge is less than this
            fraction of the neck-tail distance.
        likelihood_thresh: only frames where both keypoints meet this likelihood are
            considered (a low-confidence prediction can't be trusted as ground truth).

    Returns:
        boolean array of shape (len(pose_df),), True for a flagged frame.

    Raises:
        ValueError: if the rectangle CSV and pose frame don't have the same length.
    """
    rect_df = pd.read_csv(rect_csv_path)
    if len(rect_df) != len(pose_df):
        raise ValueError(
            f'{rect_csv_path} has {len(rect_df)} frames but the pose data has '
            f'{len(pose_df)}; expected one row per video frame in both'
        )

    corners = rect_df[CORNER_COLUMNS].to_numpy().reshape(-1, 4, 2)
    tl, tr, bl = corners[:, 0], corners[:, 1], corners[:, 3]
    long_edge = np.maximum(
        np.linalg.norm(tr - tl, axis=1), np.linalg.norm(bl - tl, axis=1),
    )

    tail_xy = pose_df[['tail_x', 'tail_y']].to_numpy()
    neck_xy = pose_df[['neck_x', 'neck_y']].to_numpy()
    neck_tail_dist = np.linalg.norm(neck_xy - tail_xy, axis=1)
    likelihood_ok = (
        (pose_df['tail_likelihood'] >= likelihood_thresh)
        & (pose_df['neck_likelihood'] >= likelihood_thresh)
    ).to_numpy()

    return likelihood_ok & (long_edge < ratio_thresh * neck_tail_dist)


def compute_keypoint_jump(
    pose_df: pd.DataFrame,
    likelihood_thresh: float = pose.DEFAULT_LIKELIHOOD_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute each frame's largest keypoint jump to its neighbors, and per-frame confidence.

    Args:
        pose_df: tidy pose frame from `pose.load_pose_predictions`.
        likelihood_thresh: minimum likelihood for a keypoint to be trusted.

    Returns:
        `(jump, ok)`: `jump` has shape (n_frames,), the max euclidean displacement (px) over
        keypoints and over the steps t-1 -> t and t -> t+1, counting only steps whose two
        endpoints are both high-confidence (nan where no step qualifies); `ok` is a boolean
        mask of frames whose own keypoints are all high-confidence.
    """
    n_frames = len(pose_df)
    jump = np.full(n_frames, np.nan)
    ok = np.ones(n_frames, dtype=bool)
    for keypoint in pose.KEYPOINTS:
        xy = pose_df[[f'{keypoint}_x', f'{keypoint}_y']].to_numpy()
        good = pose_df[f'{keypoint}_likelihood'].to_numpy() >= likelihood_thresh
        ok &= good
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # step[i] is frame i -> i + 1
        step = np.where(good[:-1] & good[1:], step, np.nan)
        # frame t sees step[t - 1] (t - 1 -> t) and step[t] (t -> t + 1)
        before = np.concatenate([[np.nan], step])
        after = np.concatenate([step, [np.nan]])
        jump = np.fmax(jump, np.fmax(before, after))
    return jump, ok


def compute_video_pixel_stats(
    video_path: Path,
    resize_dims: int = ME_RESIZE_DIMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame motion energy and pure-black pixel fraction in one decode pass.

    Motion energy matches `beast.video.compute_video_motion_energy`: the summed absolute
    difference between consecutive frames after resizing to `resize_dims` x `resize_dims`
    (`cv2.resize`, linear). It is a backward difference, so frame 0 has no meaningful value.

    Args:
        video_path: aligned rectangle video.
        resize_dims: side length frames are resized to before differencing.

    Returns:
        `(motion_energy, black_frac)`, float32 arrays of shape (n_frames,). `black_frac` is the
        share of full-resolution pixels whose max channel value is exactly 0.

    Raises:
        OSError: if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion, black = [], []
    previous = None
    with tqdm(total=n_total, desc=video_path.stem, unit='frame', leave=False) as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (resize_dims, resize_dims)).astype(np.int32)
            motion.append(0.0 if previous is None else float(np.abs(small - previous).sum()))
            previous = small
            black.append(float((frame.max(axis=2) == 0).mean()))
            pbar.update(1)
    cap.release()
    return np.array(motion, dtype=np.float32), np.array(black, dtype=np.float32)


def build_eligibility_table(
    pose_df: pd.DataFrame,
    blank_frame_idxs: list[int],
    small_rect: np.ndarray,
    motion_energy: np.ndarray,
    black_frac: np.ndarray,
    jump_max: float = DEFAULT_JUMP_MAX,
    me_percentile: float = DEFAULT_ME_PERCENTILE,
    black_max: float = DEFAULT_BLACK_MAX,
    likelihood_thresh: float = pose.DEFAULT_LIKELIHOOD_THRESH,
) -> pd.DataFrame:
    """Combine all criteria into one per-frame table.

    Args:
        pose_df: tidy pose frame, one row per video frame.
        blank_frame_idxs: indices flagged as blank.
        small_rect: boolean array from `compute_small_rectangle_mask`.
        motion_energy: per-frame motion energy from `compute_video_pixel_stats`.
        black_frac: per-frame pure-black fraction from `compute_video_pixel_stats`.
        jump_max: largest allowed keypoint jump, in px.
        me_percentile: frames above this motion-energy percentile (among frames passing the
            other pose/blank/rectangle criteria) are dropped.
        black_max: frames with at least this fraction of pure-black pixels are dropped.
        likelihood_thresh: minimum keypoint likelihood.

    Returns:
        one row per frame with `TABLE_COLUMNS`; a True flag means the frame fails that
        criterion, and `eligible` is True only if it fails none of them.

    Raises:
        ValueError: if the per-frame inputs differ in length.
    """
    n_frames = len(pose_df)
    if not len(small_rect) == len(motion_energy) == len(black_frac) == n_frames:
        raise ValueError(
            f'per-frame inputs differ in length: pose {n_frames}, small_rect '
            f'{len(small_rect)}, motion energy {len(motion_energy)}, black fraction '
            f'{len(black_frac)}'
        )

    blank = np.zeros(n_frames, dtype=bool)
    idx_blank = np.asarray(blank_frame_idxs, dtype=int)
    blank[idx_blank[idx_blank < n_frames]] = True
    jump, ok = compute_keypoint_jump(pose_df, likelihood_thresh)
    low_likelihood = ~ok
    big_jump = jump > jump_max  # nan (no valid step) compares False
    frame0 = np.arange(n_frames) == 0

    passing = ~(blank | low_likelihood | small_rect | big_jump | frame0)
    me_top = np.zeros(n_frames, dtype=bool)
    if passing.any():
        me_cut = np.percentile(motion_energy[passing], me_percentile)
        me_top = passing & (motion_energy > me_cut)
    black_high = black_frac >= black_max

    table = pd.DataFrame({
        'frame_idx': np.arange(n_frames, dtype=np.int32),
        'blank': blank,
        'low_likelihood': low_likelihood,
        'small_rect': small_rect,
        'big_jump': big_jump,
        'frame0': frame0,
        'me_top': me_top,
        'black_high': black_high,
        'motion_energy': motion_energy.astype(np.float32),
        'black_frac': black_frac.astype(np.float32),
    })
    table['eligible'] = passing & ~me_top & ~black_high
    return table[TABLE_COLUMNS]


def compute_video_eligibility(
    video_path: Path,
    pose_path: Path,
    rect_csv_path: Path,
    blank_frame_idxs: list[int],
    jump_max: float = DEFAULT_JUMP_MAX,
    me_percentile: float = DEFAULT_ME_PERCENTILE,
    black_max: float = DEFAULT_BLACK_MAX,
) -> pd.DataFrame:
    """Compute the eligibility table of one aligned rectangle video.

    Args:
        video_path: aligned rectangle video (`results_dir/rectangles/{video}.mp4`).
        pose_path: tail/neck pose CSV for this video (crop-frame coordinates).
        rect_csv_path: `cuttle inscribe` corner geometry CSV for this video.
        blank_frame_idxs: indices flagged as blank.
        jump_max: see `build_eligibility_table`.
        me_percentile: see `build_eligibility_table`.
        black_max: see `build_eligibility_table`.

    Returns:
        the table from `build_eligibility_table`.

    Raises:
        ValueError: if the pose, rectangle, and video frame counts disagree.
    """
    pose_df = pose.load_pose_predictions(pose_path)
    small_rect = compute_small_rectangle_mask(rect_csv_path, pose_df)
    motion_energy, black_frac = compute_video_pixel_stats(video_path)
    if len(motion_energy) != len(pose_df):
        raise ValueError(
            f'{video_path} has {len(motion_energy)} frames but the pose data has '
            f'{len(pose_df)}'
        )
    return build_eligibility_table(
        pose_df, blank_frame_idxs, small_rect, motion_energy, black_frac,
        jump_max=jump_max, me_percentile=me_percentile, black_max=black_max,
    )


def run_eligibility(
    video_paths: list[Path],
    pose_dir: Path,
    data_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    jump_max: float = DEFAULT_JUMP_MAX,
    me_percentile: float = DEFAULT_ME_PERCENTILE,
    black_max: float = DEFAULT_BLACK_MAX,
    shard: int = 0,
    n_shards: int = 1,
) -> list[Path]:
    """Compute and save the eligibility table of every (sharded) video.

    A video is skipped (with a warning) if its pose CSV or rectangle geometry CSV is missing;
    a missing blank-frames file is treated as "no blank frames", also with a warning. Videos
    whose table already exists are skipped unless `overwrite` is set. Progress is written to
    `progress_shard{i}of{n}.json` in `out_dir`.

    Args:
        video_paths: aligned rectangle videos (`{video}.mp4`, with `{video}.csv` beside each).
        pose_dir: directory of `{video}.csv` pose predictions.
        data_dir: directory holding `{session_id}_{fish_id}_black_frames.txt` files.
        out_dir: directory to write `{video}.parquet` tables into.
        overwrite: recompute tables that already exist.
        jump_max: see `build_eligibility_table`.
        me_percentile: see `build_eligibility_table`.
        black_max: see `build_eligibility_table`.
        shard: this process's shard index.
        n_shards: total number of shards.

    Returns:
        paths of the tables written by this call.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    video_paths = shard_items(video_paths, shard, n_shards)
    progress_file = progress_path(out_dir, shard, n_shards)
    written = []
    for idx, video_path in enumerate(video_paths):
        write_progress(progress_file, idx, len(video_paths), video_path.stem)
        out_path = out_dir / f'{video_path.stem}.parquet'
        if out_path.exists() and not overwrite:
            _logger.info(f'{video_path.stem}: exists, skipping')
            continue
        match = FILENAME_PATTERN.match(video_path.stem)
        if match is None:
            _logger.warning(f'skipping file with unexpected name: {video_path}')
            continue
        pose_path = pose_dir / f'{video_path.stem}.csv'
        rect_csv_path = video_path.with_suffix('.csv')
        if not pose_path.exists() or not rect_csv_path.exists():
            absent = pose_path if not pose_path.exists() else rect_csv_path
            _logger.warning(f'{video_path.stem}: missing {absent}, skipping')
            continue
        blank_path = data_dir / f'{match["session_id"]}_{match["fish_id"]}_black_frames.txt'
        blank_frame_idxs = []
        if blank_path.exists():
            blank_frame_idxs = read_blank_frame_indices(blank_path)
        else:
            _logger.warning(f'{video_path.stem}: no blank-frames file at {blank_path}')

        table = compute_video_eligibility(
            video_path, pose_path, rect_csv_path, blank_frame_idxs,
            jump_max=jump_max, me_percentile=me_percentile, black_max=black_max,
        )
        save_parquet_atomic(out_path, table)
        written.append(out_path)
        _logger.info(
            f'{video_path.stem}: {int(table["eligible"].sum())}/{len(table)} frames eligible'
        )
    write_progress(progress_file, len(video_paths), len(video_paths), None)
    return written
