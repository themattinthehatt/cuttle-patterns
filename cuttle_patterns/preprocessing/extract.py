"""Select and export representative frames from aligned videos for BEAST training.

Sampling strategy per video (see `docs/PHASES.md` Phase 3):

1. Remove frames that are blank, have any keypoint likelihood below threshold, or have a
   rectangle (from `cuttle inscribe`) whose long edge is too short relative to the
   neck-tail distance — a body clearly longer than the box drawn around it
   (`compute_filtered_frame_mask`; the rectangle check is `compute_small_rectangle_mask`,
   promoted from `scratch/qc_small_rectangles.py` after QC on real data showed it a cheap,
   reliable defect signal).
2. Keep only frames whose immediate neighbors also survived step 1, so every frame BEAST
   uses as temporal context is itself a valid frame (`build_candidate_frame_idxs`).
3. From that candidate set, select diverse anchor frames during movement via motion-energy
   thresholding, PCA, and k-means (`select_frame_idxs_kmeans_restricted`) — a fork of
   `beast.extraction.select_frame_idxs_kmeans` (v1.4.0,
   https://github.com/paninski-lab/beast/blob/v1.4.0/beast/extraction.py), restricted to
   only ever pick anchors from the candidate set. A true restriction isn't possible by
   calling that function directly: its only subsetting knob is `frame_range`, a contiguous
   fractional window, which can't express an arbitrary/non-contiguous allowed-frame set.

`export_frames` and `compute_video_motion_energy` are reused unmodified from
`beast.extraction`/`beast.video`.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from beast.extraction import export_frames
from beast.video import compute_video_motion_energy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cuttle_patterns.preprocessing import pose
from cuttle_patterns.preprocessing.align import CORNER_COLUMNS

_logger = logging.getLogger(__name__)

DEFAULT_FRAMES_PER_VIDEO = 1000
DEFAULT_RESIZE_DIMS = 32
DEFAULT_SMALL_RECT_RATIO_THRESH = 0.5

MANIFEST_RELPATH = Path('manifests') / 'extract.parquet'


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


def compute_filtered_frame_mask(
    n_frames: int,
    blank_frame_idxs: list[int],
    pose_path: Path | None,
    rect_csv_path: Path | None = None,
    likelihood_thresh: float = pose.DEFAULT_LIKELIHOOD_THRESH,
    small_rect_ratio_thresh: float = DEFAULT_SMALL_RECT_RATIO_THRESH,
) -> np.ndarray:
    """Flag frames that are blank, have a low-likelihood keypoint, or a too-small rectangle.

    Args:
        n_frames: total number of frames in the video.
        blank_frame_idxs: indices flagged as blank (e.g. from `ingest.read_blank_frame_indices`).
        pose_path: path to a tail/neck pose CSV (see
            `cuttle_patterns.preprocessing.pose.load_pose_predictions`); if None, only the
            blank-frame criterion is applied.
        rect_csv_path: path to `cuttle inscribe`'s per-frame corner geometry CSV; if None,
            the small-rectangle criterion is skipped even if `pose_path` is given.
        likelihood_thresh: minimum per-keypoint likelihood to trust a frame's prediction.
        small_rect_ratio_thresh: passed through to `compute_small_rectangle_mask`.

    Returns:
        boolean array of shape (n_frames,), True for a filtered-out (disallowed) frame.
    """
    filtered = np.zeros(n_frames, dtype=bool)
    if blank_frame_idxs:
        filtered[np.array(blank_frame_idxs)] = True

    if pose_path is not None:
        pose_df = pose.load_pose_predictions(pose_path)
        _, _, is_low_likelihood = pose.interpolate_pose(
            pose_df, likelihood_thresh=likelihood_thresh,
        )
        filtered |= is_low_likelihood

        if rect_csv_path is not None:
            filtered |= compute_small_rectangle_mask(
                rect_csv_path, pose_df,
                ratio_thresh=small_rect_ratio_thresh, likelihood_thresh=likelihood_thresh,
            )

    return filtered


def build_candidate_frame_idxs(filtered_mask: np.ndarray) -> np.ndarray:
    """Find frames that survived filtering and whose immediate neighbors did too.

    Frame 0 and the last frame are never candidates, since one of their neighbors doesn't
    exist.

    Args:
        filtered_mask: boolean array from `compute_filtered_frame_mask`, True for a
            filtered-out frame.

    Returns:
        sorted array of candidate frame indices.
    """
    is_candidate = ~filtered_mask
    is_candidate[1:] &= ~filtered_mask[:-1]
    is_candidate[:-1] &= ~filtered_mask[1:]
    is_candidate[0] = False
    is_candidate[-1] = False

    return np.where(is_candidate)[0]


def select_frame_idxs_kmeans_restricted(
    video_path: str | Path,
    candidate_idxs: np.ndarray,
    n_frames_to_select: int = DEFAULT_FRAMES_PER_VIDEO,
    resize_dims: int = DEFAULT_RESIZE_DIMS,
) -> np.ndarray:
    """Select distinct frames during movement, restricted to a set of candidate frames.

    Same motion-energy + PCA + k-means algorithm as
    `beast.extraction.select_frame_idxs_kmeans` (v1.4.0), except the high-motion-energy
    percentile, PCA, and k-means steps only ever see `candidate_idxs`, so a filtered-out
    frame can never be selected as an anchor.

    Args:
        video_path: absolute path to the video file.
        candidate_idxs: allowed frame indices to select anchors from (see
            `build_candidate_frame_idxs`).
        n_frames_to_select: maximum number of anchor frames to select; if fewer than this
            many candidates are available, all of them are used instead.
        resize_dims: number of pixels (in both dimensions) to downsample frames to before
            computing motion energy and PCA.

    Returns:
        array of selected frame indices, a subset of `candidate_idxs`.
    """
    if len(candidate_idxs) == 0:
        return np.array([], dtype=int)

    n_select = min(n_frames_to_select, len(candidate_idxs))
    if n_select < n_frames_to_select:
        _logger.warning(
            f'only {len(candidate_idxs)} candidate frames available for {video_path}, '
            f'fewer than the requested {n_frames_to_select}; using all of them'
        )

    _logger.info('computing motion energy...')
    me, frames = compute_video_motion_energy(
        video_file=video_path, resize_dims=resize_dims, return_frames=True,
    )

    # find high-me candidates, defined as those with me larger than the nth percentile me
    # among candidates (take fewer if there are many, same threshold as upstream)
    prctile = 50 if len(candidate_idxs) < 1e5 else 75
    me_candidates = me[candidate_idxs]
    idxs_high_me = candidate_idxs[me_candidates > np.percentile(me_candidates, prctile)]
    if len(idxs_high_me) < n_select:
        idxs_high_me = candidate_idxs

    _logger.info('performing pca over high motion energy candidate frames...')
    pca_obj = PCA(n_components=min(len(idxs_high_me), 32))
    embedding = pca_obj.fit_transform(X=frames[idxs_high_me])
    del frames

    _logger.info('performing kmeans clustering...')
    kmeans_obj = KMeans(n_select, n_init='auto', random_state=0)
    kmeans_obj.fit(embedding)
    centers = kmeans_obj.cluster_centers_.T[None, :]

    # find high-me candidate closest to each cluster center
    dists = np.linalg.norm(embedding[:, :, None] - centers, axis=1)
    idxs_prototypes_ = np.argmin(dists, axis=0)

    return idxs_high_me[idxs_prototypes_]


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    blank_frame_idxs: list[int],
    pose_path: Path,
    rect_csv_path: Path,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
    resize_dims: int = DEFAULT_RESIZE_DIMS,
    likelihood_thresh: float = pose.DEFAULT_LIKELIHOOD_THRESH,
    small_rect_ratio_thresh: float = DEFAULT_SMALL_RECT_RATIO_THRESH,
) -> tuple[Path, np.ndarray]:
    """Select and export representative frames from one aligned video.

    Args:
        video_path: path to the aligned (rectangle) video.
        output_dir: directory under which `{video_path.stem}/` is created.
        blank_frame_idxs: indices flagged as blank for this video.
        pose_path: path to this video's tail/neck pose CSV.
        rect_csv_path: path to this video's `cuttle inscribe` corner geometry CSV.
        frames_per_video: passed through to `select_frame_idxs_kmeans_restricted`.
        resize_dims: passed through to `select_frame_idxs_kmeans_restricted`.
        likelihood_thresh: passed through to `compute_filtered_frame_mask`.
        small_rect_ratio_thresh: passed through to `compute_filtered_frame_mask`.

    Returns:
        (save_dir, selected_idxs): the per-video output directory, and the array of
        selected anchor frame indices (not including exported context frames).

    Raises:
        OSError: if the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    filtered_mask = compute_filtered_frame_mask(
        n_frames, blank_frame_idxs, pose_path, rect_csv_path,
        likelihood_thresh=likelihood_thresh, small_rect_ratio_thresh=small_rect_ratio_thresh,
    )
    candidate_idxs = build_candidate_frame_idxs(filtered_mask)
    selected_idxs = select_frame_idxs_kmeans_restricted(
        video_path, candidate_idxs, n_frames_to_select=frames_per_video, resize_dims=resize_dims,
    )

    n_digits = 8
    extension = 'png'
    save_dir = output_dir / video_path.stem
    export_frames(
        video_file=video_path,
        output_dir=save_dir,
        frame_idxs=selected_idxs,
        context_frames=1,
        n_digits=n_digits,
        extension=extension,
    )

    frames_to_label = np.array([
        f'img{str(idx).zfill(n_digits)}.{extension}' for idx in selected_idxs
    ])
    np.savetxt(save_dir / 'selected_frames.csv', np.sort(frames_to_label), delimiter=',', fmt='%s')

    return save_dir, selected_idxs


def build_extraction_manifest(rows: list[dict]) -> pd.DataFrame:
    """Build the combined frame manifest from per-video selection rows.

    Args:
        rows: dicts with keys session_id, fish_id, frame_idx, image_path — one per
            selected anchor frame.

    Returns:
        one row per selected anchor frame, columns session_id, fish_id, frame_idx,
        image_path.
    """
    return pd.DataFrame(rows, columns=['session_id', 'fish_id', 'frame_idx', 'image_path'])
