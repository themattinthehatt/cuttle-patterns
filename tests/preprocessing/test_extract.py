"""Tests for cuttle_patterns.preprocessing.extract."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.align import CORNER_COLUMNS
from cuttle_patterns.preprocessing.extract import (
    build_candidate_frame_idxs,
    build_extraction_manifest,
    compute_filtered_frame_mask,
    compute_small_rectangle_mask,
    extract_video_frames,
    select_frame_idxs_kmeans_restricted,
)
from cuttle_patterns.preprocessing.pose import load_pose_predictions


def _write_pose_csv(path: Path, likelihoods: list[float]) -> Path:
    # tail/neck share the same likelihood per frame; xy values are irrelevant here
    columns = pd.MultiIndex.from_tuples(
        [
            ('m', 'neck', 'x'), ('m', 'neck', 'y'), ('m', 'neck', 'likelihood'),
            ('m', 'tail', 'x'), ('m', 'tail', 'y'), ('m', 'tail', 'likelihood'),
        ],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = [[0.0, 0.0, lk, 0.0, 0.0, lk] for lk in likelihoods]
    pd.DataFrame(data, columns=columns).to_csv(path)
    return path


def _write_pose_csv_xy(path: Path, rows: list[dict]) -> Path:
    # like _write_pose_csv, but with explicit per-frame tail/neck xy, for tests that
    # care about neck-tail distance
    columns = pd.MultiIndex.from_tuples(
        [
            ('m', 'neck', 'x'), ('m', 'neck', 'y'), ('m', 'neck', 'likelihood'),
            ('m', 'tail', 'x'), ('m', 'tail', 'y'), ('m', 'tail', 'likelihood'),
        ],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = [
        [row['neck_x'], row['neck_y'], row['neck_likelihood'],
         row['tail_x'], row['tail_y'], row['tail_likelihood']]
        for row in rows
    ]
    pd.DataFrame(data, columns=columns).to_csv(path)
    return path


def _write_rect_csv(path: Path, corners_per_frame: list[tuple]) -> Path:
    # corners_per_frame: list of (tl, tr, br, bl), each an (x, y) tuple
    rows = []
    for idx, (tl, tr, br, bl) in enumerate(corners_per_frame):
        rows.append({
            'frame_idx': idx,
            'corner_tl_x': tl[0], 'corner_tl_y': tl[1],
            'corner_tr_x': tr[0], 'corner_tr_y': tr[1],
            'corner_br_x': br[0], 'corner_br_y': br[1],
            'corner_bl_x': bl[0], 'corner_bl_y': bl[1],
            'is_interpolated': False,
        })
    pd.DataFrame(rows, columns=['frame_idx', *CORNER_COLUMNS, 'is_interpolated']).to_csv(
        path, index=False,
    )
    return path


def _uniform_rect_corners(
    n_frames: int, long_edge: float = 100.0, short_edge: float = 20.0,
) -> list[tuple]:
    corners = ((0.0, 0.0), (long_edge, 0.0), (long_edge, short_edge), (0.0, short_edge))
    return [corners] * n_frames


def _make_ramp_video(make_custom_video: Callable, path: Path, n_frames: int = 12) -> Path:
    # varying uniform-fill intensity per frame so consecutive frames have nonzero,
    # unequal motion energy
    intensities = [(idx * 37) % 256 for idx in range(n_frames)]
    frames = [np.full((32, 32, 3), val, dtype=np.uint8) for val in intensities]
    return make_custom_video(path, frames)


class TestComputeFilteredFrameMask:
    """Test the function compute_filtered_frame_mask."""

    def test_marks_blank_frames(self):
        # Act
        mask = compute_filtered_frame_mask(5, [1, 3], pose_path=None)

        # Assert
        assert list(mask) == [False, True, False, True, False]

    def test_marks_low_likelihood_frames_from_pose_path(self, tmp_path: Path):
        # Arrange
        pose_path = _write_pose_csv(tmp_path / 'pose.csv', [0.99, 0.2, 0.99])

        # Act
        mask = compute_filtered_frame_mask(3, [], pose_path=pose_path)

        # Assert
        assert list(mask) == [False, True, False]

    def test_combines_blank_and_pose_filters(self, tmp_path: Path):
        # Arrange
        pose_path = _write_pose_csv(tmp_path / 'pose.csv', [0.99, 0.99, 0.2])

        # Act
        mask = compute_filtered_frame_mask(3, [0], pose_path=pose_path)

        # Assert
        assert list(mask) == [True, False, True]

    def test_no_pose_path_only_blank_filter_applies(self):
        # Act
        mask = compute_filtered_frame_mask(4, [2], pose_path=None)

        # Assert
        assert list(mask) == [False, False, True, False]

    def test_marks_small_rectangle_frames_when_rect_csv_path_given(self, tmp_path: Path):
        # Arrange: long edge 10, short edge 2; frame 0's neck-tail distance (100) makes
        # the rectangle too small, frames 1-2's distance (15) doesn't
        rect_path = _write_rect_csv(
            tmp_path / 'rect.csv', _uniform_rect_corners(3, long_edge=10.0, short_edge=2.0),
        )
        pose_path = _write_pose_csv_xy(tmp_path / 'pose.csv', [
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
             'neck_x': 100.0, 'neck_y': 0.0, 'neck_likelihood': 0.99},
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
             'neck_x': 15.0, 'neck_y': 0.0, 'neck_likelihood': 0.99},
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
             'neck_x': 15.0, 'neck_y': 0.0, 'neck_likelihood': 0.99},
        ])

        # Act
        mask = compute_filtered_frame_mask(3, [], pose_path, rect_path)

        # Assert
        assert list(mask) == [True, False, False]

    def test_rect_csv_path_ignored_without_pose_path(self, tmp_path: Path):
        # Arrange: a degenerate (zero-size) rectangle would always be "small" if checked
        rect_path = _write_rect_csv(
            tmp_path / 'rect.csv', _uniform_rect_corners(3, long_edge=0.0, short_edge=0.0),
        )

        # Act
        mask = compute_filtered_frame_mask(3, [], pose_path=None, rect_csv_path=rect_path)

        # Assert
        assert list(mask) == [False, False, False]


class TestComputeSmallRectangleMask:
    """Test the function compute_small_rectangle_mask."""

    def test_flags_small_rectangle_only_when_likelihood_ok(self, tmp_path: Path):
        # Arrange: long edge 10, short edge 2
        rect_path = _write_rect_csv(
            tmp_path / 'rect.csv', _uniform_rect_corners(3, long_edge=10.0, short_edge=2.0),
        )
        pose_path = _write_pose_csv_xy(tmp_path / 'pose.csv', [
            # distance 100: 10 < 0.5 * 100 -> flagged
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
             'neck_x': 100.0, 'neck_y': 0.0, 'neck_likelihood': 0.99},
            # distance 15: 10 is not < 0.5 * 15 (7.5) -> not flagged
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
             'neck_x': 15.0, 'neck_y': 0.0, 'neck_likelihood': 0.99},
            # distance 100 again, but likelihood too low -> not flagged
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.1,
             'neck_x': 100.0, 'neck_y': 0.0, 'neck_likelihood': 0.1},
        ])
        pose_df = load_pose_predictions(pose_path)

        # Act
        mask = compute_small_rectangle_mask(rect_path, pose_df, ratio_thresh=0.5)

        # Assert
        assert list(mask) == [True, False, False]

    def test_raises_on_length_mismatch(self, tmp_path: Path):
        # Arrange
        rect_path = _write_rect_csv(tmp_path / 'rect.csv', _uniform_rect_corners(2))
        pose_path = _write_pose_csv_xy(tmp_path / 'pose.csv', [
            {'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
             'neck_x': 1.0, 'neck_y': 0.0, 'neck_likelihood': 0.99},
        ])
        pose_df = load_pose_predictions(pose_path)

        # Act & Assert
        with pytest.raises(ValueError, match='expected one row per video frame'):
            compute_small_rectangle_mask(rect_path, pose_df)


class TestBuildCandidateFrameIdxs:
    """Test the function build_candidate_frame_idxs."""

    def test_first_and_last_frame_never_candidates(self):
        # Arrange: nothing filtered out
        mask = np.zeros(5, dtype=bool)

        # Act
        candidates = build_candidate_frame_idxs(mask)

        # Assert
        assert list(candidates) == [1, 2, 3]

    def test_isolated_filtered_frame_removes_both_neighbors(self):
        # Arrange: frame 2 filtered out of 5 frames
        mask = np.array([False, False, True, False, False])

        # Act
        candidates = build_candidate_frame_idxs(mask)

        # Assert: frame 1 (next is filtered) and frame 3 (prev is filtered) drop out too
        assert list(candidates) == []

    def test_all_valid_mask_all_interior_frames_candidates(self):
        # Arrange
        mask = np.zeros(6, dtype=bool)

        # Act
        candidates = build_candidate_frame_idxs(mask)

        # Assert
        assert list(candidates) == [1, 2, 3, 4]


class TestSelectFrameIdxsKmeansRestricted:
    """Test the function select_frame_idxs_kmeans_restricted."""

    def test_empty_candidates_returns_empty(self, tmp_path: Path, make_custom_video: Callable):
        # Arrange
        video_path = _make_ramp_video(make_custom_video, tmp_path / 'video.mp4')

        # Act
        selected = select_frame_idxs_kmeans_restricted(
            video_path, np.array([], dtype=int), n_frames_to_select=3,
        )

        # Assert
        assert len(selected) == 0

    def test_selected_idxs_are_subset_of_candidates(
        self, tmp_path: Path, make_custom_video: Callable,
    ):
        # Arrange
        video_path = _make_ramp_video(make_custom_video, tmp_path / 'video.mp4', n_frames=12)
        candidate_idxs = np.array([1, 2, 4, 5, 7, 8, 10])

        # Act
        selected = select_frame_idxs_kmeans_restricted(
            video_path, candidate_idxs, n_frames_to_select=4,
        )

        # Assert
        assert len(selected) == 4
        assert set(selected).issubset(set(candidate_idxs))

    def test_requesting_more_than_available_warns_and_returns_all(
        self, tmp_path: Path, make_custom_video: Callable, caplog: pytest.LogCaptureFixture,
    ):
        # Arrange
        video_path = _make_ramp_video(make_custom_video, tmp_path / 'video.mp4', n_frames=12)
        candidate_idxs = np.array([1, 2, 4, 5, 7])

        # Act
        with caplog.at_level(logging.WARNING):
            selected = select_frame_idxs_kmeans_restricted(
                video_path, candidate_idxs, n_frames_to_select=10,
            )

        # Assert
        assert sorted(selected) == list(candidate_idxs)
        assert 'fewer than the requested' in caplog.text


class TestExtractVideoFrames:
    """Test the function extract_video_frames."""

    def test_writes_expected_files_and_returns_selected_idxs(
        self, tmp_path: Path, make_custom_video: Callable,
    ):
        # Arrange
        video_path = _make_ramp_video(make_custom_video, tmp_path / 'clip.mp4', n_frames=8)
        pose_path = _write_pose_csv(tmp_path / 'clip_pose.csv', [0.99] * 8)
        rect_path = _write_rect_csv(
            tmp_path / 'clip_rect.csv', _uniform_rect_corners(8, long_edge=50.0, short_edge=10.0),
        )
        output_dir = tmp_path / 'beast_frames'

        # Act
        save_dir, selected_idxs = extract_video_frames(
            video_path, output_dir, blank_frame_idxs=[], pose_path=pose_path,
            rect_csv_path=rect_path, frames_per_video=3,
        )

        # Assert
        assert save_dir == output_dir / 'clip'
        assert (save_dir / 'selected_frames.csv').exists()
        assert len(selected_idxs) == 3
        # candidates exclude frame 0 and frame 7 (no both-sided neighbor)
        assert set(selected_idxs).issubset(set(range(1, 7)))
        for idx in selected_idxs:
            assert (save_dir / f'img{str(idx).zfill(8)}.png').exists()


class TestBuildExtractionManifest:
    """Test the function build_extraction_manifest."""

    def test_columns_and_rows(self):
        # Arrange
        rows = [
            {'session_id': 'Day1_Tank2', 'fish_id': 'Cuttle1_Resident', 'frame_idx': 5,
             'image_path': '/tmp/img00000005.png'},
        ]

        # Act
        manifest = build_extraction_manifest(rows)

        # Assert
        assert list(manifest.columns) == ['session_id', 'fish_id', 'frame_idx', 'image_path']
        assert len(manifest) == 1

    def test_empty_rows_gives_empty_dataframe_with_columns(self):
        # Act
        manifest = build_extraction_manifest([])

        # Assert
        assert list(manifest.columns) == ['session_id', 'fish_id', 'frame_idx', 'image_path']
        assert len(manifest) == 0
