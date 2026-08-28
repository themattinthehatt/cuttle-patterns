"""Tests for cuttle_patterns.preprocessing.extract."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.extract import (
    build_candidate_frame_idxs,
    build_extraction_manifest,
    compute_filtered_frame_mask,
    extract_video_frames,
    select_frame_idxs_kmeans_restricted,
)


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
        pose_path = _write_pose_csv(tmp_path / 'clip.csv', [0.99] * 8)
        output_dir = tmp_path / 'beast_frames'

        # Act
        save_dir, selected_idxs = extract_video_frames(
            video_path, output_dir, blank_frame_idxs=[], pose_path=pose_path,
            frames_per_video=3,
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
