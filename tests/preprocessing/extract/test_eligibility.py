"""Tests for cuttle_patterns.preprocessing.extract.eligibility."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.extract.eligibility import (
    TABLE_COLUMNS,
    build_eligibility_table,
    compute_keypoint_jump,
    compute_small_rectangle_mask,
    compute_video_eligibility,
    compute_video_pixel_stats,
    run_eligibility,
)
from cuttle_patterns.preprocessing.pose import load_pose_predictions
from tests.preprocessing.extract.conftest import pose_row


def _pose_df(tmp_path: Path, write_pose_csv: Callable, rows: list[dict]) -> pd.DataFrame:
    return load_pose_predictions(write_pose_csv(tmp_path / 'pose.csv', rows))


class TestComputeSmallRectangleMask:
    """Test the function compute_small_rectangle_mask."""

    def test_flags_small_rectangle_only_when_likelihood_ok(
        self, tmp_path: Path, write_pose_csv: Callable, write_rect_csv: Callable,
    ):
        # Arrange: long edge 10, short edge 2
        rect_path = write_rect_csv(tmp_path / 'rect.csv', 3, long_edge=10.0, short_edge=2.0)
        pose_df = _pose_df(tmp_path, write_pose_csv, [
            pose_row(tail=(0, 0), neck=(100, 0)),  # 10 < 0.5 * 100 -> flagged
            pose_row(tail=(0, 0), neck=(15, 0)),  # 10 >= 7.5 -> not flagged
            pose_row(tail=(0, 0), neck=(100, 0), likelihood=0.1),  # low likelihood -> not flagged
        ])

        # Act
        mask = compute_small_rectangle_mask(rect_path, pose_df, ratio_thresh=0.5)

        # Assert
        assert list(mask) == [True, False, False]

    def test_raises_on_length_mismatch(
        self, tmp_path: Path, write_pose_csv: Callable, write_rect_csv: Callable,
    ):
        # Arrange
        rect_path = write_rect_csv(tmp_path / 'rect.csv', 2)
        pose_df = _pose_df(tmp_path, write_pose_csv, [pose_row()])

        # Act & Assert
        with pytest.raises(ValueError, match='expected one row per video frame'):
            compute_small_rectangle_mask(rect_path, pose_df)


class TestComputeKeypointJump:
    """Test the function compute_keypoint_jump."""

    def test_jump_is_max_over_both_steps_and_keypoints(
        self, tmp_path: Path, write_pose_csv: Callable,
    ):
        # Arrange: neck jumps 30 px between frames 1 and 2; tail never moves
        pose_df = _pose_df(tmp_path, write_pose_csv, [
            pose_row(neck=(0, 0)), pose_row(neck=(0, 0)), pose_row(neck=(30, 0)),
            pose_row(neck=(30, 0)),
        ])

        # Act
        jump, ok = compute_keypoint_jump(pose_df)

        # Assert: frames 1 and 2 both touch the 30 px step; frames 0 and 3 do not
        np.testing.assert_allclose(jump, [0, 30, 30, 0])
        assert ok.all()

    def test_steps_with_a_low_confidence_endpoint_are_ignored(
        self, tmp_path: Path, write_pose_csv: Callable,
    ):
        # Arrange: frame 1 is low-confidence and far away
        rows = [pose_row(neck=(0, 0)), pose_row(neck=(500, 0), likelihood=0.1),
                pose_row(neck=(0, 0))]
        pose_df = _pose_df(tmp_path, write_pose_csv, rows)

        # Act
        jump, ok = compute_keypoint_jump(pose_df)

        # Assert
        assert np.isnan(jump).all()
        assert list(ok) == [True, False, True]

    def test_single_frame_has_no_jump(self, tmp_path: Path, write_pose_csv: Callable):
        # Arrange
        pose_df = _pose_df(tmp_path, write_pose_csv, [pose_row()])

        # Act
        jump, _ = compute_keypoint_jump(pose_df)

        # Assert
        assert np.isnan(jump).all()


class TestBuildEligibilityTable:
    """Test the function build_eligibility_table."""

    def _table(self, tmp_path, write_pose_csv, rows, **overrides):
        n = len(rows)
        kwargs = dict(
            blank_frame_idxs=[], small_rect=np.zeros(n, dtype=bool),
            motion_energy=np.arange(n, dtype=np.float32), black_frac=np.zeros(n),
            me_percentile=100.0,
        )
        kwargs.update(overrides)
        return build_eligibility_table(_pose_df(tmp_path, write_pose_csv, rows), **kwargs)

    def test_columns_and_all_good_frames_eligible_except_frame_zero(
        self, tmp_path: Path, write_pose_csv: Callable,
    ):
        # Act
        table = self._table(tmp_path, write_pose_csv, [pose_row()] * 4)

        # Assert
        assert list(table.columns) == TABLE_COLUMNS
        assert list(table['eligible']) == [False, True, True, True]
        assert list(table['frame0']) == [True, False, False, False]

    def test_each_criterion_removes_its_frame(self, tmp_path: Path, write_pose_csv: Callable):
        # Arrange: 8 frames; frame 1 blank, 2 low likelihood, 3 small rect, 5 black
        rows = [pose_row() for _ in range(8)]
        rows[2] = pose_row(likelihood=0.1)
        small_rect = np.zeros(8, dtype=bool)
        small_rect[3] = True
        black = np.zeros(8)
        black[5] = 0.06

        # Act
        table = self._table(
            tmp_path, write_pose_csv, rows, blank_frame_idxs=[1], small_rect=small_rect,
            black_frac=black,
        )

        # Assert
        flags = table[['blank', 'low_likelihood', 'small_rect', 'black_high']]
        assert list(flags.sum()) == [1] * 4
        assert list(table['eligible']) == [False, False, False, False, True, False, True, True]

    def test_big_jump_removes_both_neighbors_of_the_step(
        self, tmp_path: Path, write_pose_csv: Callable,
    ):
        # Arrange: neck jumps 40 px between frames 2 and 3
        rows = [pose_row(neck=(0, 0))] * 3 + [pose_row(neck=(40, 0))] * 3

        # Act
        table = self._table(tmp_path, write_pose_csv, rows, jump_max=25.0)

        # Assert
        assert list(table['big_jump']) == [False, False, True, True, False, False]

    def test_motion_energy_gate_drops_top_percentile_of_passing_frames(
        self, tmp_path: Path, write_pose_csv: Callable,
    ):
        # Arrange: motion energy 0..99 over 100 frames; frame 50 is blank with huge energy,
        # which must not move the percentile cut
        n = 100
        me = np.arange(n, dtype=np.float32)
        me[50] = 1e6

        # Act
        table = self._table(
            tmp_path, write_pose_csv, [pose_row()] * n, blank_frame_idxs=[50],
            motion_energy=me, me_percentile=99.0,
        )

        # Assert: only the single highest passing frame (99) is above the 99th percentile
        assert list(np.flatnonzero(table['me_top'])) == [99]

    def test_no_passing_frames_does_not_error(self, tmp_path: Path, write_pose_csv: Callable):
        # Act
        table = self._table(tmp_path, write_pose_csv, [pose_row(likelihood=0.0)] * 3)

        # Assert
        assert not table['eligible'].any()
        assert not table['me_top'].any()

    def test_raises_on_length_mismatch(self, tmp_path: Path, write_pose_csv: Callable):
        with pytest.raises(ValueError, match='differ in length'):
            self._table(tmp_path, write_pose_csv, [pose_row()] * 3,
                        motion_energy=np.zeros(2, dtype=np.float32))


class TestComputeVideoPixelStats:
    """Test the function compute_video_pixel_stats."""

    def test_black_fraction_and_motion_energy(
        self, tmp_path: Path, make_custom_video: Callable,
    ):
        # Arrange: frame 0 half black / half bright, frames 1-2 fully bright and identical
        half = np.zeros((32, 32, 3), dtype=np.uint8)
        half[:, 16:] = 200
        bright = np.full((32, 32, 3), 200, dtype=np.uint8)
        video = make_custom_video(tmp_path / 'v.mp4', [half, bright, bright])

        # Act
        motion, black = compute_video_pixel_stats(video)

        # Assert
        assert len(motion) == len(black) == 3
        assert black[0] == pytest.approx(0.5, abs=0.05)
        assert black[1] == 0.0
        assert motion[1] > 10 * motion[2]  # the change is between frames 0 and 1
        assert motion[0] == 0.0

    def test_raises_on_missing_video(self, tmp_path: Path):
        with pytest.raises(OSError, match='could not open'):
            compute_video_pixel_stats(tmp_path / 'missing.mp4')


class TestComputeVideoEligibility:
    """Test the function compute_video_eligibility."""

    def test_raises_when_video_and_pose_lengths_differ(
        self, tmp_path: Path, make_video: Callable, write_pose_csv: Callable,
        write_rect_csv: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / 'v.mp4', n_frames=4)
        pose_path = write_pose_csv(tmp_path / 'pose.csv', [pose_row()] * 5)
        rect_path = write_rect_csv(tmp_path / 'rect.csv', 5)

        # Act & Assert
        with pytest.raises(ValueError, match='frames but the pose data has'):
            compute_video_eligibility(video, pose_path, rect_path, [])


class TestRunEligibility:
    """Test the function run_eligibility."""

    def _setup(self, tmp_path, make_custom_video, write_pose_csv, write_rect_csv, constant_frames):
        stem = 'Day2_Tank1_Cuttle1_Resident_Crop'
        rect_dir, pose_dir, data_dir = tmp_path / 'rect', tmp_path / 'pose', tmp_path / 'data'
        for d in (rect_dir, pose_dir, data_dir):
            d.mkdir()
        video = make_custom_video(rect_dir / f'{stem}.mp4', constant_frames([50] * 6))
        write_rect_csv(video.with_suffix('.csv'), 6)
        write_pose_csv(pose_dir / f'{stem}.csv', [pose_row()] * 6)
        (data_dir / 'Day2_Tank1_Cuttle1_Resident_black_frames.txt').write_text('2\n3\n')
        return video, pose_dir, data_dir

    def test_writes_table_with_blank_frames_applied(
        self, tmp_path: Path, make_custom_video: Callable, write_pose_csv: Callable,
        write_rect_csv: Callable, constant_frames: Callable,
    ):
        # Arrange
        video, pose_dir, data_dir = self._setup(
            tmp_path, make_custom_video, write_pose_csv, write_rect_csv, constant_frames,
        )

        # Act
        written = run_eligibility([video], pose_dir, data_dir, tmp_path / 'out')

        # Assert
        table = pd.read_parquet(written[0])
        assert list(table['blank']) == [False, False, True, True, False, False]
        assert (tmp_path / 'out' / 'progress_shard0of1.json').exists()

    def test_skips_existing_unless_overwrite(
        self, tmp_path: Path, make_custom_video: Callable, write_pose_csv: Callable,
        write_rect_csv: Callable, constant_frames: Callable,
    ):
        # Arrange
        video, pose_dir, data_dir = self._setup(
            tmp_path, make_custom_video, write_pose_csv, write_rect_csv, constant_frames,
        )
        out_dir = tmp_path / 'out'
        run_eligibility([video], pose_dir, data_dir, out_dir)

        # Act
        again = run_eligibility([video], pose_dir, data_dir, out_dir)
        forced = run_eligibility([video], pose_dir, data_dir, out_dir, overwrite=True)

        # Assert
        assert again == []
        assert len(forced) == 1

    def test_skips_video_without_pose_file(
        self, tmp_path: Path, make_custom_video: Callable, write_pose_csv: Callable,
        write_rect_csv: Callable, constant_frames: Callable,
    ):
        # Arrange
        video, pose_dir, data_dir = self._setup(
            tmp_path, make_custom_video, write_pose_csv, write_rect_csv, constant_frames,
        )
        (pose_dir / f'{video.stem}.csv').unlink()

        # Act
        written = run_eligibility([video], pose_dir, data_dir, tmp_path / 'out')

        # Assert
        assert written == []

    def test_sharding_splits_videos(
        self, tmp_path: Path, make_custom_video: Callable, write_pose_csv: Callable,
        write_rect_csv: Callable, constant_frames: Callable,
    ):
        # Arrange
        video, pose_dir, data_dir = self._setup(
            tmp_path, make_custom_video, write_pose_csv, write_rect_csv, constant_frames,
        )

        # Act: this video belongs to shard 0 of 2, so shard 1 does nothing
        written = run_eligibility([video], pose_dir, data_dir, tmp_path / 'out', shard=1,
                                  n_shards=2)

        # Assert
        assert written == []
