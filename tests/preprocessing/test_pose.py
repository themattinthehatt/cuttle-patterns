"""Tests for cuttle_patterns.preprocessing.pose."""

from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.preprocessing.pose import interpolate_pose, load_pose_predictions


def _write_pose_csv(path: Path, rows: list[dict]) -> Path:
    # build a synthetic multi-header (scorer/bodyparts/coords) pose CSV, matching the
    # format written by the pose-estimation pipeline
    columns = pd.MultiIndex.from_tuples(
        [
            ('mymodel', 'neck', 'x'),
            ('mymodel', 'neck', 'y'),
            ('mymodel', 'neck', 'likelihood'),
            ('mymodel', 'tail', 'x'),
            ('mymodel', 'tail', 'y'),
            ('mymodel', 'tail', 'likelihood'),
        ],
        names=['scorer', 'bodyparts', 'coords'],
    )
    data = [
        [
            row['neck_x'], row['neck_y'], row['neck_likelihood'],
            row['tail_x'], row['tail_y'], row['tail_likelihood'],
        ]
        for row in rows
    ]
    pd.DataFrame(data, columns=columns).to_csv(path)
    return path


class TestLoadPosePredictions:
    """Test the function load_pose_predictions."""

    def test_load_pose_predictions_parses_multiheader_csv(self, tmp_path: Path):
        # Arrange
        csv_path = _write_pose_csv(
            tmp_path / 'pose.csv',
            [
                {
                    'neck_x': 10.0, 'neck_y': 20.0, 'neck_likelihood': 0.99,
                    'tail_x': 1.0, 'tail_y': 2.0, 'tail_likelihood': 0.95,
                },
                {
                    'neck_x': 11.0, 'neck_y': 21.0, 'neck_likelihood': 0.98,
                    'tail_x': 1.5, 'tail_y': 2.5, 'tail_likelihood': 0.90,
                },
            ],
        )

        # Act
        df = load_pose_predictions(csv_path)

        # Assert
        assert list(df.index) == [0, 1]
        assert df.loc[0, 'neck_x'] == pytest.approx(10.0)
        assert df.loc[0, 'tail_y'] == pytest.approx(2.0)
        assert df.loc[1, 'neck_likelihood'] == pytest.approx(0.98)


class TestInterpolatePose:
    """Test the function interpolate_pose."""

    def test_interpolate_pose_fills_low_likelihood_frame(self, tmp_path: Path):
        # Arrange: middle frame's neck likelihood is below threshold
        csv_path = _write_pose_csv(
            tmp_path / 'pose.csv',
            [
                {
                    'neck_x': 0.0, 'neck_y': 0.0, 'neck_likelihood': 0.99,
                    'tail_x': 0.0, 'tail_y': 0.0, 'tail_likelihood': 0.99,
                },
                {
                    'neck_x': 999.0, 'neck_y': 999.0, 'neck_likelihood': 0.2,
                    'tail_x': 999.0, 'tail_y': 999.0, 'tail_likelihood': 0.99,
                },
                {
                    'neck_x': 100.0, 'neck_y': 100.0, 'neck_likelihood': 0.99,
                    'tail_x': 100.0, 'tail_y': 100.0, 'tail_likelihood': 0.99,
                },
            ],
        )
        df = load_pose_predictions(csv_path)

        # Act
        tail_xy, neck_xy, is_interpolated = interpolate_pose(df)

        # Assert
        assert list(is_interpolated) == [False, True, False]
        assert neck_xy[1] == pytest.approx([50.0, 50.0])
        assert tail_xy[1] == pytest.approx([50.0, 50.0])

    def test_interpolate_pose_flat_fills_edges(self, tmp_path: Path):
        # Arrange: first and last frames are low-confidence
        csv_path = _write_pose_csv(
            tmp_path / 'pose.csv',
            [
                {
                    'neck_x': 999.0, 'neck_y': 999.0, 'neck_likelihood': 0.1,
                    'tail_x': 999.0, 'tail_y': 999.0, 'tail_likelihood': 0.99,
                },
                {
                    'neck_x': 10.0, 'neck_y': 10.0, 'neck_likelihood': 0.99,
                    'tail_x': 5.0, 'tail_y': 5.0, 'tail_likelihood': 0.99,
                },
                {
                    'neck_x': 999.0, 'neck_y': 999.0, 'neck_likelihood': 0.1,
                    'tail_x': 999.0, 'tail_y': 999.0, 'tail_likelihood': 0.99,
                },
            ],
        )
        df = load_pose_predictions(csv_path)

        # Act
        tail_xy, neck_xy, is_interpolated = interpolate_pose(df)

        # Assert: edges filled with the nearest valid frame's keypoints (flat
        # extrapolation), same convention as align.interpolate_corners
        assert list(is_interpolated) == [True, False, True]
        assert neck_xy[0] == pytest.approx([10.0, 10.0])
        assert neck_xy[2] == pytest.approx([10.0, 10.0])

    def test_interpolate_pose_no_valid_frames_raises(self, tmp_path: Path):
        # Arrange
        csv_path = _write_pose_csv(
            tmp_path / 'pose.csv',
            [
                {
                    'neck_x': 10.0, 'neck_y': 10.0, 'neck_likelihood': 0.1,
                    'tail_x': 5.0, 'tail_y': 5.0, 'tail_likelihood': 0.1,
                },
            ],
        )
        df = load_pose_predictions(csv_path)

        # Act & Assert
        with pytest.raises(ValueError, match='no frame'):
            interpolate_pose(df)
