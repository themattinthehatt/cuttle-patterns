"""Tests for cuttle_patterns.visualization.cluster_frames."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuttle_patterns.visualization.cluster_frames import (
    build_frame_path,
    load_frame_image,
    plot_cluster_grid,
    sample_cluster_frames,
)


class TestSampleClusterFrames:
    """Test the function sample_cluster_frames."""

    def test_sample_cluster_frames_returns_all_rows_when_fewer_than_n_frames(self):
        # Arrange
        cluster_df = pd.DataFrame({'frame_number': [0, 1, 2]})
        rng = np.random.default_rng(0)

        # Act
        sample_df = sample_cluster_frames(cluster_df, n_frames=5, rng=rng)

        # Assert
        assert len(sample_df) == 3

    def test_sample_cluster_frames_samples_exactly_n_frames(self):
        # Arrange
        cluster_df = pd.DataFrame({'frame_number': list(range(10))})
        rng = np.random.default_rng(0)

        # Act
        sample_df = sample_cluster_frames(cluster_df, n_frames=4, rng=rng)

        # Assert
        assert len(sample_df) == 4
        assert set(sample_df['frame_number']).issubset(set(range(10)))

    def test_sample_cluster_frames_is_reproducible_given_a_seeded_generator(self):
        # Arrange
        cluster_df = pd.DataFrame({'frame_number': list(range(10))})

        # Act
        sample_a = sample_cluster_frames(cluster_df, n_frames=4, rng=np.random.default_rng(0))
        sample_b = sample_cluster_frames(cluster_df, n_frames=4, rng=np.random.default_rng(0))

        # Assert
        assert sample_a['frame_number'].tolist() == sample_b['frame_number'].tolist()


class TestBuildFramePath:
    """Test the function build_frame_path."""

    def test_build_frame_path_zero_pads_frame_number(self):
        # Arrange
        results_dir = Path('/results')

        # Act
        frame_path = build_frame_path(results_dir, 'Day1_Tank2_Cuttle1_Resident_Crop', 42)

        # Assert
        assert frame_path == (
            results_dir / 'beast_frames' / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000042.png'
        )


class TestLoadFrameImage:
    """Test the function load_frame_image."""

    def test_load_frame_image_missing_file_returns_none(self, tmp_path: Path):
        # Arrange
        frame_path = tmp_path / 'missing.png'

        # Act
        image = load_frame_image(frame_path)

        # Assert
        assert image is None

    def test_load_frame_image_unreadable_file_returns_none(self, tmp_path: Path):
        # Arrange
        frame_path = tmp_path / 'garbage.png'
        frame_path.write_bytes(b'not a real image')

        # Act
        image = load_frame_image(frame_path)

        # Assert
        assert image is None

    def test_load_frame_image_valid_file_returns_rgb_array(self, tmp_path: Path):
        # Arrange: a BGR image with a distinct value per channel so the RGB swap is
        # unambiguous to check
        frame_path = tmp_path / 'frame.png'
        bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        bgr[:, :, 0] = 10
        bgr[:, :, 1] = 20
        bgr[:, :, 2] = 30
        cv2.imwrite(str(frame_path), bgr)

        # Act
        image = load_frame_image(frame_path)

        # Assert
        assert image.shape == (4, 6, 3)
        assert (image[:, :, 0] == 30).all()
        assert (image[:, :, 1] == 20).all()
        assert (image[:, :, 2] == 10).all()


class TestPlotClusterGrid:
    """Test the function plot_cluster_grid."""

    def test_plot_cluster_grid_missing_images_produces_correct_grid_and_title(
        self,
        tmp_path: Path,
    ):
        # Arrange: no frame images on disk, so every tile falls back to the
        # missing-image path exercised by load_frame_image
        sample_df = pd.DataFrame({
            'video_name': ['vidA', 'vidA', 'vidB'],
            'frame_number': [0, 1, 2],
        })

        # Act
        fig = plot_cluster_grid(sample_df, tmp_path, cluster_label=3, n_cols=2)

        # Assert
        try:
            assert len(fig.axes) == 4  # ceil(3 / 2) rows * 2 cols
            assert fig.get_suptitle() == 'cluster 3 (n=3 shown)'
        finally:
            plt.close(fig)

    def test_plot_cluster_grid_with_real_images(self, tmp_path: Path):
        # Arrange
        frame_dir = tmp_path / 'beast_frames' / 'vidA'
        frame_dir.mkdir(parents=True)
        cv2.imwrite(str(frame_dir / 'img00000000.png'), np.zeros((10, 20, 3), dtype=np.uint8))
        sample_df = pd.DataFrame({'video_name': ['vidA'], 'frame_number': [0]})

        # Act
        fig = plot_cluster_grid(sample_df, tmp_path, cluster_label=0, n_cols=4)

        # Assert
        try:
            assert len(fig.axes) == 4  # ceil(1 / 4) rows * 4 cols
        finally:
            plt.close(fig)
