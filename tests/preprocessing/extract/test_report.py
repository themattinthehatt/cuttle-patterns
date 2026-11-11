"""Tests for cuttle_patterns.preprocessing.extract.report (CPU, synthetic data)."""

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from cuttle_patterns.preprocessing.extract.kmeans import (
    normalize,
    run_kmeans_pooled,
    run_kmeans_video,
)
from cuttle_patterns.preprocessing.extract.report import (
    format_report,
    load_pools,
    make_sheet,
    near_fraction,
    nearest_distance,
    pool_metrics,
    run_report,
)
from tests.preprocessing.extract.conftest import DIM, make_pipeline_dirs

CPU = torch.device('cpu')
CLASSES = ['A', 'B']


class TestNearestDistance:
    """Test the function nearest_distance."""

    def test_nearest_distance_is_cosine_distance_to_closest_reference(self):
        # Arrange
        refs = normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        queries = normalize(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))

        # Act
        dist = nearest_distance(queries, refs)

        # Assert
        np.testing.assert_allclose(dist.numpy(), [0.0, 1 - np.sqrt(0.5)], atol=1e-6)

    def test_nearest_distance_exclude_self_ignores_own_row(self):
        # Arrange: two identical points; without exclusion each is at distance 0 from itself
        x = normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

        # Act
        dist = nearest_distance(x, x, exclude_self=True)

        # Assert: the point at (0, 1) is at distance 1 from both others
        np.testing.assert_allclose(dist.numpy(), [0.0, 0.0, 1.0], atol=1e-6)


class TestPoolMetrics:
    """Test the function pool_metrics."""

    def test_pool_metrics_coverage_zero_when_selection_is_the_sample(self):
        # Arrange: two states, 3 points on one axis and 1 on the other
        sample = normalize(torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]]))
        centroids = normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        sizes = np.array([3, 1])

        # Act
        metrics = pool_metrics(sample, sample, centroids, sizes, rare=np.array([1]))

        # Assert
        assert metrics['cov_mean'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['dens_ratio'] == pytest.approx(1.0)
        assert metrics['rare_ratio'] == pytest.approx(1.0)

    def test_pool_metrics_selection_only_from_dense_state_has_high_dens_ratio(self):
        # Arrange
        sample = normalize(torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]]))
        selection = sample[:2]
        centroids = normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

        # Act
        metrics = pool_metrics(selection, sample, centroids, np.array([3, 1]), np.array([1]))

        # Assert
        assert metrics['dens_ratio'] == pytest.approx(1 / 0.75)
        assert metrics['rare_ratio'] == 0.0


class TestNearFraction:
    """Test the function near_fraction."""

    def test_near_fraction_counts_frames_with_a_neighbor_within_gap(self):
        # Arrange: video 0 has frames 0, 5, 100; video 1 has frame 3 alone
        df = pd.DataFrame({'video': [0, 0, 0, 1], 'frame_idx': [0, 5, 100, 3]})

        # Act & Assert
        assert near_fraction(df, 10) == pytest.approx(0.5)
        assert near_fraction(df, 1000) == pytest.approx(0.75)


class TestMakeSheet:
    """Test the function make_sheet."""

    def test_make_sheet_writes_image_with_grid_of_tiles(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        make_video(tmp_path / 'Day2_Tank1_Cuttle1_Resident_Crop.mp4', n_frames=6)
        picks = pd.DataFrame({
            'video': ['Day2_Tank1_Cuttle1_Resident_Crop'] * 3, 'frame_idx': [0, 2, 4],
            'confidence': [0.5, 0.6, 0.7],
        })

        # Act
        make_sheet(picks, tmp_path, tmp_path / 'sheet.jpg')

        # Assert: one row of 10 tiles
        assert cv2.imread(str(tmp_path / 'sheet.jpg')).shape == (128, 2560, 3)


def _selection(tmp_path: Path) -> dict:
    d = make_pipeline_dirs(tmp_path)
    run_kmeans_video(
        d['stems'], d['elig_dir'], d['clf_dir'], d['emb_dir'], tmp_path / 'km', CLASSES, 300,
        classifier_name='clf', embedding_name='emb', device=CPU,
    )
    run_kmeans_pooled(tmp_path / 'km', tmp_path / 'sel', picks_per_group=5, device=CPU)
    return d


class TestLoadPools:
    """Test the function load_pools."""

    def test_load_pools_groups_eligible_frames_by_class(self, tmp_path: Path):
        # Arrange
        d = make_pipeline_dirs(tmp_path, n_videos=2)

        # Act
        pools = load_pools(d['stems'], d['elig_dir'], d['clf_dir'], d['emb_dir'], CLASSES)

        # Assert
        total = sum(len(p['frame']) for p in pools.values())
        assert total == 2 * 320
        assert pools['A']['emb'].shape[1] == DIM
        assert set(np.unique(pools['A']['video'])) == {0, 1}


class TestRunReport:
    """Test the function run_report."""

    def test_run_report_writes_files_and_selection_beats_random_coverage(
        self, tmp_path: Path,
    ):
        # Arrange
        d = _selection(tmp_path)

        # Act
        report = run_report(
            tmp_path / 'sel', d['elig_dir'], d['clf_dir'], d['emb_dir'], tmp_path,
            sample=500, sheets=False, device=CPU,
        )

        # Assert
        assert (tmp_path / 'sel' / 'report.md').exists()
        assert json.loads((tmp_path / 'sel' / 'report.json').read_text())['n_videos'] == 3
        for cls in CLASSES:
            entry = report['classes'][cls]
            assert entry['selected'] == entry['quota']
            assert 0 < entry['top5_video_share'] <= 1.0
            selected, random = entry['selected_metrics'], entry['random_metrics']
            assert selected['cov_p999'] <= random['cov_p999'] + 0.05
        assert 'class | quota' in (tmp_path / 'sel' / 'report.md').read_text()

    def test_run_report_raises_when_manifest_has_ineligible_frame(self, tmp_path: Path):
        # Arrange: mark one selected frame ineligible after selection
        d = _selection(tmp_path)
        manifest = pd.read_parquet(tmp_path / 'sel' / 'manifest.parquet')
        row = manifest.iloc[0]
        elig_path = d['elig_dir'] / f'{row["video"]}.parquet'
        elig = pd.read_parquet(elig_path)
        elig.loc[elig['frame_idx'] == row['frame_idx'], 'eligible'] = False
        elig.to_parquet(elig_path)

        # Act & Assert
        with pytest.raises(ValueError, match='not eligible'):
            run_report(tmp_path / 'sel', d['elig_dir'], d['clf_dir'], d['emb_dir'], tmp_path,
                       sample=500, sheets=False, device=CPU)


class TestFormatReport:
    """Test the function format_report."""

    def test_format_report_has_header_and_one_row_per_class(self, tmp_path: Path):
        # Arrange
        d = _selection(tmp_path)
        report = run_report(tmp_path / 'sel', d['elig_dir'], d['clf_dir'], d['emb_dir'], tmp_path,
                            sample=500, sheets=False, device=CPU)

        # Act
        text = format_report(report)

        # Assert
        lines = text.strip().splitlines()
        assert lines[3].startswith('class | quota')
        assert len(lines) == 4 + len(CLASSES)
