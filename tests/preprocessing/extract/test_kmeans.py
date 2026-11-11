"""Tests for cuttle_patterns.preprocessing.extract.kmeans (CPU, synthetic embeddings)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cuttle_patterns.preprocessing.extract.kmeans import (
    assign,
    cluster_block,
    compute_quotas,
    load_eligible_labels,
    normalize,
    run_kmeans_pooled,
    run_kmeans_video,
    segment_sums,
    select_pooled,
    spherical_kmeans,
    water_fill,
)
from tests.preprocessing.extract.conftest import DIM, make_pipeline_dirs

CPU = torch.device('cpu')
CLASSES = ['A', 'B']


def _blobs(n_per: int, centers: np.ndarray, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    pts = np.concatenate([c + 0.05 * rng.standard_normal((n_per, centers.shape[1]))
                          for c in centers])
    return normalize(torch.from_numpy(pts.astype(np.float32)))


class TestNormalize:
    """Test the function normalize."""

    def test_normalize_gives_unit_rows(self):
        x = normalize(torch.tensor([[3.0, 4.0], [0.0, 2.0]]))
        np.testing.assert_allclose(x.norm(dim=1).numpy(), 1.0, atol=1e-6)


class TestAssign:
    """Test the function assign."""

    def test_assign_picks_highest_dot_product(self):
        # Arrange
        centroids = torch.eye(3)
        x = torch.tensor([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8]])

        # Act & Assert
        assert assign(x, centroids, chunk=1).tolist() == [0, 2]


class TestSegmentSums:
    """Test the function segment_sums."""

    def test_segment_sums_matches_index_add_and_handles_empty_clusters(self):
        # Arrange
        torch.manual_seed(0)
        x = torch.randn(50, 4)
        labels = torch.randint(0, 5, (50,))
        labels[labels == 3] = 4  # cluster 3 is empty
        counts = torch.bincount(labels, minlength=6)

        # Act
        sums = segment_sums(x, labels, counts)

        # Assert
        expected = torch.zeros(6, 4).index_add_(0, labels, x)
        torch.testing.assert_close(sums, expected, atol=1e-5, rtol=1e-5)
        assert sums[3].abs().sum() == 0


class TestSphericalKmeans:
    """Test the function spherical_kmeans."""

    def test_spherical_kmeans_recovers_separated_blobs(self):
        # Arrange
        x = _blobs(40, np.eye(DIM)[:3])

        # Act
        labels, centroids = spherical_kmeans(x, 3, seed=0)

        # Assert: each blob of 40 consecutive points shares one label
        labels = labels.numpy().reshape(3, 40)
        assert all(len(set(row)) == 1 for row in labels)
        assert len({row[0] for row in labels}) == 3
        np.testing.assert_allclose(centroids.norm(dim=1).numpy(), 1.0, atol=1e-5)

    def test_spherical_kmeans_is_deterministic_for_a_seed(self):
        # Arrange
        x = _blobs(50, np.eye(DIM)[:4], seed=1)

        # Act
        a, _ = spherical_kmeans(x, 6, seed=3)
        b, _ = spherical_kmeans(x, 6, seed=3)

        # Assert
        assert torch.equal(a, b)


class TestWaterFill:
    """Test the function water_fill."""

    def test_water_fill_equal_split_when_groups_are_large(self):
        assert list(water_fill(np.array([10, 10, 10]), 9)) == [3, 3, 3]

    def test_water_fill_small_groups_give_everything_and_surplus_goes_to_others(self):
        # Arrange: group 0 only has 1 item
        quota = water_fill(np.array([1, 10, 10]), 12)

        # Assert
        assert quota[0] == 1
        assert quota.sum() == 12
        assert quota[1] + quota[2] == 11

    def test_water_fill_clips_total_to_availability(self):
        assert water_fill(np.array([2, 3]), 100).sum() == 5


class TestComputeQuotas:
    """Test the function compute_quotas."""

    def test_square_root_quotas_sum_to_budget_and_flatten_the_mix(self):
        # Arrange
        counts = np.array([10000, 2500, 400])

        # Act
        quota = compute_quotas(counts, 1000, 0.5)

        # Assert: shares proportional to 100 : 50 : 20
        assert quota.sum() == 1000
        np.testing.assert_allclose(quota / 1000, np.array([100, 50, 20]) / 170, atol=0.01)

    def test_uniform_power_gives_equal_quotas(self):
        assert list(compute_quotas(np.array([500, 900, 700]), 300, 0.0)) == [100, 100, 100]

    def test_quota_is_capped_at_class_availability_and_surplus_redistributed(self):
        # Arrange: class 2 only has 10 frames, uniform wants 100
        quota = compute_quotas(np.array([500, 500, 10]), 300, 0.0)

        # Assert
        assert quota[2] == 10
        assert quota.sum() == 300

    def test_budget_larger_than_pool_returns_everything(self):
        assert list(compute_quotas(np.array([3, 4]), 100, 0.5)) == [3, 4]


class TestClusterBlock:
    """Test the function cluster_block."""

    def test_cluster_block_with_k_at_least_n_keeps_every_frame(self):
        # Arrange
        x = _blobs(3, np.eye(DIM)[:2])

        # Act
        centroids, labels = cluster_block(x, 10, seed=0)

        # Assert
        assert len(centroids) == 6
        assert list(labels) == list(range(6))

    def test_cluster_block_labels_are_contiguous_over_centroids(self):
        # Arrange
        x = _blobs(30, np.eye(DIM)[:3])

        # Act
        centroids, labels = cluster_block(x, 5, seed=0)

        # Assert
        assert set(labels) == set(range(len(centroids)))


def _stage1(tmp_path: Path, d: dict, **overrides) -> dict:
    kwargs = dict(
        stems=d['stems'], elig_dir=d['elig_dir'], clf_dir=d['clf_dir'], emb_dir=d['emb_dir'],
        out_dir=tmp_path / 'km', classes=CLASSES, budget=300, classifier_name='clf',
        embedding_name='emb', device=CPU,
    )
    kwargs.update(overrides)
    return run_kmeans_video(**kwargs)


class TestLoadEligibleLabels:
    """Test the function load_eligible_labels."""

    def test_load_eligible_labels_keeps_only_eligible_rows(self, tmp_path: Path):
        # Arrange
        d = make_pipeline_dirs(tmp_path, n_videos=1, n_frames=10)
        stem = d['stems'][0]

        # Act
        table = load_eligible_labels(d['elig_dir'] / f'{stem}.parquet',
                                     d['clf_dir'] / f'{stem}.parquet')

        # Assert
        assert list(table['frame_idx']) == [1, 2, 3, 4, 6, 7, 8, 9]
        assert list(table.columns) == ['frame_idx', 'predicted_pattern', 'confidence']

    def test_load_eligible_labels_raises_on_length_mismatch(self, tmp_path: Path):
        # Arrange
        d = make_pipeline_dirs(tmp_path, n_videos=1, n_frames=10)
        stem = d['stems'][0]
        pd.read_parquet(d['clf_dir'] / f'{stem}.parquet').iloc[:5].to_parquet(
            d['clf_dir'] / f'{stem}.parquet')

        # Act & Assert
        with pytest.raises(ValueError, match='eligibility has 10 frames'):
            load_eligible_labels(d['elig_dir'] / f'{stem}.parquet',
                                 d['clf_dir'] / f'{stem}.parquet')


class TestRunKmeansVideo:
    """Test the function run_kmeans_video."""

    def test_run_kmeans_video_writes_config_with_quotas_and_per_video_files(
        self, tmp_path: Path,
    ):
        # Arrange
        d = make_pipeline_dirs(tmp_path)

        # Act
        config = _stage1(tmp_path, d)

        # Assert
        assert sum(config['quotas']) == 300
        assert sum(config['counts']) == 3 * 320  # 400 frames, a fifth ineligible
        assert sorted(p.stem for p in (tmp_path / 'km').glob('*.npz')) == sorted(d['stems'])
        saved = yaml.safe_load((tmp_path / 'km' / 'config.yaml').read_text())
        assert saved['classifier_name'] == 'clf'
        with np.load(tmp_path / 'km' / f'{d["stems"][0]}.npz') as z:
            assert len(z['labels_0']) == len(z['frames_0'])
            assert z['centroids_0'].shape[1] == DIM

    def test_run_kmeans_video_is_resumable_and_raises_on_changed_settings(
        self, tmp_path: Path,
    ):
        # Arrange
        d = make_pipeline_dirs(tmp_path)
        _stage1(tmp_path, d)

        # Act & Assert: the same settings pass, new quotas need --overwrite
        _stage1(tmp_path, d)
        with pytest.raises(ValueError, match='different settings'):
            _stage1(tmp_path, d, budget=200)
        _stage1(tmp_path, d, budget=200, overwrite=True)

    def test_run_kmeans_video_is_deterministic(self, tmp_path: Path):
        # Arrange
        d = make_pipeline_dirs(tmp_path)
        _stage1(tmp_path, d, out_dir=tmp_path / 'a')
        _stage1(tmp_path, d, out_dir=tmp_path / 'b')
        stem = d['stems'][0]

        # Act
        a = np.load(tmp_path / 'a' / f'{stem}.npz')
        b = np.load(tmp_path / 'b' / f'{stem}.npz')

        # Assert
        assert np.array_equal(a['labels_0'], b['labels_0'])
        assert np.array_equal(a['centroids_1'], b['centroids_1'])


class TestSelectPooled:
    """Test the function select_pooled."""

    def test_select_pooled_returns_unique_centroids_spread_over_groups(self):
        # Arrange: 4 well separated blobs of 50 centroids; ask for 20 with 5 per group
        x = _blobs(50, np.eye(DIM)[:4])

        # Act
        chosen, groups = select_pooled(x, quota=20, picks_per_group=5, seed=0)

        # Assert
        assert len(chosen) == len(set(chosen)) == 20
        assert len(set(groups)) == 4
        assert all((groups == g).sum() == 5 for g in set(groups))

    def test_select_pooled_quota_larger_than_pool_returns_everything(self):
        # Arrange
        x = _blobs(3, np.eye(DIM)[:2])

        # Act
        chosen, _ = select_pooled(x, quota=100, picks_per_group=2, seed=0)

        # Assert
        assert sorted(chosen) == list(range(6))


class TestRunKmeansPooled:
    """Test the function run_kmeans_pooled."""

    def test_run_kmeans_pooled_hits_quotas_and_selects_only_eligible_frames_with_right_labels(
        self, tmp_path: Path,
    ):
        # Arrange
        d = make_pipeline_dirs(tmp_path)
        config = _stage1(tmp_path, d)

        # Act
        manifest = run_kmeans_pooled(tmp_path / 'km', tmp_path / 'sel', picks_per_group=5,
                                     device=CPU)

        # Assert
        counts = manifest['predicted_pattern'].value_counts().to_dict()
        assert counts == dict(zip(CLASSES, config['quotas'], strict=True))
        assert not manifest.duplicated(['video', 'frame_idx']).any()
        for stem, group in manifest.groupby('video'):
            elig = pd.read_parquet(d['elig_dir'] / f'{stem}.parquet').set_index('frame_idx')
            clf = pd.read_parquet(d['clf_dir'] / f'{stem}.parquet').set_index('frame_idx')
            assert elig.loc[group['frame_idx'], 'eligible'].all()
            assert (clf.loc[group['frame_idx'], 'predicted_pattern'].to_numpy()
                    == group['predicted_pattern'].to_numpy()).all()
        assert (tmp_path / 'sel' / 'summary.json').exists()

    def test_run_kmeans_pooled_is_deterministic(self, tmp_path: Path):
        # Arrange
        d = make_pipeline_dirs(tmp_path)
        _stage1(tmp_path, d)

        # Act
        a = run_kmeans_pooled(tmp_path / 'km', tmp_path / 'a', seed=4, device=CPU)
        b = run_kmeans_pooled(tmp_path / 'km', tmp_path / 'b', seed=4, device=CPU)

        # Assert
        pd.testing.assert_frame_equal(a, b)

    def test_run_kmeans_pooled_raises_without_stage1_output(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match='run extract-kmeans-video'):
            run_kmeans_pooled(tmp_path / 'missing', tmp_path / 'sel', device=CPU)
