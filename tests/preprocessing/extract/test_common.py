"""Tests for cuttle_patterns.preprocessing.extract.common."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.extract.common import (
    atomic_write,
    get_day,
    list_videos,
    prefetch_batches,
    progress_path,
    save_npy_atomic,
    save_parquet_atomic,
    shard_items,
    video_frame_count,
    write_progress,
)


class TestListVideos:
    """Test the function list_videos."""

    def test_list_videos_matches_crop_videos_and_skips_overlays(self, tmp_path: Path):
        # Arrange
        for name in ['Day1_Tank1_Cuttle1_Resident_Crop.mp4',
                     'Day1_Tank1_Cuttle1_Resident_Crop_overlay.mp4',
                     'Day1_Tank1_Cuttle1_Resident_Crop.csv', 'notes.mp4']:
            (tmp_path / name).touch()

        # Act
        videos = list_videos(tmp_path)

        # Assert
        assert [p.name for p in videos] == ['Day1_Tank1_Cuttle1_Resident_Crop.mp4']

    def test_list_videos_filters_by_stem(self, tmp_path: Path):
        # Arrange
        names = ['Day1_Tank1_Cuttle1_Resident_Crop.mp4', 'Day2_Tank1_Cuttle1_Resident_Crop.mp4']
        for name in names:
            (tmp_path / name).touch()

        # Act
        videos = list_videos(tmp_path, ['Day2_Tank1_Cuttle1_Resident_Crop'])

        # Assert
        assert [p.stem for p in videos] == ['Day2_Tank1_Cuttle1_Resident_Crop']


class TestGetDay:
    """Test the function get_day."""

    def test_get_day_parses_day_number(self):
        assert get_day('Day12_Tank3_Cuttle2_Resident_Crop') == 12

    def test_get_day_raises_on_unexpected_name(self):
        with pytest.raises(ValueError, match='cannot parse a day'):
            get_day('video_Crop')


class TestShardItems:
    """Test the function shard_items."""

    def test_shard_items_partitions_all_items(self):
        # Arrange
        items = list(range(7))

        # Act
        shards = [shard_items(items, i, 3) for i in range(3)]

        # Assert
        assert sorted(x for s in shards for x in s) == items
        assert shards[0] == [0, 3, 6]

    def test_shard_items_raises_on_bad_shard(self):
        with pytest.raises(ValueError, match='shard must be in'):
            shard_items([1, 2], 2, 2)


class TestAtomicWrites:
    """Test the atomic file writers."""

    def test_atomic_write_leaves_no_temporary_file(self, tmp_path: Path):
        # Act
        atomic_write(tmp_path / 'a.txt', lambda tmp: tmp.write_text('hi'))

        # Assert
        assert (tmp_path / 'a.txt').read_text() == 'hi'
        assert [p.name for p in tmp_path.iterdir()] == ['a.txt']

    def test_atomic_write_failure_leaves_no_final_file(self, tmp_path: Path):
        # Arrange
        def boom(tmp: Path) -> None:
            tmp.write_text('partial')
            raise RuntimeError('boom')

        # Act & Assert
        with pytest.raises(RuntimeError):
            atomic_write(tmp_path / 'a.txt', boom)
        assert not (tmp_path / 'a.txt').exists()

    def test_save_npy_atomic_roundtrip_without_extension_suffix(self, tmp_path: Path):
        # Arrange
        array = np.arange(6, dtype=np.float16).reshape(2, 3)

        # Act
        save_npy_atomic(tmp_path / 'x.npy', array)

        # Assert
        np.testing.assert_array_equal(np.load(tmp_path / 'x.npy'), array)
        assert [p.name for p in tmp_path.iterdir()] == ['x.npy']

    def test_save_parquet_atomic_roundtrip(self, tmp_path: Path):
        # Arrange
        df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

        # Act
        save_parquet_atomic(tmp_path / 'x.parquet', df)

        # Assert
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / 'x.parquet'), df)


class TestProgress:
    """Test the progress-file helpers."""

    def test_write_progress_writes_json(self, tmp_path: Path):
        # Arrange
        path = progress_path(tmp_path, 1, 3)

        # Act
        write_progress(path, 2, 10, 'vid')

        # Assert
        assert path.name == 'progress_shard1of3.json'
        assert json.loads(path.read_text()) == {'done': 2, 'total': 10, 'current': 'vid'}


class TestVideoFrameCount:
    """Test the function video_frame_count."""

    def test_video_frame_count_matches_video(self, tmp_path: Path, make_video: Callable):
        assert video_frame_count(make_video(tmp_path / 'v.mp4', n_frames=7)) == 7

    def test_video_frame_count_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(OSError, match='could not open'):
            video_frame_count(tmp_path / 'missing.mp4')


class TestPrefetchBatches:
    """Test the function prefetch_batches."""

    def test_prefetch_batches_yields_all_frames_in_batches(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / 'v.mp4', n_frames=7, width=16, height=8)

        # Act
        batches = list(prefetch_batches(video, batch_size=3))

        # Assert
        assert [len(b) for b in batches] == [3, 3, 1]
        assert batches[0].shape == (3, 8, 16, 3)

    def test_prefetch_batches_applies_transform(self, tmp_path: Path, make_video: Callable):
        # Arrange
        video = make_video(tmp_path / 'v.mp4', n_frames=4)

        # Act
        batches = list(prefetch_batches(video, 2, transform=lambda b: b.mean(axis=(1, 2, 3))))

        # Assert
        assert [b.shape for b in batches] == [(2,), (2,)]

    def test_prefetch_batches_raises_on_missing_video(self, tmp_path: Path):
        with pytest.raises(OSError, match='could not open'):
            list(prefetch_batches(tmp_path / 'missing.mp4', 2))
