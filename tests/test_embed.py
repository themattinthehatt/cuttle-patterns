"""Tests for cuttle_patterns.embed."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cuttle_patterns.embed import (
    _get_git_commit,
    _parse_frame_path,
    build_embedder,
    find_frame_paths,
    run_embed,
    write_embedder_output,
)
from cuttle_patterns.embedders.base import Embedder


def _write_frame(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((4, 4, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


class _FakeBackbone:
    """Minimal stand-in for cuttle_patterns.embedders.base.Backbone."""

    key = 'fake_backbone'

    def preprocess(self, frames):
        return frames

    def forward(self, x):
        raise NotImplementedError

    def metadata(self) -> dict:
        return {'fake_backbone_meta': True}


class _FakeReadout:
    """Minimal stand-in for cuttle_patterns.embedders.base.Readout."""

    name = 'fake_readout'

    def __call__(self, tokens):
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return 1

    def metadata(self) -> dict:
        return {'fake_readout_meta': True}


class _FakeEmbedder(Embedder):
    """Embeds a batch as each frame's mean pixel value, for deterministic tests."""

    def __init__(self):
        super().__init__(_FakeBackbone(), _FakeReadout())

    def embed(self, frames: np.ndarray) -> np.ndarray:
        return frames.astype(np.float32).mean(axis=(1, 2, 3)).reshape(-1, 1)


class TestFindFramePaths:
    """Test the function find_frame_paths."""

    def test_find_frame_paths_missing_dir(self, tmp_path: Path):
        # Act & Assert
        with pytest.raises(FileNotFoundError, match='does not exist'):
            find_frame_paths(tmp_path / 'does_not_exist')

    def test_find_frame_paths_empty_dir(self, tmp_path: Path):
        # Arrange
        input_dir = tmp_path / 'beast_frames'
        input_dir.mkdir()

        # Act & Assert
        with pytest.raises(ValueError, match='no frame images found'):
            find_frame_paths(input_dir)

    def test_find_frame_paths_success_sorted_across_videos(self, tmp_path: Path):
        # Arrange
        input_dir = tmp_path / 'beast_frames'
        _write_frame(input_dir / 'Day1_Tank2_Cuttle2_Intruder_Crop' / 'img00000001.png', 1)
        _write_frame(input_dir / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000002.png', 2)
        _write_frame(input_dir / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000001.png', 3)

        # Act
        result = find_frame_paths(input_dir)

        # Assert
        assert [p.parent.name for p in result] == [
            'Day1_Tank2_Cuttle1_Resident_Crop',
            'Day1_Tank2_Cuttle1_Resident_Crop',
            'Day1_Tank2_Cuttle2_Intruder_Crop',
        ]


class TestBuildEmbedder:
    """Test the function build_embedder."""

    def test_build_embedder_unknown_readout_raises_before_loading_backbone(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        def _fail_if_called(*args, **kwargs):
            raise AssertionError('DINOv3Backbone should not be constructed')

        monkeypatch.setattr('cuttle_patterns.embed.DINOv3Backbone', _fail_if_called)

        # Act & Assert
        with pytest.raises(ValueError, match='unknown readout'):
            build_embedder('vitb16', 224, 'not-a-real-readout', device=torch.device('cpu'))


class TestParseFramePath:
    """Test the function _parse_frame_path."""

    def test_parse_frame_path_success(self):
        # Act
        result = _parse_frame_path(
            Path('/x/Day1_Tank2_Cuttle1_Resident_Crop/img00000042.png'),
        )

        # Assert
        assert result == {
            'day': 1, 'tank': 2, 'role': 'Resident',
            'video_name': 'Day1_Tank2_Cuttle1_Resident_Crop', 'frame_number': 42,
        }

    def test_parse_frame_path_bad_filename(self):
        # Act & Assert
        with pytest.raises(ValueError, match='unexpected frame filename'):
            _parse_frame_path(Path('/x/Day1_Tank2_Cuttle1_Resident_Crop/not_a_frame.png'))


class TestRunEmbed:
    """Test the function run_embed."""

    def test_run_embed_sorts_and_aligns_output(self, tmp_path: Path):
        # Arrange
        input_dir = tmp_path / 'beast_frames'
        _write_frame(input_dir / 'Day1_Tank2_Cuttle2_Intruder_Crop' / 'img00000001.png', 30)
        _write_frame(input_dir / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000002.png', 20)
        _write_frame(input_dir / 'Day1_Tank2_Cuttle1_Resident_Crop' / 'img00000001.png', 10)
        frame_paths = find_frame_paths(input_dir)
        embedder = _FakeEmbedder()

        # Act
        embeddings, meta = run_embed(embedder, frame_paths, batch_size=2)

        # Assert
        assert list(meta.columns) == ['video_name', 'day', 'tank', 'role', 'frame_number']
        assert meta['frame_number'].tolist() == [1, 2, 1]
        assert meta['video_name'].tolist() == [
            'Day1_Tank2_Cuttle1_Resident_Crop',
            'Day1_Tank2_Cuttle1_Resident_Crop',
            'Day1_Tank2_Cuttle2_Intruder_Crop',
        ]
        np.testing.assert_allclose(embeddings[:, 0], [10.0, 20.0, 30.0])


class TestGetGitCommit:
    """Test the function _get_git_commit."""

    def test_get_git_commit_inside_repo(self):
        # Act
        result = _get_git_commit(Path(__file__).resolve().parent)

        # Assert
        assert result is not None
        assert len(result) == 40

    def test_get_git_commit_outside_repo(self, tmp_path: Path):
        # Act
        result = _get_git_commit(tmp_path)

        # Assert
        assert result is None


class TestWriteEmbedderOutput:
    """Test the function write_embedder_output."""

    def test_write_embedder_output_writes_expected_files(self, tmp_path: Path):
        # Arrange
        embeddings = np.array([[1.0], [2.0]], dtype=np.float32)
        meta = pd.DataFrame({
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'] * 2,
            'day': [1, 1], 'tank': [2, 2], 'role': ['Resident', 'Resident'],
            'frame_number': [1, 2],
        })
        embedder = _FakeEmbedder()
        model_dir = tmp_path / 'beast_models' / 'fake-model'

        # Act
        latents_dir = write_embedder_output(embeddings, meta, embedder, model_dir, 'beast_frames')

        # Assert
        with (model_dir / 'config.yaml').open() as f:
            config = yaml.safe_load(f)
        assert config['model']['model_class'] == 'embedder'
        model_params = config['model']['model_params']
        assert model_params['backbone'] == 'fake_backbone'
        assert model_params['readout'] == 'fake_readout'
        assert model_params['embed_dim'] == 1
        assert model_params['fake_backbone_meta'] is True
        assert model_params['fake_readout_meta'] is True

        assert latents_dir == model_dir / 'image_predictions' / 'beast_frames' / 'latents'
        np.testing.assert_array_equal(np.load(latents_dir / 'embeddings.npy'), embeddings)
        pd.testing.assert_frame_equal(pd.read_parquet(latents_dir / 'manifest.parquet'), meta)
