"""Tests for cuttle_patterns.preprocessing.extract.dino_cls (with a fake backbone, on CPU)."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from cuttle_patterns.embedders.base import TokenOutput
from cuttle_patterns.preprocessing.extract import dino_cls
from cuttle_patterns.preprocessing.extract.dino_cls import (
    embed_video,
    embedding_dir_name,
    run_dino_cls,
)

DIM = 6


class _FakeBackbone:
    """Stand-in for DINOv3Backbone: CLS = per-channel mean of the input, no model download."""

    key = 'fake_224'
    resolution = 32
    device = torch.device('cpu')

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x: torch.Tensor) -> TokenOutput:
        cls = x.mean(dim=(2, 3)).repeat(1, DIM // 3)
        return TokenOutput(cls=cls, patches=cls[:, None, :], grid_hw=(1, 1))

    def metadata(self) -> dict:
        return {'hf_model_id': 'fake'}


@pytest.fixture(autouse=True)
def _use_fake_backbone(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dino_cls, 'DINOv3Backbone', _FakeBackbone)


@pytest.mark.filterwarnings('ignore:In CPU autocast')
class TestEmbedVideo:
    """Test the function embed_video."""

    def test_embed_video_returns_one_float16_row_per_frame(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / 'v.mp4', n_frames=5)

        # Act
        features = embed_video(video, _FakeBackbone(), batch_size=2)

        # Assert
        assert features.shape == (5, DIM)
        assert features.dtype == np.float16

    def test_embed_video_preserves_frame_order(
        self, tmp_path: Path, make_custom_video: Callable, constant_frames: Callable,
    ):
        # Arrange: frames get brighter over time, so the fake CLS (channel mean) must increase
        video = make_custom_video(tmp_path / 'v.mp4', constant_frames([20, 90, 160, 230]))

        # Act
        features = embed_video(video, _FakeBackbone(), batch_size=3)

        # Assert
        assert (np.diff(features[:, 0].astype(np.float32)) > 0).all()


@pytest.mark.filterwarnings('ignore:In CPU autocast')
class TestRunDinoCls:
    """Test the function run_dino_cls."""

    def test_run_dino_cls_writes_array_config_and_progress(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / 'Day1_Tank1_Cuttle1_Resident_Crop.mp4', n_frames=4)
        out_dir = tmp_path / 'out'

        # Act
        written = run_dino_cls([video], out_dir, device=torch.device('cpu'))

        # Assert
        assert np.load(written[0]).shape == (4, DIM)
        assert yaml.safe_load((out_dir / 'config.yaml').read_text())['backbone'] == 'fake_224'
        assert json.loads((out_dir / 'progress_shard0of1.json').read_text())['done'] == 1

    def test_run_dino_cls_skips_finished_videos_unless_overwrite(
        self, tmp_path: Path, make_video: Callable,
    ):
        # Arrange
        video = make_video(tmp_path / 'Day1_Tank1_Cuttle1_Resident_Crop.mp4', n_frames=3)
        out_dir = tmp_path / 'out'
        cpu = torch.device('cpu')
        run_dino_cls([video], out_dir, device=cpu)

        # Act
        again = run_dino_cls([video], out_dir, device=cpu)
        forced = run_dino_cls([video], out_dir, overwrite=True, device=cpu)

        # Assert
        assert again == []
        assert len(forced) == 1


class TestEmbeddingDirName:
    """Test the function embedding_dir_name."""

    def test_embedding_dir_name(self):
        assert embedding_dir_name('vits16') == 'dinov3_vits16'
