"""Tests for cuttle_patterns.cli.cmd_embed."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cuttle_patterns.cli.cmd_embed import cmd_embed
from cuttle_patterns.embedders.base import Embedder


class _FakeBackbone:
    key = 'fake_backbone'

    def preprocess(self, frames):
        return frames

    def forward(self, x):
        raise NotImplementedError

    def metadata(self) -> dict:
        return {}


class _FakeReadout:
    name = 'fake_readout'

    def __call__(self, tokens):
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return 2

    def metadata(self) -> dict:
        return {}


class _FakeEmbedder(Embedder):
    """Deterministic stand-in for a real DINOv3 embedder, used to skip model loading."""

    def __init__(self):
        super().__init__(_FakeBackbone(), _FakeReadout())

    def embed(self, frames: np.ndarray) -> np.ndarray:
        return np.zeros((len(frames), 2), dtype=np.float32)


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        backbone='vitb16',
        resolution=224,
        readout='cls',
        model_name=None,
        input_dir=None,
        predictions_name=None,
        batch_size=2,
        device='cpu',
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_frames(input_dir: Path, n_frames: int = 3) -> None:
    video_dir = input_dir / 'Day1_Tank2_Cuttle1_Resident_Crop'
    video_dir.mkdir(parents=True)
    for i in range(n_frames):
        frame = np.full((4, 4, 3), i, dtype=np.uint8)
        cv2.imwrite(str(video_dir / f'img{i:08d}.png'), frame)


class TestCmdEmbed:
    """Test the function cmd_embed."""

    def test_cmd_embed_missing_input_dir(
        self, tmp_path: Path, capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_embed(args)
        assert exc_info.value.code == 1
        assert 'does not exist' in capsys.readouterr().out

    def test_cmd_embed_writes_output_with_default_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=3)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.build_embedder',
            lambda *args, **kwargs: _FakeEmbedder(),
        )
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_embed(args)

        # Assert
        model_dir = results_dir / 'beast_models' / 'dinov3_vitb16_224_cls'
        with (model_dir / 'config.yaml').open() as f:
            config = yaml.safe_load(f)
        assert config['model']['model_class'] == 'embedder'

        latents_dir = model_dir / 'image_predictions' / 'beast_frames' / 'latents'
        embeddings = np.load(latents_dir / 'embeddings.npy')
        assert embeddings.shape == (3, 2)

    def test_cmd_embed_respects_custom_model_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=1)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.build_embedder',
            lambda *args, **kwargs: _FakeEmbedder(),
        )
        args = _make_args(results_dir=results_dir, model_name='my-custom-embedder')

        # Act
        cmd_embed(args)

        # Assert
        model_dir = results_dir / 'beast_models' / 'my-custom-embedder'
        assert (model_dir / 'config.yaml').is_file()
