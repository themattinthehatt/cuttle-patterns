"""Tests for cuttle_patterns.cli.cmd_embed."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from cuttle_patterns.cli.cmd_embed import VGG_BATCH_SIZE, cmd_embed
from cuttle_patterns.embedders.base import Embedder


class _FakeBackbone:
    # matches _make_args' default --backbone/--resolution so tests asserting on
    # cmd_embed's default model-name (embedder.id = backbone.key + readout.name) see a
    # realistic backbone.key without loading a real DINOv3 model
    key = 'dinov3_vitb16_224'

    def preprocess(self, frames):
        return frames

    def forward(self, x):
        raise NotImplementedError

    def metadata(self) -> dict:
        return {}


class _FakeReadout:
    # matches _make_args' default --readout so the default model-name test (which uses
    # embedder.readout.name, not raw args.readout) stays consistent with 'cls'
    name = 'cls'
    fit_passes = 0

    def __call__(self, tokens):
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return 2

    def state_dict(self) -> dict:
        return {}

    def metadata(self) -> dict:
        return {}


class _FakeEmbedder(Embedder):
    """Deterministic stand-in for a real DINOv3 embedder, used to skip model loading."""

    def __init__(self):
        super().__init__(_FakeBackbone(), _FakeReadout())

    def embed(self, frames: np.ndarray) -> np.ndarray:
        return np.zeros((len(frames), 2), dtype=np.float32)


class _FakeGramReadout:
    """Stand-in for a stateful readout, to test cmd_embed's fitting step."""

    name = 'gram_k4_taper'
    fit_passes = 1

    def __call__(self, tokens):
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return 10

    def state_dict(self) -> dict:
        return {}

    def metadata(self) -> dict:
        return {'gram_k': 4}


class _FakeGramEmbedder(Embedder):
    """Deterministic stand-in for a gram embedder, used to skip model loading/fitting."""

    def __init__(self):
        super().__init__(_FakeBackbone(), _FakeGramReadout())

    def embed(self, frames: np.ndarray) -> np.ndarray:
        return np.zeros((len(frames), 10), dtype=np.float32)


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
        gram_k=64,
        gram_weights='taper',
        vgg_layer='relu3_1',
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

    def test_cmd_embed_fits_stateful_readout_before_embedding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=3)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.build_embedder',
            lambda *args, **kwargs: _FakeGramEmbedder(),
        )
        fit_calls = []
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.fit_readout',
            lambda embedder, fit_frame_paths, batch_size: fit_calls.append(fit_frame_paths),
        )
        args = _make_args(results_dir=results_dir, readout='gram')

        # Act
        cmd_embed(args)

        # Assert -- fitting ran once, before writing; default model name derives from
        # the embedder's actual readout.name (gram_k4_taper), not raw args.readout ('gram')
        assert len(fit_calls) == 1
        model_dir = results_dir / 'beast_models' / 'dinov3_vitb16_224_gram_k4_taper'
        assert (model_dir / 'config.yaml').is_file()

    def test_cmd_embed_passes_gram_kwargs_to_build_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=1)
        captured = {}

        def _fake_build_embedder(backbone_arch, resolution, readout_name, device, **kwargs):
            captured['readout_kwargs'] = kwargs.get('readout_kwargs')
            return _FakeGramEmbedder()

        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.build_embedder', _fake_build_embedder)
        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.fit_readout', lambda *a, **k: None)
        args = _make_args(
            results_dir=results_dir, readout='gram', gram_k=32, gram_weights='uniform',
        )

        # Act
        cmd_embed(args)

        # Assert
        assert captured['readout_kwargs'] == {'k': 32, 'weights': 'uniform'}

    def test_cmd_embed_passes_vgg_layer_and_default_resolution_to_build_embedder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=1)
        captured = {}

        def _fake_build_embedder(backbone_arch, resolution, readout_name, device, **kwargs):
            captured['resolution'] = resolution
            captured['vgg_layer'] = kwargs.get('vgg_layer')
            return _FakeEmbedder()

        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.build_embedder', _fake_build_embedder)
        args = _make_args(
            results_dir=results_dir, backbone='vgg19', resolution=None, vgg_layer='relu4_1',
        )

        # Act
        cmd_embed(args)

        # Assert -- --resolution wasn't passed, so it defaults to DEFAULT_VGG_RESOLUTION
        # (448) for backbone vgg19, not DEFAULT_RESOLUTION (224)
        assert captured == {'resolution': 448, 'vgg_layer': 'relu4_1'}

    def test_cmd_embed_forces_batch_size_for_vgg19(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=1)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.build_embedder',
            lambda *args, **kwargs: _FakeGramEmbedder(),
        )
        captured = {}

        def _fake_fit_readout(embedder, fit_frame_paths, batch_size):
            captured['fit_batch_size'] = batch_size

        def _fake_run_embed(embedder, frame_paths, batch_size):
            captured['run_batch_size'] = batch_size
            embeddings = embedder.embed(np.zeros((len(frame_paths), 4, 4, 3), dtype=np.uint8))
            meta = pd.DataFrame({
                'video_name': ['v'] * len(frame_paths),
                'day': [1] * len(frame_paths),
                'tank': [1] * len(frame_paths),
                'role': ['Resident'] * len(frame_paths),
                'frame_number': list(range(len(frame_paths))),
            })
            return embeddings, meta

        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.fit_readout', _fake_fit_readout)
        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.run_embed', _fake_run_embed)
        args = _make_args(
            results_dir=results_dir, backbone='vgg19', readout='gram', batch_size=128,
        )

        # Act
        cmd_embed(args)

        # Assert -- --batch-size 128 was passed but ignored for vgg19
        assert captured == {'fit_batch_size': VGG_BATCH_SIZE, 'run_batch_size': VGG_BATCH_SIZE}

    def test_cmd_embed_respects_batch_size_for_dinov3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=1)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.build_embedder',
            lambda *args, **kwargs: _FakeEmbedder(),
        )
        captured = {}

        def _fake_run_embed(embedder, frame_paths, batch_size):
            captured['run_batch_size'] = batch_size
            embeddings = embedder.embed(np.zeros((len(frame_paths), 4, 4, 3), dtype=np.uint8))
            meta = pd.DataFrame({
                'video_name': ['v'] * len(frame_paths),
                'day': [1] * len(frame_paths),
                'tank': [1] * len(frame_paths),
                'role': ['Resident'] * len(frame_paths),
                'frame_number': list(range(len(frame_paths))),
            })
            return embeddings, meta

        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.run_embed', _fake_run_embed)
        args = _make_args(results_dir=results_dir, batch_size=7)

        # Act
        cmd_embed(args)

        # Assert -- unlike vgg19, DINOv3 keeps the user's --batch-size unchanged
        assert captured == {'run_batch_size': 7}

    def test_cmd_embed_does_not_fit_stateless_readout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _write_frames(results_dir / 'beast_frames', n_frames=1)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_embed.build_embedder',
            lambda *args, **kwargs: _FakeEmbedder(),
        )

        def _fail_if_called(*args, **kwargs):
            raise AssertionError('fit_readout should not be called for a stateless readout')

        monkeypatch.setattr('cuttle_patterns.cli.cmd_embed.fit_readout', _fail_if_called)
        args = _make_args(results_dir=results_dir)

        # Act & Assert -- would raise if fit_readout were called
        cmd_embed(args)
