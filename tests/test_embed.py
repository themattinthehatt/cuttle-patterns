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
    fit_readout,
    run_embed,
    sample_fit_frame_paths,
    write_embedder_output,
)
from cuttle_patterns.embedders.base import Embedder
from cuttle_patterns.embedders.readouts import READOUTS_BY_NAME


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
    fit_passes = 0

    def __call__(self, tokens):
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return 1

    def state_dict(self) -> dict:
        return {}

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

    def test_build_embedder_forwards_readout_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange
        captured = {}

        class _FakeReadoutClass:
            def __init__(self, dim, **kwargs):
                captured['dim'] = dim
                captured['kwargs'] = kwargs

        class _FakeBackboneClass:
            embed_dim = 8

            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr('cuttle_patterns.embed.DINOv3Backbone', _FakeBackboneClass)
        monkeypatch.setitem(READOUTS_BY_NAME, 'fake_for_test', _FakeReadoutClass)

        # Act
        build_embedder(
            'vitb16', 224, 'fake_for_test', device=torch.device('cpu'),
            readout_kwargs={'k': 4, 'weights': 'uniform'},
        )

        # Assert
        assert captured == {'dim': 8, 'kwargs': {'k': 4, 'weights': 'uniform'}}


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


class TestSampleFitFramePaths:
    """Test the function sample_fit_frame_paths."""

    def test_sample_fit_frame_paths_caps_evenly_per_video(self):
        # Arrange
        frame_paths = (
            [Path(f'/x/video_a/img{i:08d}.png') for i in range(10)]
            + [Path(f'/x/video_b/img{i:08d}.png') for i in range(10)]
        )

        # Act
        result = sample_fit_frame_paths(frame_paths, fit_set_size=6, seed=0)

        # Assert -- 2 videos -> cap of ceil(6 / 2) = 3 per video
        by_video: dict[str, int] = {}
        for p in result:
            by_video[p.parent.name] = by_video.get(p.parent.name, 0) + 1
        assert by_video == {'video_a': 3, 'video_b': 3}

    def test_sample_fit_frame_paths_takes_all_when_fewer_than_cap(self):
        # Arrange
        frame_paths = [Path(f'/x/video_a/img{i:08d}.png') for i in range(2)]

        # Act
        result = sample_fit_frame_paths(frame_paths, fit_set_size=100, seed=0)

        # Assert
        assert sorted(result) == sorted(frame_paths)

    def test_sample_fit_frame_paths_deterministic_for_fixed_seed(self):
        # Arrange
        frame_paths = [Path(f'/x/video_a/img{i:08d}.png') for i in range(20)]

        # Act
        result_a = sample_fit_frame_paths(frame_paths, fit_set_size=5, seed=7)
        result_b = sample_fit_frame_paths(frame_paths, fit_set_size=5, seed=7)

        # Assert
        assert result_a == result_b


class TestFitReadout:
    """Test the function fit_readout."""

    def test_fit_readout_calls_partial_fit_and_finalize_pass_per_pass(self, tmp_path: Path):
        # Arrange
        input_dir = tmp_path / 'beast_frames'
        _write_frame(input_dir / 'video_a' / 'img00000001.png', 1)
        _write_frame(input_dir / 'video_a' / 'img00000002.png', 2)
        frame_paths = find_frame_paths(input_dir)
        calls = []

        class _FitTrackingReadout:
            name = 'fit_tracking'
            fit_passes = 2

            def partial_fit(self, tokens, pass_idx):
                calls.append(('partial_fit', pass_idx, len(tokens)))

            def finalize_pass(self, pass_idx):
                calls.append(('finalize_pass', pass_idx))

        class _FitTrackingBackbone:
            def preprocess(self, frames):
                return frames

            def forward(self, x):
                return x

        embedder = Embedder(_FitTrackingBackbone(), _FitTrackingReadout())

        # Act
        fit_readout(embedder, frame_paths, batch_size=1)

        # Assert
        assert calls == [
            ('partial_fit', 0, 1), ('partial_fit', 0, 1),
            ('finalize_pass', 0),
            ('partial_fit', 1, 1), ('partial_fit', 1, 1),
            ('finalize_pass', 1),
        ]

    def test_fit_readout_noop_for_stateless_readout(self, tmp_path: Path):
        # Arrange
        input_dir = tmp_path / 'beast_frames'
        _write_frame(input_dir / 'video_a' / 'img00000001.png', 1)
        frame_paths = find_frame_paths(input_dir)
        embedder = _FakeEmbedder()

        # Act & Assert -- fit_passes == 0, so partial_fit/finalize_pass are never
        # called; _FakeReadout.__call__ would raise if the embedder tried to embed
        fit_readout(embedder, frame_paths, batch_size=1)


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
        assert not (model_dir / 'readout_state.pt').exists()

    def test_write_embedder_output_persists_nonempty_readout_state(self, tmp_path: Path):
        # Arrange
        embeddings = np.array([[1.0]], dtype=np.float32)
        meta = pd.DataFrame({
            'video_name': ['Day1_Tank2_Cuttle1_Resident_Crop'], 'day': [1], 'tank': [2],
            'role': ['Resident'], 'frame_number': [1],
        })

        class _StatefulFakeReadout(_FakeReadout):
            def state_dict(self) -> dict:
                return {'P': torch.zeros(2, 2)}

        embedder = Embedder(_FakeBackbone(), _StatefulFakeReadout())
        model_dir = tmp_path / 'beast_models' / 'fake-model'

        # Act
        write_embedder_output(embeddings, meta, embedder, model_dir, 'beast_frames')

        # Assert
        state = torch.load(model_dir / 'readout_state.pt', weights_only=False)
        torch.testing.assert_close(state['P'], torch.zeros(2, 2))
