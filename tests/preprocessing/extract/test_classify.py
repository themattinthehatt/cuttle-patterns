"""Tests for cuttle_patterns.preprocessing.extract.classify (run on CPU with random weights)."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from torchvision.models import resnet18

from cuttle_patterns.preprocessing.extract.classify import (
    BEHAVIORS_FILENAME,
    WEIGHTS_FILENAME,
    TwoHead,
    build_predictions,
    classify_video,
    load_classifier,
    preprocess,
    run_classify,
)

CLASSES = ['Dark aggression', 'Leopard', 'Pale aggression']
CPU = torch.device('cpu')


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A directory with random two-head resnet18 weights and a behaviors.json."""
    directory = tmp_path / 'model'
    directory.mkdir()
    torch.manual_seed(0)
    model = resnet18(weights=None)
    model.fc = TwoHead(model.fc.in_features, len(CLASSES))
    torch.save(model.state_dict(), directory / WEIGHTS_FILENAME)
    (directory / BEHAVIORS_FILENAME).write_text(json.dumps(CLASSES))
    return directory


class TestTwoHead:
    """Test the class TwoHead."""

    def test_forward_returns_pattern_and_border_shapes(self):
        # Arrange
        head = TwoHead(512, 5)

        # Act
        pattern, border = head(torch.zeros(4, 512))

        # Assert
        assert pattern.shape == (4, 5)
        assert border.shape == (4, 1)


class TestLoadClassifier:
    """Test the function load_classifier."""

    def test_load_classifier_returns_eval_model_classes_and_feature_hook(self, model_dir: Path):
        # Act
        model, classes, captured = load_classifier(model_dir, CPU)
        with torch.no_grad():
            pattern, _ = model(torch.zeros(2, 3, 224, 224))

        # Assert
        assert classes == CLASSES
        assert not model.training
        assert pattern.shape == (2, 3)
        assert captured['value'].shape == (2, 512)


class TestPreprocess:
    """Test the function preprocess."""

    def test_preprocess_resizes_flips_channels_and_normalizes(self):
        # Arrange: a pure-blue bgr frame (b=255, g=0, r=0)
        frames = np.zeros((2, 64, 48, 3), dtype=np.uint8)
        frames[..., 0] = 255

        # Act
        x = preprocess(frames, CPU)

        # Assert: after the bgr -> rgb flip the blue value is in channel 2
        assert x.shape == (2, 3, 224, 224)
        assert x[0, 2].mean() == pytest.approx((1.0 - 0.406) / 0.225, abs=1e-3)
        assert x[0, 0].mean() == pytest.approx((0.0 - 0.485) / 0.229, abs=1e-3)


class TestBuildPredictions:
    """Test the function build_predictions."""

    def test_build_predictions_columns_labels_and_confidence(self):
        # Arrange
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=np.float32)

        # Act
        df = build_predictions(probs, CLASSES)

        # Assert
        assert list(df['frame_idx']) == [0, 1]
        assert list(df['predicted_pattern']) == ['Dark aggression', 'Pale aggression']
        np.testing.assert_allclose(df['confidence'], [0.7, 0.8])
        assert 'prob_Dark_aggression' in df.columns
        np.testing.assert_allclose(df['prob_Leopard'], [0.2, 0.1])


class TestClassifyVideo:
    """Test the function classify_video."""

    def test_classify_video_gives_one_probability_row_per_frame(
        self, tmp_path: Path, make_video: Callable, model_dir: Path,
    ):
        # Arrange
        model, _, captured = load_classifier(model_dir, CPU)
        video = make_video(tmp_path / 'v.mp4', n_frames=5)

        # Act
        probs, feats = classify_video(video, model, captured, CPU, batch_size=2,
                                      save_features=True)

        # Assert
        assert probs.shape == (5, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert feats.shape == (5, 512)
        assert feats.dtype == np.float16

    def test_classify_video_without_features_returns_none(
        self, tmp_path: Path, make_video: Callable, model_dir: Path,
    ):
        # Arrange
        model, _, captured = load_classifier(model_dir, CPU)

        # Act
        _, feats = classify_video(make_video(tmp_path / 'v.mp4', n_frames=2), model, captured, CPU)

        # Assert
        assert feats is None


class TestRunClassify:
    """Test the function run_classify."""

    def test_run_classify_writes_table_provenance_and_progress(
        self, tmp_path: Path, make_video: Callable, model_dir: Path,
    ):
        # Arrange
        video = make_video(tmp_path / 'Day1_Tank1_Cuttle1_Resident_Crop.mp4', n_frames=4)
        out_dir = tmp_path / 'out'

        # Act
        written = run_classify([video], model_dir, out_dir, batch_size=2, device=CPU)

        # Assert
        table = pd.read_parquet(written[0])
        assert len(table) == 4
        assert yaml.safe_load((out_dir / 'classifier.yaml').read_text())['classes'] == CLASSES
        assert json.loads((out_dir / 'progress_shard0of1.json').read_text())['done'] == 1

    def test_run_classify_skips_finished_videos_unless_overwrite(
        self, tmp_path: Path, make_video: Callable, model_dir: Path,
    ):
        # Arrange
        video = make_video(tmp_path / 'Day1_Tank1_Cuttle1_Resident_Crop.mp4', n_frames=3)
        out_dir = tmp_path / 'out'
        run_classify([video], model_dir, out_dir, device=CPU)

        # Act
        again = run_classify([video], model_dir, out_dir, device=CPU)
        forced = run_classify([video], model_dir, out_dir, overwrite=True, device=CPU)

        # Assert
        assert again == []
        assert len(forced) == 1

    def test_run_classify_saves_features_when_requested(
        self, tmp_path: Path, make_video: Callable, model_dir: Path,
    ):
        # Arrange
        video = make_video(tmp_path / 'Day1_Tank1_Cuttle1_Resident_Crop.mp4', n_frames=3)
        out_dir = tmp_path / 'out'

        # Act
        run_classify([video], model_dir, out_dir, save_features=True, device=CPU)

        # Assert
        feats = np.load(out_dir / 'Day1_Tank1_Cuttle1_Resident_Crop_features.npy')
        assert feats.shape == (3, 512)
