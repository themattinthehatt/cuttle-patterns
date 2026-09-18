"""Tests for eval.report."""

import json

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.eval.load_embeddings import EmbedderSpec
from cuttle_patterns.eval.report import (
    SCOREBOARD_COLUMNS,
    run_report,
    score_embedder,
    write_report,
)

VIDEO_A = 'Day1_Tank1_Cuttle1_Resident_Crop'
VIDEO_B = 'Day1_Tank1_Cuttle2_Intruder_Crop'


def _make_manifest(n_per_video: int = 16) -> pd.DataFrame:
    video_names = [VIDEO_A] * n_per_video + [VIDEO_B] * n_per_video
    frame_numbers = list(range(n_per_video)) * 2
    half = n_per_video // 2
    predicted_pattern = (['Leopard'] * half + ['White'] * half) * 2
    return pd.DataFrame({
        'video_name': video_names,
        'frame_number': frame_numbers,
        'predicted_pattern': predicted_pattern,
    })


def _make_video_frames(manifest: pd.DataFrame, dim: int = 8) -> dict[str, dict[int, np.ndarray]]:
    rng = np.random.default_rng(0)
    return {
        video_name: {
            int(frame_number): rng.normal(size=dim)
            for frame_number in manifest.loc[manifest['video_name'] == video_name, 'frame_number']
        }
        for video_name in manifest['video_name'].unique()
    }


class TestScoreEmbedder:
    """Test the function score_embedder."""

    def test_returns_all_scoreboard_keys(self, tmp_path, make_latents_dir, write_model_config):
        # Arrange
        model_dir = tmp_path / 'beast_models' / 'my-model'
        write_model_config(model_dir, model_class='resnet')
        manifest = _make_manifest()
        make_latents_dir(model_dir, _make_video_frames(manifest))
        spec = EmbedderSpec(id='my-model', model_dir=model_dir)

        # Act
        result = score_embedder(spec, manifest, n_clusters=2, seeds=(0, 1), knn_k=3, cv_splits=2)

        # Assert
        assert set(result) == set(SCOREBOARD_COLUMNS)
        assert result['embedder'] == 'my-model'


class TestRunReport:
    """Test the function run_report."""

    def test_produces_one_row_per_spec(self, tmp_path, make_latents_dir, write_model_config):
        # Arrange
        model_dir = tmp_path / 'beast_models' / 'my-model'
        write_model_config(model_dir, model_class='resnet')
        manifest = _make_manifest()
        make_latents_dir(model_dir, _make_video_frames(manifest))
        specs = [EmbedderSpec(id='my-model', model_dir=model_dir)]

        # Act
        scoreboard = run_report(specs, manifest, n_clusters=2, seeds=(0,), knn_k=3, cv_splits=2)

        # Assert
        assert len(scoreboard) == 1
        assert list(scoreboard.columns) == SCOREBOARD_COLUMNS


def _make_scoreboard_row(embedder: str, value: float = 0.5) -> pd.DataFrame:
    row = pd.DataFrame([dict.fromkeys(SCOREBOARD_COLUMNS, value)])
    row['embedder'] = embedder
    return row


class TestWriteReport:
    """Test the function write_report."""

    def test_writes_markdown_and_json(self, tmp_path):
        # Arrange
        scoreboard = _make_scoreboard_row('my-model')
        out_dir = tmp_path / 'eval_results' / 'run1'

        # Act
        write_report(scoreboard, out_dir)

        # Assert
        assert (out_dir / 'scoreboard.md').is_file()
        metrics = json.loads((out_dir / 'metrics.json').read_text())
        assert metrics[0]['embedder'] == 'my-model'

    def test_adds_new_embedder_without_dropping_existing(self, tmp_path):
        # Arrange
        out_dir = tmp_path / 'eval_results'
        write_report(_make_scoreboard_row('model-a', value=0.1), out_dir)

        # Act
        write_report(_make_scoreboard_row('model-b', value=0.2), out_dir)

        # Assert
        metrics = json.loads((out_dir / 'metrics.json').read_text())
        assert {row['embedder'] for row in metrics} == {'model-a', 'model-b'}

    def test_rerunning_same_embedder_replaces_its_row(self, tmp_path):
        # Arrange
        out_dir = tmp_path / 'eval_results'
        write_report(_make_scoreboard_row('model-a', value=0.1), out_dir)

        # Act
        write_report(_make_scoreboard_row('model-a', value=0.9), out_dir)

        # Assert
        metrics = json.loads((out_dir / 'metrics.json').read_text())
        assert len(metrics) == 1
        assert metrics[0]['embedder'] == 'model-a'
        assert metrics[0]['ami_cluster_class_mean'] == pytest.approx(0.9)
