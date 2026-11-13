"""Tests for eval.build_manifest."""

from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.eval.build_manifest import build_eval_manifest


def _write_extract(results_dir: Path, rows: dict) -> None:
    path = results_dir / 'manifests'
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path / 'extract.parquet')


def _write_classifications(results_dir: Path, name: str, rows: dict) -> None:
    path = results_dir / 'classifications'
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path / f'{name}.parquet')


class TestBuildEvalManifest:
    """Test the function build_eval_manifest."""

    def test_joins_predicted_pattern(self, tmp_path):
        # Arrange
        _write_extract(tmp_path, {
            'session_id': ['Day1_Tank1'],
            'fish_id': ['Cuttle1_Resident'],
            'frame_idx': [7],
            'image_path': ['/x/img7.png'],
        })
        _write_classifications(tmp_path, 'clf', {
            'video_name': ['Day1_Tank1_Cuttle1_Resident_Crop'],
            'frame_number': [7],
            'predicted_pattern': ['Leopard'],
            'confidence': [0.9],
        })

        # Act
        manifest = build_eval_manifest(tmp_path, 'clf')

        # Assert
        assert manifest.loc[0, 'predicted_pattern'] == 'Leopard'
        assert manifest.loc[0, 'video_name'] == 'Day1_Tank1_Cuttle1_Resident_Crop'
        assert manifest.loc[0, 'frame_number'] == 7

    def test_raises_on_missing_extract_manifest(self, tmp_path):
        # Act / Assert
        with pytest.raises(FileNotFoundError):
            build_eval_manifest(tmp_path, 'clf')

    def test_raises_on_missing_classifications_file(self, tmp_path):
        # Arrange
        _write_extract(tmp_path, {
            'session_id': ['Day1_Tank1'],
            'fish_id': ['Cuttle1_Resident'],
            'frame_idx': [7],
            'image_path': ['/x/img7.png'],
        })

        # Act / Assert
        with pytest.raises(FileNotFoundError):
            build_eval_manifest(tmp_path, 'clf')

    def test_raises_on_unmatched_frame(self, tmp_path):
        # Arrange
        _write_extract(tmp_path, {
            'session_id': ['Day1_Tank1'],
            'fish_id': ['Cuttle1_Resident'],
            'frame_idx': [7],
            'image_path': ['/x/img7.png'],
        })
        _write_classifications(tmp_path, 'clf', {
            'video_name': ['Day1_Tank1_Cuttle1_Resident_Crop'],
            'frame_number': [999],
            'predicted_pattern': ['Leopard'],
            'confidence': [0.9],
        })

        # Act / Assert
        with pytest.raises(ValueError, match='no matching classifier prediction'):
            build_eval_manifest(tmp_path, 'clf')

    def test_joins_per_video_directory_layout(self, tmp_path):
        # Arrange
        _write_extract(tmp_path, {
            'session_id': ['Day1_Tank1', 'Day1_Tank1'],
            'fish_id': ['Cuttle1_Resident', 'Cuttle1_Resident'],
            'frame_idx': [7, 9],
            'image_path': ['/x/img7.png', '/x/img9.png'],
        })
        clf_dir = tmp_path / 'classifications' / 'clf'
        clf_dir.mkdir(parents=True)
        pd.DataFrame({
            'frame_idx': [7, 8, 9],
            'predicted_pattern': ['Leopard', 'White', 'Pale_aggression'],
            'confidence': [0.9, 0.8, 0.7],
        }).to_parquet(clf_dir / 'Day1_Tank1_Cuttle1_Resident_Crop.parquet')

        # Act
        manifest = build_eval_manifest(tmp_path, 'clf')

        # Assert
        assert list(manifest['predicted_pattern']) == ['Leopard', 'Pale_aggression']
        assert list(manifest['frame_number']) == [7, 9]

    def test_raises_on_missing_video_in_directory_layout(self, tmp_path):
        # Arrange
        _write_extract(tmp_path, {
            'session_id': ['Day1_Tank1'],
            'fish_id': ['Cuttle1_Resident'],
            'frame_idx': [7],
            'image_path': ['/x/img7.png'],
        })
        (tmp_path / 'classifications' / 'clf').mkdir(parents=True)

        # Act / Assert
        with pytest.raises(FileNotFoundError):
            build_eval_manifest(tmp_path, 'clf')
