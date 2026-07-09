"""Tests for cuttle_patterns.cli.cmd_ingest."""

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_ingest import cmd_ingest


def _raise_file_not_found() -> None:
    raise FileNotFoundError('no config file found at ~/.cuttle-patterns/config.yaml')


class TestCmdIngest:
    """Test the function cmd_ingest."""

    def test_cmd_ingest_missing_config_and_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_ingest.load_config', _raise_file_not_found,
        )
        args = argparse.Namespace(data_dir=None, results_dir=None)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_ingest(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_cmd_ingest_with_overrides(
        self,
        tmp_path: Path,
        make_video: Callable,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        make_video(data_dir / 'session-01_cuttle-01.mp4', n_frames=5)
        args = argparse.Namespace(data_dir=data_dir, results_dir=results_dir)

        # Act
        cmd_ingest(args)

        # Assert
        manifest_path = results_dir / 'manifests' / 'ingest.parquet'
        assert manifest_path.exists()
        assert 'Ingested 1 videos across 1 sessions.' in capsys.readouterr().out
