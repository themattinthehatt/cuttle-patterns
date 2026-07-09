"""Tests for cuttle_patterns.cli.main."""

import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.cli.main import main


class TestMain:
    """Test the function main."""

    def test_main_help_lists_ingest_subcommand(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(sys, 'argv', ['cuttle', '--help'])

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        assert 'ingest' in capsys.readouterr().out

    def test_main_dispatches_to_ingest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        make_video: Callable,
    ):
        # Arrange
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        data_dir.mkdir()
        make_video(data_dir / 'session-01_cuttle-01.mp4', n_frames=10)
        (data_dir / 'session-01_cuttle-01.txt').write_text('1\n2\n')
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'cuttle', 'ingest',
                '--data-dir', str(data_dir),
                '--results-dir', str(results_dir),
            ],
        )

        # Act
        main()

        # Assert
        manifest_path = results_dir / 'manifests' / 'ingest.parquet'
        assert manifest_path.exists()
        manifest = pd.read_parquet(manifest_path)
        assert len(manifest) == 1
