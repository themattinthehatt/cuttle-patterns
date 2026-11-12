"""Tests for cuttle_patterns.cli.cmd_extract_kmeans_pooled."""

from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.cli.cmd_extract_kmeans_pooled import cmd_extract_kmeans_pooled, register


def _args(parse_cli, tmp_path: Path, *extra: str):
    return parse_cli(register, [
        'extract-kmeans-pooled', '--results-dir', str(tmp_path / 'results'),
        '--data-dir', 'unused', '--name', 'sel', *extra,
    ])


class TestCmdExtractKmeansPooled:
    """Test the function cmd_extract_kmeans_pooled."""

    def test_missing_config_and_overrides_exits_with_error(self, parse_cli, no_config):
        with pytest.raises(SystemExit) as exc_info:
            args = parse_cli(register, ['extract-kmeans-pooled', '--name', 'n'])
            cmd_extract_kmeans_pooled(args)
        assert exc_info.value.code == 1

    def test_missing_stage1_output_exits_with_error(self, parse_cli, tmp_path: Path, capsys):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_kmeans_pooled(_args(parse_cli, tmp_path))
        assert exc_info.value.code == 1
        assert 'extract-kmeans-video' in capsys.readouterr().out

    def test_delegates_with_selection_paths_and_prints_next_steps(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        calls = []

        def fake(kmeans_dir, out_dir, **kwargs):
            calls.append((kmeans_dir, out_dir, kwargs))
            return pd.DataFrame({'video': ['v1', 'v2'], 'frame_idx': [1, 2]})

        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_kmeans_pooled.run_kmeans_pooled', fake,
        )

        # Act
        cmd_extract_kmeans_pooled(_args(parse_cli, tmp_path, '--picks-per-group', '20'))

        # Assert
        results = tmp_path / 'results' / 'extraction'
        assert calls[0][0] == results / 'kmeans_video' / 'sel'
        assert calls[0][1] == results / 'selections' / 'sel'
        assert calls[0][2]['picks_per_group'] == 20
        out = capsys.readouterr().out
        assert 'selected 2 frames from 2 videos' in out
        assert 'extract-report' in out
