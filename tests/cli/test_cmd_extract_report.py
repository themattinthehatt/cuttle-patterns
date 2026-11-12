"""Tests for cuttle_patterns.cli.cmd_extract_report."""

import json
from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_extract_report import cmd_extract_report, register


def _args(parse_cli, tmp_path: Path, *extra: str):
    return parse_cli(register, [
        'extract-report', '--results-dir', str(tmp_path / 'results'), '--data-dir', 'unused',
        '--name', 'sel', *extra,
    ])


class TestCmdExtractReport:
    """Test the function cmd_extract_report."""

    def test_missing_config_and_overrides_exits_with_error(self, parse_cli, no_config):
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_report(parse_cli(register, ['extract-report', '--name', 'n']))
        assert exc_info.value.code == 1

    def test_missing_selection_exits_with_error(self, parse_cli, tmp_path: Path, capsys):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_report(_args(parse_cli, tmp_path))
        assert exc_info.value.code == 1
        assert 'extract-kmeans-pooled' in capsys.readouterr().out

    def test_uses_classifier_and_embedding_names_recorded_by_stage1(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        sel_dir = tmp_path / 'results' / 'extraction' / 'selections' / 'sel'
        sel_dir.mkdir(parents=True)
        (sel_dir / 'summary.json').write_text(json.dumps({
            'stage1': {'classifier_name': 'my_clf', 'embedding_name': 'my_emb'},
        }))
        calls = []

        def fake(*args, **kwargs):
            calls.append((args, kwargs))
            return {'n_videos': 0, 'per_video': {}, 'classes': {}}

        monkeypatch.setattr('cuttle_patterns.cli.cmd_extract_report.run_report', fake)
        monkeypatch.setattr('cuttle_patterns.cli.cmd_extract_report.format_report',
                            lambda report: 'REPORT TEXT')

        # Act
        cmd_extract_report(_args(parse_cli, tmp_path, '--no-sheets', '--sample', '10'))

        # Assert
        (selection_dir, elig_dir, clf_dir, emb_dir, video_dir), kwargs = calls[0]
        assert clf_dir == tmp_path / 'results' / 'classifications' / 'my_clf'
        assert emb_dir == tmp_path / 'results' / 'embeddings' / 'my_emb'
        assert video_dir == tmp_path / 'results' / 'rectangles'
        assert kwargs['sheets'] is False
        assert kwargs['sample'] == 10
        assert 'REPORT TEXT' in capsys.readouterr().out
