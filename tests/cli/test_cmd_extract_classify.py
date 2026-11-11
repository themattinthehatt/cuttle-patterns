"""Tests for cuttle_patterns.cli.cmd_extract_classify."""

from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_extract_classify import cmd_extract_classify, register


def _args(parse_cli, tmp_path: Path, *extra: str):
    return parse_cli(register, [
        'extract-classify', '--data-dir', str(tmp_path / 'data'),
        '--results-dir', str(tmp_path / 'results'),
        '--model-dir', str(tmp_path / 'models' / '2026-11-07_model'), *extra,
    ])


class TestCmdExtractClassify:
    """Test the function cmd_extract_classify."""

    def test_missing_config_and_overrides_exits_with_error(
        self, parse_cli, no_config, capsys: pytest.CaptureFixture,
    ):
        # Arrange
        args = parse_cli(register, ['extract-classify', '--model-dir', 'm'])

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_classify(args)
        assert exc_info.value.code == 1

    def test_no_videos_prints_message(self, parse_cli, tmp_path: Path, capsys):
        # Arrange
        (tmp_path / 'data').mkdir()

        # Act
        cmd_extract_classify(_args(parse_cli, tmp_path))

        # Assert
        assert 'No videos found' in capsys.readouterr().out

    def test_output_dir_defaults_to_classifications_named_after_model_dir(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        (tmp_path / 'data').mkdir()
        (tmp_path / 'data' / 'Day2_Tank1_Cuttle1_Resident_Crop.mp4').touch()
        calls = []
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_classify.run_classify',
            lambda *a, **k: calls.append((a, k)) or [],
        )

        # Act
        cmd_extract_classify(_args(parse_cli, tmp_path, '--save-features'))

        # Assert
        (video_paths, model_dir, out_dir), kwargs = calls[0]
        assert out_dir == tmp_path / 'results' / 'classifications' / '2026-11-07_model'
        assert kwargs['save_features'] is True
        assert 'next: cuttle extract-dino-cls' in capsys.readouterr().out

    def test_model_name_overrides_output_subdirectory(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        (tmp_path / 'data').mkdir()
        (tmp_path / 'data' / 'Day2_Tank1_Cuttle1_Resident_Crop.mp4').touch()
        calls = []
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_classify.run_classify',
            lambda *a, **k: calls.append(a) or [],
        )

        # Act
        cmd_extract_classify(_args(parse_cli, tmp_path, '--model-name', 'mine'))

        # Assert
        assert calls[0][2].name == 'mine'
