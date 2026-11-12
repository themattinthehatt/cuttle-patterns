"""Tests for cuttle_patterns.cli.cmd_extract_dino_cls."""

from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_extract_dino_cls import cmd_extract_dino_cls, register


def _args(parse_cli, tmp_path: Path, *extra: str):
    return parse_cli(register, [
        'extract-dino-cls', '--results-dir', str(tmp_path / 'results'), *extra,
        '--data-dir', str(tmp_path / 'data'),
    ])


class TestCmdExtractDinoCls:
    """Test the function cmd_extract_dino_cls."""

    def test_missing_config_and_overrides_exits_with_error(self, parse_cli, no_config):
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_dino_cls(parse_cli(register, ['extract-dino-cls']))
        assert exc_info.value.code == 1

    def test_no_videos_prints_message(self, parse_cli, tmp_path: Path, capsys):
        # Arrange
        (tmp_path / 'results' / 'rectangles').mkdir(parents=True)

        # Act
        cmd_extract_dino_cls(_args(parse_cli, tmp_path))

        # Assert
        assert 'No videos found' in capsys.readouterr().out

    def test_output_dir_is_named_after_the_architecture(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        rect_dir = tmp_path / 'results' / 'rectangles'
        rect_dir.mkdir(parents=True)
        (rect_dir / 'Day2_Tank1_Cuttle1_Resident_Crop.mp4').touch()
        calls = []
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_dino_cls.run_dino_cls',
            lambda *a, **k: calls.append((a, k)) or [],
        )

        # Act
        cmd_extract_dino_cls(_args(parse_cli, tmp_path, '--arch', 'vitb16'))

        # Assert
        (video_paths, out_dir), kwargs = calls[0]
        assert out_dir == tmp_path / 'results' / 'embeddings' / 'dinov3_vitb16'
        assert kwargs['arch'] == 'vitb16'
        assert 'next: cuttle extract-kmeans-video' in capsys.readouterr().out
