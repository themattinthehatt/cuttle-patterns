"""Tests for cuttle_patterns.cli.cmd_extract_eligibility."""

from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_extract_eligibility import cmd_extract_eligibility, register


def _args(parse_cli, tmp_path: Path, *extra: str):
    return parse_cli(register, [
        'extract-eligibility', '--data-dir', str(tmp_path / 'data'),
        '--results-dir', str(tmp_path / 'results'), '--pose-dir', str(tmp_path / 'pose'), *extra,
    ])


class TestCmdExtractEligibility:
    """Test the function cmd_extract_eligibility."""

    def test_missing_config_and_overrides_exits_with_error(
        self, parse_cli, no_config, capsys: pytest.CaptureFixture,
    ):
        # Arrange
        args = parse_cli(register, ['extract-eligibility', '--pose-dir', 'x'])

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_eligibility(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_no_videos_prints_message(self, parse_cli, tmp_path: Path, capsys):
        # Arrange
        (tmp_path / 'results' / 'rectangles').mkdir(parents=True)

        # Act
        cmd_extract_eligibility(_args(parse_cli, tmp_path))

        # Assert
        assert 'No videos found' in capsys.readouterr().out

    def test_delegates_to_run_eligibility_with_resolved_paths_and_options(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        rect_dir = tmp_path / 'results' / 'rectangles'
        rect_dir.mkdir(parents=True)
        (rect_dir / 'Day2_Tank1_Cuttle1_Resident_Crop.mp4').touch()
        calls = []
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_eligibility.run_eligibility',
            lambda *a, **k: calls.append((a, k)) or [Path('x')],
        )

        # Act
        cmd_extract_eligibility(_args(parse_cli, tmp_path, '--jump-max', '30', '--shard', '1',
                                      '--n-shards', '2'))

        # Assert
        (video_paths, pose_dir, data_dir, out_dir), kwargs = calls[0]
        assert [p.name for p in video_paths] == ['Day2_Tank1_Cuttle1_Resident_Crop.mp4']
        assert pose_dir == tmp_path / 'pose'
        assert out_dir == tmp_path / 'results' / 'extraction' / 'eligibility'
        assert kwargs['jump_max'] == 30.0
        assert (kwargs['shard'], kwargs['n_shards']) == (1, 2)
        assert 'next: cuttle extract-classify' in capsys.readouterr().out
