"""Tests for cuttle_patterns.cli.cmd_extract_export."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from cuttle_patterns.cli.cmd_extract_export import cmd_extract_export, register

STEM = 'Day2_Tank1_Cuttle1_Resident_Crop'


def _args(parse_cli, tmp_path: Path, *extra: str):
    return parse_cli(register, [
        'extract-export', '--results-dir', str(tmp_path / 'results'), '--data-dir', 'unused',
        '--name', 'sel', *extra,
    ])


def _write_selection(tmp_path: Path, frames: list[int]) -> None:
    sel_dir = tmp_path / 'results' / 'extraction' / 'selections' / 'sel'
    sel_dir.mkdir(parents=True)
    pd.DataFrame({'video': STEM, 'frame_idx': frames}).to_parquet(sel_dir / 'manifest.parquet')


class TestCmdExtractExport:
    """Test the function cmd_extract_export."""

    def test_missing_config_and_overrides_exits_with_error(self, parse_cli, no_config):
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_export(parse_cli(register, ['extract-export', '--name', 'n']))
        assert exc_info.value.code == 1

    def test_missing_selection_exits_with_error(self, parse_cli, tmp_path: Path, capsys):
        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_export(_args(parse_cli, tmp_path))
        assert exc_info.value.code == 1
        assert 'extract-kmeans-pooled' in capsys.readouterr().out

    def test_exports_frames_and_writes_extract_manifest(
        self, parse_cli, tmp_path: Path, make_video: Callable, capsys,
    ):
        # Arrange
        _write_selection(tmp_path, [1, 3])
        rect_dir = tmp_path / 'results' / 'rectangles'
        rect_dir.mkdir()
        make_video(rect_dir / f'{STEM}.mp4', n_frames=5)

        # Act
        cmd_extract_export(_args(parse_cli, tmp_path))

        # Assert
        frames_dir = tmp_path / 'results' / 'beast_frames' / STEM
        assert sorted(p.name for p in frames_dir.glob('*.png')) == [
            'img00000001.png', 'img00000003.png',
        ]
        manifest = pd.read_parquet(tmp_path / 'results' / 'manifests' / 'extract.parquet')
        assert list(manifest.columns) == ['session_id', 'fish_id', 'frame_idx', 'image_path']
        assert 'next: cuttle train' in capsys.readouterr().out

    def test_output_dir_override(self, parse_cli, tmp_path: Path, make_video: Callable):
        # Arrange
        _write_selection(tmp_path, [2])
        rect_dir = tmp_path / 'results' / 'rectangles'
        rect_dir.mkdir()
        make_video(rect_dir / f'{STEM}.mp4', n_frames=4)

        # Act
        cmd_extract_export(_args(parse_cli, tmp_path, '--output-dir', str(tmp_path / 'frames')))

        # Assert
        assert (tmp_path / 'frames' / STEM / 'img00000002.png').exists()
