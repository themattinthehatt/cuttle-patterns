"""Tests for cuttle_patterns.visualization.video_utils."""

import subprocess
from pathlib import Path

import pytest

from cuttle_patterns.visualization.video_utils import build_gif_command, open_ffmpeg_raw_writer


class TestBuildGifCommand:
    """Test the function build_gif_command."""

    def test_build_gif_command_uses_paths_and_defaults(self):
        # Arrange
        mp4_path = Path('/tmp/clip.mp4')
        gif_path = Path('/tmp/clip.gif')

        # Act
        command = build_gif_command(mp4_path, gif_path)

        # Assert
        assert command[0] == 'ffmpeg'
        assert str(mp4_path) in command
        assert str(gif_path) in command
        assert 'fps=15' in command[command.index('-vf') + 1]
        assert 'scale=320:-1' in command[command.index('-vf') + 1]

    def test_build_gif_command_honors_custom_fps_and_scale(self):
        # Arrange / Act
        command = build_gif_command(Path('a.mp4'), Path('a.gif'), fps=5, scale_width=100)

        # Assert
        filter_arg = command[command.index('-vf') + 1]
        assert 'fps=5' in filter_arg
        assert 'scale=100:-1' in filter_arg


class TestOpenFfmpegRawWriter:
    """Test the function open_ffmpeg_raw_writer."""

    def test_open_ffmpeg_raw_writer_raises_when_ffmpeg_missing(self, monkeypatch, tmp_path):
        # Arrange
        monkeypatch.setattr(
            'cuttle_patterns.visualization.video_utils.shutil.which', lambda _: None,
        )

        # Act / Assert
        with pytest.raises(OSError, match='ffmpeg not found'):
            open_ffmpeg_raw_writer(tmp_path / 'out.mp4', width=10, height=10, fps=5)

    def test_open_ffmpeg_raw_writer_creates_parent_dir_and_writes_valid_mp4(self, tmp_path):
        # Arrange
        output_path = tmp_path / 'nested' / 'out.mp4'
        width, height = 8, 8

        # Act
        writer = open_ffmpeg_raw_writer(output_path, width=width, height=height, fps=5)
        try:
            writer.stdin.write(b'\x00' * width * height * 3)
        finally:
            writer.stdin.close()
            writer.wait()

        # Assert
        assert output_path.parent.exists()
        assert writer.returncode == 0
        assert output_path.exists()
        probe = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(output_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert probe.stdout.strip() == f'{width},{height}'
