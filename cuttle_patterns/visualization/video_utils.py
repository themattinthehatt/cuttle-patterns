"""Shared helpers for building side-by-side QC review clips (mp4 + matching gif).

Used by `scripts/make_mantle_clip.py` and `scripts/make_reconstruction_clip.py`, neither of
which is yet promoted into the `cuttle` CLI (run directly).
"""

import shutil
import subprocess
from pathlib import Path

GIF_FPS = 15
GIF_SCALE_WIDTH = 320
VIDEO_CRF = 18
VIDEO_PRESET = 'medium'


def build_gif_command(
    mp4_path: Path,
    gif_path: Path,
    fps: int = GIF_FPS,
    scale_width: int = GIF_SCALE_WIDTH,
) -> list[str]:
    """Build the ffmpeg command that converts a clip to a palette-optimized gif.

    Args:
        mp4_path: path to the source mp4.
        gif_path: path to write the gif to.
        fps: gif frame rate.
        scale_width: gif width in pixels; height is scaled to preserve aspect ratio.

    Returns:
        ffmpeg command as a list of args.
    """
    filter_complex = (
        f'fps={fps},scale={scale_width}:-1:flags=lanczos,split[s0][s1];'
        f'[s0]palettegen[p];[s1][p]paletteuse'
    )
    return [
        'ffmpeg', '-y', '-i', str(mp4_path),
        '-vf', filter_complex,
        '-loop', '0',
        str(gif_path),
    ]


def open_ffmpeg_raw_writer(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int = VIDEO_CRF,
    preset: str = VIDEO_PRESET,
) -> subprocess.Popen:
    """Launch an ffmpeg subprocess that reads raw BGR24 frames on stdin and writes H.264.

    Same piped-frames approach as `preprocessing.overlay._open_ffmpeg_writer`: cv2.VideoWriter's
    ffmpeg backend has no H.264 encoder available in some environments (falls back to the far
    less efficient mp4v/MPEG-4 Part 2 codec).

    Args:
        output_path: path to write the encoded mp4 to; parent directory is created if needed.
        width: frame width in pixels.
        height: frame height in pixels.
        fps: output frame rate.
        crf: x264 constant rate factor (lower = higher quality/larger file).
        preset: x264 encoding preset (speed/compression trade-off).

    Returns:
        the running subprocess, with `stdin` open for raw frame bytes.

    Raises:
        OSError: if the ffmpeg binary is not found on PATH.
    """
    if shutil.which('ffmpeg') is None:
        raise OSError('ffmpeg not found on PATH; required to write review clips')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', 'yuv420p',
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)
