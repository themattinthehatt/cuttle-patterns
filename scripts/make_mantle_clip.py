"""Cut a side-by-side full/crop review clip (and matching gif) for one time range.

Given a video name and a start/end time, trims both `{video_name}_overlay.mp4` (the full
frame with pose overlay drawn) and `{video_name}.mp4` (the raw inscribed rectangle crop) from
`{results_dir}/rectangles/`, and stacks them side by side: full video on the left at its
native height, crop video on the right scaled to half that height and padded with black to
vertically center it. Writes the result plus a matching gif to `{results_dir}/media/`.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI; run directly, e.g.:

    python scripts/make_mantle_clip.py --video-name Day1_Tank2_Cuttle1_Resident_Crop \
        --start 50 --end 55
"""

import argparse
import subprocess
from pathlib import Path

from cuttle_patterns.config import load_config
from cuttle_patterns.visualization.video_utils import (
    VIDEO_CRF,
    VIDEO_PRESET,
    build_gif_command,
)


def format_seconds_for_filename(seconds: float) -> str:
    """Format a time in seconds for use in a filename (no trailing '.0' for whole numbers)."""
    return str(int(seconds)) if seconds == int(seconds) else f'{seconds:g}'


def probe_video_height(video_path: Path) -> int:
    """Read a video's frame height via ffprobe.

    Args:
        video_path: path to the video file.

    Returns:
        frame height in pixels.
    """
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=height', '-of', 'csv=p=0', str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def build_clip_command(
    full_path: Path,
    crop_path: Path,
    start: float,
    end: float,
    full_height: int,
    output_path: Path,
) -> list[str]:
    """Build the ffmpeg command that trims and hstacks the full and crop videos.

    The full video is left at its native height. The crop video is scaled (preserving aspect
    ratio) to half that height, then padded with black top/bottom to vertically center it
    within the full video's height, so the two share a common height for stacking.

    Args:
        full_path: path to the full-frame pose-overlay video.
        crop_path: path to the raw inscribed-rectangle crop video.
        start: clip start time, in seconds.
        end: clip end time, in seconds.
        full_height: native height (px) of the full video; also the final canvas height.
        output_path: where to write the combined mp4.

    Returns:
        ffmpeg command as a list of args.
    """
    duration = end - start
    crop_height = full_height // 2
    pad_y_offset = (full_height - crop_height) // 2
    filter_complex = (
        f'[1:v]scale=-2:{crop_height}[crop_scaled];'
        f'[crop_scaled]pad=iw:{full_height}:0:{pad_y_offset}:black[crop_padded];'
        f'[0:v][crop_padded]hstack=inputs=2[v]'
    )
    return [
        'ffmpeg', '-y',
        '-ss', str(start), '-t', str(duration), '-i', str(full_path),
        '-ss', str(start), '-t', str(duration), '-i', str(crop_path),
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-c:v', 'libx264', '-crf', str(VIDEO_CRF), '-preset', VIDEO_PRESET,
        str(output_path),
    ]


def main() -> None:
    """Parse arguments and write the trimmed side-by-side clip and matching gif."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--video-name',
        required=True,
        help='rectangle video stem, e.g. Day1_Tank2_Cuttle1_Resident_Crop',
    )
    parser.add_argument('--start', type=float, required=True, help='clip start time, in seconds')
    parser.add_argument('--end', type=float, required=True, help='clip end time, in seconds')
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='defaults to results_dir from the cuttle config',
    )
    args = parser.parse_args()

    if args.end <= args.start:
        raise ValueError(f'--end ({args.end}) must be greater than --start ({args.start})')

    results_dir = args.results_dir if args.results_dir is not None else load_config().results_dir

    full_path = results_dir / 'rectangles' / f'{args.video_name}_overlay.mp4'
    crop_path = results_dir / 'rectangles' / f'{args.video_name}.mp4'
    for path in (full_path, crop_path):
        if not path.exists():
            raise FileNotFoundError(f'expected video not found: {path}')

    output_dir = results_dir / 'media'
    output_dir.mkdir(parents=True, exist_ok=True)

    start_str = format_seconds_for_filename(args.start)
    end_str = format_seconds_for_filename(args.end)
    mp4_path = output_dir / f'{args.video_name}_mantle_{start_str}-{end_str}.mp4'
    gif_path = mp4_path.with_suffix('.gif')

    full_height = probe_video_height(full_path)
    clip_cmd = build_clip_command(
        full_path, crop_path, args.start, args.end, full_height, mp4_path,
    )
    subprocess.run(clip_cmd, check=True)
    print(f'wrote {mp4_path}')

    gif_cmd = build_gif_command(mp4_path, gif_path)
    subprocess.run(gif_cmd, check=True)
    print(f'wrote {gif_path}')


if __name__ == '__main__':
    main()
