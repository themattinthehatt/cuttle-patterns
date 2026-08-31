"""Extract subcommand: select representative frames from aligned videos for BEAST."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.config import load_config
from cuttle_patterns.ingest import FILENAME_PATTERN, read_blank_frame_indices
from cuttle_patterns.preprocessing.extract import (
    DEFAULT_FRAMES_PER_VIDEO,
    MANIFEST_RELPATH,
    build_extraction_manifest,
    extract_video_frames,
)

INPUT_RELPATH = Path('rectangles')
OUTPUT_RELPATH = Path('beast_frames')


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract',
        help='select representative frames from aligned videos for BEAST training',
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        metavar='PATH',
        help='override the data directory from config',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        metavar='PATH',
        help='override the results directory from config',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        metavar='PATH',
        help=f'directory of aligned videos to select frames from; defaults to '
        f'results_dir/{INPUT_RELPATH}',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        metavar='PATH',
        help=f'directory to write per-video frame subdirectories into; defaults to '
        f'results_dir/{OUTPUT_RELPATH}',
    )
    parser.add_argument(
        '--pose-dir',
        type=Path,
        metavar='PATH',
        required=True,
        help='directory containing {video_name}.csv pose predictions (see '
        'cuttle_patterns.preprocessing.pose); a video with no matching pose file is '
        'skipped, since keypoint-likelihood filtering is required, not optional',
    )
    parser.add_argument(
        '--pose-path',
        type=Path,
        metavar='PATH',
        help='pose predictions CSV for --video-path; overrides --pose-dir lookup',
    )
    parser.add_argument(
        '--video-path',
        type=Path,
        metavar='PATH',
        help='process a single video instead of every video in input_dir',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip a video if {output_dir}/{video_name}/selected_frames.csv already exists',
    )
    parser.add_argument(
        '--frames-per-video', '-n',
        type=int,
        default=DEFAULT_FRAMES_PER_VIDEO,
        help='maximum number of frames to select per video',
    )
    parser.set_defaults(handler=cmd_extract)


def cmd_extract(args: argparse.Namespace) -> None:
    """Select and export representative frames from aligned videos.

    Args:
        args: parsed command-line arguments
    """
    if args.data_dir is not None and args.results_dir is not None:
        data_dir = args.data_dir
        results_dir = args.results_dir
    else:
        try:
            config = load_config()
        except (FileNotFoundError, ValueError) as e:
            print(f'Error: {e}')
            sys.exit(1)
        data_dir = args.data_dir if args.data_dir is not None else config.data_dir
        results_dir = args.results_dir if args.results_dir is not None else config.results_dir

    input_dir = args.input_dir if args.input_dir is not None else results_dir / INPUT_RELPATH
    output_dir = args.output_dir if args.output_dir is not None else results_dir / OUTPUT_RELPATH

    if args.video_path is not None:
        video_paths = [args.video_path]
    else:
        video_paths = sorted(
            p for p in input_dir.glob('*.mp4') if not p.stem.endswith('_overlay')
        )
        if not video_paths:
            print(f'No videos found in {input_dir}.')
            return

    rows = []
    for video_path in video_paths:
        save_dir = output_dir / video_path.stem
        if args.skip_existing and (save_dir / 'selected_frames.csv').exists():
            print(f'skipping {video_path} ({save_dir} already exists)')
            continue

        match = FILENAME_PATTERN.match(video_path.stem)
        if match is None:
            print(f'  skipping file with unexpected name: {video_path}')
            continue
        session_id = match['session_id']
        fish_id = match['fish_id']

        pose_path = (
            args.pose_path if args.pose_path is not None
            else args.pose_dir / f'{video_path.stem}.csv'
        )
        if not pose_path.exists():
            print(f'  no pose predictions at {pose_path}, skipping {video_path}')
            continue

        rect_csv_path = video_path.with_suffix('.csv')
        if not rect_csv_path.exists():
            print(f'  no rectangle geometry at {rect_csv_path}, skipping {video_path}')
            continue

        blank_frames_path = data_dir / f'{session_id}_{fish_id}_black_frames.txt'
        blank_frame_idxs = []
        if blank_frames_path.exists():
            blank_frame_idxs = read_blank_frame_indices(blank_frames_path)
        else:
            print(f'  no blank-frames file found for {video_path}')

        print(f'processing {video_path}...')
        save_dir, selected_idxs = extract_video_frames(
            video_path,
            output_dir,
            blank_frame_idxs=blank_frame_idxs,
            pose_path=pose_path,
            rect_csv_path=rect_csv_path,
            frames_per_video=args.frames_per_video,
        )
        print(f'  wrote {len(selected_idxs)} frames to {save_dir}')

        rows.extend(
            {
                'session_id': session_id,
                'fish_id': fish_id,
                'frame_idx': int(idx),
                'image_path': str(save_dir / f'img{str(idx).zfill(8)}.png'),
            }
            for idx in selected_idxs
        )

    if not rows:
        print('No frames extracted.')
        return

    manifest = build_extraction_manifest(rows)
    manifest_path = results_dir / MANIFEST_RELPATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    print(f'Manifest written to {manifest_path}')
