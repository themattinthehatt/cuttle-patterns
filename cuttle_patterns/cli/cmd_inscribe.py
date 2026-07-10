"""Inscribe subcommand: build egocentrically-aligned crop videos + geometry CSVs."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns.config import load_config
from cuttle_patterns.ingest import find_raw_videos
from cuttle_patterns.preprocessing.align import (
    DEFAULT_CANONICAL_HEIGHT,
    DEFAULT_SMOOTHING_SIGMA,
    DEFAULT_SMOOTHING_WINDOW,
    align_video,
)
from cuttle_patterns.preprocessing.inscribe import DEFAULT_ASPECT_RATIO, DEFAULT_THRESHOLD

OUTPUT_RELPATH = Path('rectangles')
POSE_RELPATH = Path('pose')


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the inscribe subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'inscribe',
        help='inscribe egocentric rectangles and write aligned crop videos',
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
        '--output-dir',
        type=Path,
        metavar='PATH',
        help=f'directory to write {{video_name}}.mp4/.csv into; defaults to '
        f'results_dir/{OUTPUT_RELPATH}',
    )
    parser.add_argument(
        '--video-path',
        type=Path,
        metavar='PATH',
        help='process a single video instead of every raw video in data_dir',
    )
    parser.add_argument(
        '--pose-dir',
        type=Path,
        metavar='PATH',
        help=f'directory containing {{video_name}}.csv pose predictions (see '
        f'cuttle_patterns.preprocessing.pose); defaults to results_dir/{POSE_RELPATH}. '
        f'Videos with no matching pose file fall back to the Phase 2a PCA-based path.',
    )
    parser.add_argument(
        '--pose-path',
        type=Path,
        metavar='PATH',
        help='pose predictions CSV for --video-path; overrides --pose-dir lookup',
    )
    parser.add_argument(
        '--thresh',
        type=int,
        default=DEFAULT_THRESHOLD,
        help='pixel intensities <= this value are candidate background pixels',
    )
    parser.add_argument(
        '--aspect',
        type=float,
        default=DEFAULT_ASPECT_RATIO,
        help='fixed width-to-height ratio of the inscribed rectangle',
    )
    parser.add_argument(
        '--canonical-height',
        type=int,
        default=DEFAULT_CANONICAL_HEIGHT,
        help='output crop height in pixels; width is round(aspect * height)',
    )
    smoothing_group = parser.add_mutually_exclusive_group()
    smoothing_group.add_argument(
        '--smoothing-window',
        type=int,
        default=None,
        help=f'rolling-median window (frames) for smoothing rectangle geometry; 1 '
        f'disables smoothing; defaults to {DEFAULT_SMOOTHING_WINDOW} unless '
        f'--smoothing-sigma is given',
    )
    smoothing_group.add_argument(
        '--smoothing-sigma',
        type=float,
        nargs='?',
        const=DEFAULT_SMOOTHING_SIGMA,
        default=None,
        help=f'gaussian-filter sigma (frames) for smoothing rectangle geometry instead '
        f'of the rolling median; defaults to {DEFAULT_SMOOTHING_SIGMA} if given with no '
        f'value',
    )
    parser.set_defaults(handler=cmd_inscribe)


def cmd_inscribe(args: argparse.Namespace) -> None:
    """Build aligned crop videos and rectangle-geometry CSVs for raw videos.

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

    output_dir = args.output_dir if args.output_dir is not None else results_dir / OUTPUT_RELPATH
    pose_dir = args.pose_dir if args.pose_dir is not None else results_dir / POSE_RELPATH

    if args.video_path is not None:
        video_paths = [args.video_path]
    else:
        try:
            video_paths = find_raw_videos(data_dir)
        except FileNotFoundError as e:
            print(f'Error: {e}')
            sys.exit(1)
        if not video_paths:
            print(f'No raw videos found in {data_dir}.')
            return

    for video_path in video_paths:
        pose_path = (
            args.pose_path if args.pose_path is not None else pose_dir / f'{video_path.stem}.csv'
        )
        if not pose_path.exists():
            print(f'  no pose predictions at {pose_path}, using PCA-based inscription')
            pose_path = None

        print(f'processing {video_path}...')
        video_out_path, csv_out_path = align_video(
            video_path,
            output_dir,
            thresh=args.thresh,
            aspect=args.aspect,
            canonical_height=args.canonical_height,
            pose_path=pose_path,
            smoothing_window=args.smoothing_window,
            smoothing_sigma=args.smoothing_sigma,
        )
        print(f'  wrote {video_out_path}')
        print(f'  wrote {csv_out_path}')
