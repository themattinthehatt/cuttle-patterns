"""Extract-eligibility subcommand: per-frame eligibility tables for frame selection."""

import argparse
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, add_shard_arguments, resolve_dirs
from cuttle_patterns.preprocessing.extract.common import list_videos
from cuttle_patterns.preprocessing.extract.eligibility import (
    DEFAULT_BLACK_MAX,
    DEFAULT_JUMP_MAX,
    DEFAULT_ME_PERCENTILE,
    run_eligibility,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-eligibility subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-eligibility',
        help='compute which frames of each aligned video are eligible for selection',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    add_shard_arguments(parser)
    parser.add_argument(
        '--input-dir', type=Path, metavar='PATH',
        help=f'directory of aligned rectangle videos; defaults to '
        f'results_dir/{paths.RECTANGLES_RELPATH}',
    )
    parser.add_argument(
        '--pose-dir', type=Path, metavar='PATH', required=True,
        help='directory containing {video_name}.csv pose predictions (crop-frame coordinates)',
    )
    parser.add_argument('--videos', nargs='*', metavar='STEM', help='only these video stems')
    parser.add_argument('--overwrite', action='store_true', help='recompute existing tables')
    parser.add_argument('--jump-max', type=float, default=DEFAULT_JUMP_MAX,
                        help='largest allowed keypoint jump between neighboring frames, in px')
    parser.add_argument('--me-percentile', type=float, default=DEFAULT_ME_PERCENTILE,
                        help='drop frames above this motion-energy percentile')
    parser.add_argument('--black-max', type=float, default=DEFAULT_BLACK_MAX,
                        help='drop frames with at least this fraction of pure-black pixels')
    parser.set_defaults(handler=cmd_extract_eligibility)


def cmd_extract_eligibility(args: argparse.Namespace) -> None:
    """Compute and save per-video eligibility tables.

    Args:
        args: parsed command-line arguments
    """
    data_dir, results_dir = resolve_dirs(args)
    input_dir = (
        args.input_dir if args.input_dir is not None else results_dir / paths.RECTANGLES_RELPATH
    )
    video_paths = list_videos(input_dir, args.videos)
    if not video_paths:
        print(f'No videos found in {input_dir}.')
        return
    out_dir = results_dir / paths.ELIGIBILITY_RELPATH
    written = run_eligibility(
        video_paths, args.pose_dir, data_dir, out_dir, overwrite=args.overwrite,
        jump_max=args.jump_max, me_percentile=args.me_percentile, black_max=args.black_max,
        shard=args.shard, n_shards=args.n_shards,
    )
    print(f'wrote {len(written)} eligibility tables to {out_dir}')
    print('next: cuttle extract-classify --model-dir <classifier directory>')
