"""Extract-classify subcommand: per-frame skin-pattern classification of full videos."""

import argparse
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, add_shard_arguments, resolve_dirs
from cuttle_patterns.preprocessing.extract.classify import DEFAULT_BATCH_SIZE, run_classify
from cuttle_patterns.preprocessing.extract.common import list_videos


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-classify subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-classify',
        help='classify the skin pattern of every frame of the crop videos',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    add_shard_arguments(parser)
    parser.add_argument(
        '--model-dir', type=Path, metavar='PATH', required=True,
        help='directory holding the two-head classifier weights and behaviors.json',
    )
    parser.add_argument(
        '--model-name', metavar='NAME',
        help=f'output subdirectory of results_dir/{paths.CLASSIFICATIONS_RELPATH}; defaults '
        f'to the model directory\'s name',
    )
    parser.add_argument(
        '--video-dir', type=Path, metavar='PATH',
        help='directory of 512x512 crop videos; defaults to the data directory',
    )
    parser.add_argument('--videos', nargs='*', metavar='STEM', help='only these video stems')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--save-features', action='store_true',
                        help='also save the 512-d penultimate features (float16 .npy)')
    parser.add_argument('--overwrite', action='store_true', help='reclassify finished videos')
    parser.set_defaults(handler=cmd_extract_classify)


def cmd_extract_classify(args: argparse.Namespace) -> None:
    """Classify every frame of every crop video.

    Args:
        args: parsed command-line arguments
    """
    data_dir, results_dir = resolve_dirs(args)
    video_dir = args.video_dir if args.video_dir is not None else data_dir
    video_paths = list_videos(video_dir, args.videos)
    if not video_paths:
        print(f'No videos found in {video_dir}.')
        return
    model_name = args.model_name or args.model_dir.name
    out_dir = results_dir / paths.CLASSIFICATIONS_RELPATH / model_name
    written = run_classify(
        video_paths, args.model_dir, out_dir, batch_size=args.batch_size,
        save_features=args.save_features, overwrite=args.overwrite, shard=args.shard,
        n_shards=args.n_shards,
    )
    print(f'wrote {len(written)} prediction tables to {out_dir}')
    print('next: cuttle extract-dino-cls')
