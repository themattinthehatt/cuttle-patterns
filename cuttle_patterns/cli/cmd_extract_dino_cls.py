"""Extract-dino-cls subcommand: per-frame DINOv3 CLS features of full videos."""

import argparse
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, add_shard_arguments, resolve_dirs
from cuttle_patterns.embedders.dinov3 import ARCH_TO_HF_ID
from cuttle_patterns.preprocessing.extract.common import list_videos
from cuttle_patterns.preprocessing.extract.dino_cls import (
    DEFAULT_ARCH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_RESOLUTION,
    embedding_dir_name,
    run_dino_cls,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-dino-cls subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-dino-cls',
        help='compute DINOv3 CLS features for every frame of the aligned rectangle videos',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    add_shard_arguments(parser)
    parser.add_argument('--arch', default=DEFAULT_ARCH, choices=list(ARCH_TO_HF_ID))
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION,
                        help='square input side length; must be a multiple of 16')
    parser.add_argument(
        '--input-dir', type=Path, metavar='PATH',
        help=f'directory of aligned rectangle videos; defaults to '
        f'results_dir/{paths.RECTANGLES_RELPATH}',
    )
    parser.add_argument('--videos', nargs='*', metavar='STEM', help='only these video stems')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--overwrite', action='store_true', help='re-embed finished videos')
    parser.set_defaults(handler=cmd_extract_dino_cls)


def cmd_extract_dino_cls(args: argparse.Namespace) -> None:
    """Compute DINOv3 CLS features for every frame of every rectangle video.

    Args:
        args: parsed command-line arguments
    """
    _, results_dir = resolve_dirs(args)
    input_dir = (
        args.input_dir if args.input_dir is not None else results_dir / paths.RECTANGLES_RELPATH
    )
    video_paths = list_videos(input_dir, args.videos)
    if not video_paths:
        print(f'No videos found in {input_dir}.')
        return
    name = embedding_dir_name(args.arch)
    out_dir = results_dir / paths.EMBEDDINGS_RELPATH / name
    written = run_dino_cls(
        video_paths, out_dir, arch=args.arch, resolution=args.resolution,
        batch_size=args.batch_size, overwrite=args.overwrite, shard=args.shard,
        n_shards=args.n_shards,
    )
    print(f'wrote {len(written)} feature arrays to {out_dir}')
    print(
        'next: cuttle extract-kmeans-video --name <name> --classifier-name <classifier name> '
        f'--embedding-name {name} --budget <n frames>'
    )
