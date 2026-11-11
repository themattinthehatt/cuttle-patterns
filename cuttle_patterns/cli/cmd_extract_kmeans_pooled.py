"""Extract-kmeans-pooled subcommand: stage 2 of class-quota frame selection."""

import argparse
import sys

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, resolve_dirs
from cuttle_patterns.preprocessing.extract.kmeans import DEFAULT_PICKS_PER_GROUP, run_kmeans_pooled


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-kmeans-pooled subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-kmeans-pooled',
        help='stage 2 of frame selection: pooled k-means across videos -> selection manifest',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    parser.add_argument('--name', required=True,
                        help='name of the selection (the --name given to extract-kmeans-video)')
    parser.add_argument('--picks-per-group', type=int, default=DEFAULT_PICKS_PER_GROUP,
                        help='centroids drawn per stage-2 group (sets the number of groups)')
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(handler=cmd_extract_kmeans_pooled)


def cmd_extract_kmeans_pooled(args: argparse.Namespace) -> None:
    """Run stage 2 of the class-quota selection and write the selection manifest.

    Args:
        args: parsed command-line arguments
    """
    _, results_dir = resolve_dirs(args)
    out_dir = results_dir / paths.SELECTIONS_RELPATH / args.name
    try:
        manifest = run_kmeans_pooled(
            results_dir / paths.KMEANS_VIDEO_RELPATH / args.name, out_dir,
            picks_per_group=args.picks_per_group, seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)
    print(f'selected {len(manifest)} frames from {manifest["video"].nunique()} videos')
    print(f'wrote {out_dir / "manifest.parquet"}')
    print(f'next: cuttle extract-report --name {args.name}, then '
          f'cuttle extract-export --name {args.name}')
