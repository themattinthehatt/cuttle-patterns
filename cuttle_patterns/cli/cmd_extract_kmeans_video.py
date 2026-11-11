"""Extract-kmeans-video subcommand: stage 1 of class-quota frame selection."""

import argparse
import sys

import yaml

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, resolve_dirs
from cuttle_patterns.preprocessing.extract.classify import PROVENANCE_FILENAME
from cuttle_patterns.preprocessing.extract.common import get_day
from cuttle_patterns.preprocessing.extract.kmeans import (
    DEFAULT_FINE_MULT,
    DEFAULT_QUOTA_POWER,
    run_kmeans_video,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-kmeans-video subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-kmeans-video',
        help='stage 1 of frame selection: per-video, per-class k-means with class quotas',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    parser.add_argument('--name', required=True,
                        help='name of this selection (used for the output directories)')
    parser.add_argument('--classifier-name', required=True,
                        help=f'subdirectory of results_dir/{paths.CLASSIFICATIONS_RELPATH}')
    parser.add_argument('--embedding-name', default='dinov3_vits16',
                        help=f'subdirectory of results_dir/{paths.EMBEDDINGS_RELPATH}')
    parser.add_argument('--budget', type=int, required=True,
                        help='total number of frames to select across all videos')
    parser.add_argument('--quota-power', type=float, default=DEFAULT_QUOTA_POWER,
                        help='class quotas are proportional to count**power (0.5 = square root, '
                        '0 = uniform, 1 = proportional)')
    parser.add_argument('--fine-mult', type=float, default=DEFAULT_FINE_MULT,
                        help='stage-1 clusters = this x a video\'s share of the class quota')
    parser.add_argument('--exclude-days', type=int, nargs='*', default=[1], metavar='DAY',
                        help='recording days to leave out of the selection')
    parser.add_argument('--videos', nargs='*', metavar='STEM', help='only these video stems')
    parser.add_argument('--skip-missing', action='store_true',
                        help='skip videos lacking classifier or embedding files instead of '
                        'stopping')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true',
                        help='recompute existing per-video files (needed if quotas change)')
    parser.set_defaults(handler=cmd_extract_kmeans_video)


def cmd_extract_kmeans_video(args: argparse.Namespace) -> None:
    """Run stage 1 of the class-quota selection.

    Args:
        args: parsed command-line arguments
    """
    _, results_dir = resolve_dirs(args)
    elig_dir = results_dir / paths.ELIGIBILITY_RELPATH
    clf_dir = results_dir / paths.CLASSIFICATIONS_RELPATH / args.classifier_name
    emb_dir = results_dir / paths.EMBEDDINGS_RELPATH / args.embedding_name
    out_dir = results_dir / paths.KMEANS_VIDEO_RELPATH / args.name

    stems = sorted(p.stem for p in elig_dir.glob('*.parquet'))
    if args.videos:
        stems = [s for s in stems if s in set(args.videos)]
    stems = [s for s in stems if get_day(s) not in args.exclude_days]
    missing = [s for s in stems
               if not (clf_dir / f'{s}.parquet').exists() or not (emb_dir / f'{s}.npy').exists()]
    if missing and not args.skip_missing:
        print(f'Error: {len(missing)} videos lack classifier or embedding files '
              f'(e.g. {missing[0]}); finish those stages or pass --skip-missing')
        sys.exit(1)
    stems = [s for s in stems if s not in set(missing)]
    if not stems:
        print(f'No videos to process (eligibility tables in {elig_dir}).')
        return
    try:
        classes = yaml.safe_load((clf_dir / PROVENANCE_FILENAME).read_text())['classes']
        config = run_kmeans_video(
            stems, elig_dir, clf_dir, emb_dir, out_dir, classes, args.budget,
            classifier_name=args.classifier_name, embedding_name=args.embedding_name,
            quota_power=args.quota_power, fine_mult=args.fine_mult, seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        sys.exit(1)
    print(f'class quotas: {dict(zip(config["classes"], config["quotas"], strict=True))}')
    print(f'wrote stage-1 results for {len(stems)} videos to {out_dir}')
    print(f'next: cuttle extract-kmeans-pooled --name {args.name}')
