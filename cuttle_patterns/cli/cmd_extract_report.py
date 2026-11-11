"""Extract-report subcommand: diagnostics for a frame selection."""

import argparse
import json
import sys
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, resolve_dirs
from cuttle_patterns.preprocessing.extract.kmeans import SUMMARY_FILENAME
from cuttle_patterns.preprocessing.extract.report import format_report, run_report


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-report subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-report',
        help='diagnostics for a frame selection: allocation, coverage, contact sheets',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    parser.add_argument('--name', required=True, help='name of the selection')
    parser.add_argument(
        '--input-dir', type=Path, metavar='PATH',
        help=f'directory of aligned rectangle videos (for contact sheets); defaults to '
        f'results_dir/{paths.RECTANGLES_RELPATH}',
    )
    parser.add_argument('--sample', type=int, default=200_000,
                        help='eligible frames per class used for coverage and state metrics')
    parser.add_argument('--no-sheets', action='store_true', help='skip contact sheets')
    parser.add_argument('--seed', type=int, default=0)
    parser.set_defaults(handler=cmd_extract_report)


def cmd_extract_report(args: argparse.Namespace) -> None:
    """Write and print the selection report.

    Args:
        args: parsed command-line arguments
    """
    _, results_dir = resolve_dirs(args)
    selection_dir = results_dir / paths.SELECTIONS_RELPATH / args.name
    try:
        stage1 = json.loads((selection_dir / SUMMARY_FILENAME).read_text())['stage1']
    except FileNotFoundError:
        print(f'Error: no selection at {selection_dir}; run extract-kmeans-pooled first')
        sys.exit(1)
    input_dir = (
        args.input_dir if args.input_dir is not None else results_dir / paths.RECTANGLES_RELPATH
    )
    report = run_report(
        selection_dir, results_dir / paths.ELIGIBILITY_RELPATH,
        results_dir / paths.CLASSIFICATIONS_RELPATH / stage1['classifier_name'],
        results_dir / paths.EMBEDDINGS_RELPATH / stage1['embedding_name'], input_dir,
        sample=args.sample, seed=args.seed, sheets=not args.no_sheets,
    )
    print(format_report(report))
    print(f'wrote report.md, report.json and eval_sheets/ to {selection_dir}')
    print(f'next: cuttle extract-export --name {args.name}')
