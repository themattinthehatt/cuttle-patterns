"""Extract-export subcommand: write the selected frames as PNGs for BEAST training."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.common import add_dir_arguments, resolve_dirs
from cuttle_patterns.preprocessing.extract.export import run_export
from cuttle_patterns.preprocessing.extract.kmeans import MANIFEST_FILENAME


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract-export subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract-export',
        help='write the selected frames as PNGs (and the extract manifest) for BEAST',
        formatter_class=DefaultsHelpFormatter,
    )
    add_dir_arguments(parser)
    parser.add_argument('--name', required=True, help='name of the selection')
    parser.add_argument(
        '--input-dir', type=Path, metavar='PATH',
        help=f'directory of aligned rectangle videos to read frames from; defaults to '
        f'results_dir/{paths.RECTANGLES_RELPATH}',
    )
    parser.add_argument(
        '--output-dir', type=Path, metavar='PATH',
        help=f'directory to write per-video frame subdirectories into; defaults to '
        f'results_dir/{paths.BEAST_FRAMES_RELPATH}',
    )
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip a video if its selected_frames.csv already exists')
    parser.set_defaults(handler=cmd_extract_export)


def cmd_extract_export(args: argparse.Namespace) -> None:
    """Export the selected frames and write the extract manifest.

    Args:
        args: parsed command-line arguments
    """
    _, results_dir = resolve_dirs(args)
    manifest_path = results_dir / paths.SELECTIONS_RELPATH / args.name / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(f'Error: no selection manifest at {manifest_path}; run extract-kmeans-pooled first')
        sys.exit(1)
    input_dir = (
        args.input_dir if args.input_dir is not None else results_dir / paths.RECTANGLES_RELPATH
    )
    output_dir = (
        args.output_dir if args.output_dir is not None
        else results_dir / paths.BEAST_FRAMES_RELPATH
    )
    extract_manifest = run_export(
        pd.read_parquet(manifest_path), input_dir, output_dir, skip_existing=args.skip_existing,
    )
    if extract_manifest.empty:
        print('No frames exported.')
        return
    out_path = results_dir / paths.EXTRACT_MANIFEST_RELPATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extract_manifest.to_parquet(out_path, index=False)
    print(f'exported {len(extract_manifest)} frames to {output_dir}; manifest at {out_path}')
    print(
        'next: cuttle train --config configs/beast_resnet_ae.yaml --model-name <model-name> '
        '(or cuttle embed for a frozen pretrained embedder, no training required)'
    )
