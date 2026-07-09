"""Ingest subcommand: scan raw videos and build a manifest of sessions/fish/frames."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns.config import load_config
from cuttle_patterns.ingest import MANIFEST_RELPATH, build_manifest


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ingest subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'ingest',
        help='scan raw videos and build a manifest of sessions/fish/frame counts',
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
    parser.set_defaults(handler=cmd_ingest)


def cmd_ingest(args: argparse.Namespace) -> None:
    """Scan raw videos and write a manifest describing them.

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

    print(f'Scanning {data_dir} for raw videos...')
    try:
        manifest = build_manifest(data_dir)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        sys.exit(1)

    if manifest.empty:
        print(f'No raw videos found in {data_dir}.')
        return

    manifest_path = results_dir / MANIFEST_RELPATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)

    n_sessions = manifest['session_id'].nunique()
    n_videos = len(manifest)
    total_frames = int(manifest['n_frames'].sum())
    total_blank_frames = int(manifest['n_blank_frames'].sum())

    print(f'Ingested {n_videos} videos across {n_sessions} sessions.')
    print(f'Total frames: {total_frames} ({total_blank_frames} flagged blank).')
    print(f'Manifest written to {manifest_path}')
