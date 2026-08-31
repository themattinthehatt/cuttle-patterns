"""Serve subcommand: launch the interactive embedding explorer."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.config import load_config
from cuttle_patterns.dashboard.launch import DEFAULT_PORT, run_server


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the serve subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'serve',
        help='launch the interactive embedding explorer',
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        metavar='PATH',
        help='override the results directory from config',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=DEFAULT_PORT,
        help='port to serve the app and its frame-image route on',
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='do not open a browser tab automatically',
    )
    parser.set_defaults(handler=cmd_serve)


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch the embedding explorer and block until interrupted.

    Args:
        args: parsed command-line arguments
    """
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        try:
            config = load_config()
        except (FileNotFoundError, ValueError) as e:
            print(f'Error: {e}')
            sys.exit(1)
        results_dir = config.results_dir

    models_dir = results_dir / paths.BEAST_MODELS_RELPATH
    if not models_dir.is_dir():
        print(f'Error: no models found at {models_dir}; train one first with cuttle train')
        sys.exit(1)

    run_server(results_dir, port=args.port, show=not args.no_show)
