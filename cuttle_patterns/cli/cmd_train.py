"""Train subcommand: thin wrapper around BEAST's own `beast train` CLI."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.config import load_config


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the train subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'train',
        help='train a BEAST model via the beast CLI',
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        metavar='PATH',
        help='override the results directory from config',
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        metavar='PATH',
        required=True,
        help='path to a BEAST model configuration YAML (see configs/)',
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help=f'name for this model; saved to results_dir/{paths.BEAST_MODELS_RELPATH}/'
        f'{{model_name}}',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        metavar='PATH',
        help=f'training frames directory, passed to beast train as --data; defaults to '
        f'results_dir/{paths.BEAST_FRAMES_RELPATH}',
    )
    parser.add_argument(
        '--gpus',
        type=int,
        help='number of GPUs to use, passed through to beast train --gpus',
    )
    parser.add_argument(
        '--nodes',
        type=int,
        help='number of nodes to use, passed through to beast train --nodes',
    )
    parser.add_argument(
        '--overrides',
        nargs='*',
        metavar='KEY=VALUE',
        help='config value overrides, passed through to beast train --overrides',
    )
    parser.set_defaults(handler=cmd_train)


def cmd_train(args: argparse.Namespace) -> None:
    """Train a BEAST model by shelling out to `beast train`.

    Args:
        args: parsed command-line arguments
    """
    if shutil.which('beast') is None:
        print('Error: beast not found on PATH; is beast-backbones installed?')
        sys.exit(1)

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        try:
            config = load_config()
        except (FileNotFoundError, ValueError) as e:
            print(f'Error: {e}')
            sys.exit(1)
        results_dir = config.results_dir

    input_dir = (
        args.input_dir if args.input_dir is not None else results_dir / paths.BEAST_FRAMES_RELPATH
    )
    model_dir = results_dir / paths.BEAST_MODELS_RELPATH / args.model_name

    if model_dir.exists() and any(model_dir.iterdir()):
        print(
            f'Warning: {model_dir} already exists and is non-empty; beast train does not '
            f'support resuming, so this run will write alongside whatever is already there.'
        )

    argv = [
        'beast', 'train',
        '--config', str(args.config),
        '--data', str(input_dir),
        '--output', str(model_dir),
    ]
    if args.gpus is not None:
        argv += ['--gpus', str(args.gpus)]
    if args.nodes is not None:
        argv += ['--nodes', str(args.nodes)]
    if args.overrides:
        argv += ['--overrides', *args.overrides]

    print(f'Running: {" ".join(argv)}')
    result = subprocess.run(argv)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print(f'Model saved to {model_dir}')
