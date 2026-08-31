"""Predict subcommand: thin wrapper around BEAST's own `beast predict` CLI."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cli.cmd_train import INPUT_RELPATH, MODEL_RELPATH
from cuttle_patterns.config import load_config

DEFAULT_BATCH_SIZE = 32


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the predict subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'predict',
        help='run inference with a trained BEAST model via the beast CLI',
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        metavar='PATH',
        help='override the results directory from config',
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help=f'name of a model previously trained with cuttle train; looked up at '
        f'results_dir/{MODEL_RELPATH}/{{model_name}}',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        metavar='PATH',
        help=f'directory of images to run inference on, passed to beast predict as '
        f'--input; defaults to results_dir/{INPUT_RELPATH}',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        metavar='PATH',
        help='passed to beast predict as --output; if omitted, beast predict uses its '
        'own default (model_dir/image_predictions/{input_dir.stem})',
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='passed through to beast predict --batch-size',
    )
    parser.add_argument(
        '--save-latents', '-l',
        action='store_true',
        help='passed through to beast predict --save_latents',
    )
    parser.add_argument(
        '--save-reconstructions', '-r',
        action='store_true',
        help='passed through to beast predict --save_reconstructions',
    )
    parser.set_defaults(handler=cmd_predict)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference with a trained BEAST model by shelling out to `beast predict`.

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

    input_dir = args.input_dir if args.input_dir is not None else results_dir / INPUT_RELPATH
    model_dir = results_dir / MODEL_RELPATH / args.model_name

    if not model_dir.exists():
        print(f'Error: no model found at {model_dir}; train one first with cuttle train')
        sys.exit(1)

    argv = [
        'beast', 'predict',
        '--model', str(model_dir),
        '--input', str(input_dir),
        '--batch-size', str(args.batch_size),
    ]
    if args.output_dir is not None:
        argv += ['--output', str(args.output_dir)]
    if args.save_latents:
        argv.append('--save_latents')
    if args.save_reconstructions:
        argv.append('--save_reconstructions')

    print(f'Running: {" ".join(argv)}')
    result = subprocess.run(argv)
    if result.returncode != 0:
        sys.exit(result.returncode)
