"""Argument helpers shared by the `cuttle` commands.

Not a command itself (`main.py` only registers `cmd_*.py` modules).
"""

import argparse
import sys
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.config import load_config


def add_dir_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the `--data-dir` / `--results-dir` overrides.

    Args:
        parser: parser to add the arguments to.
    """
    parser.add_argument(
        '--data-dir', type=Path, metavar='PATH', help='override the data directory from config',
    )
    parser.add_argument(
        '--results-dir', type=Path, metavar='PATH',
        help='override the results directory from config',
    )


def add_shard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add `--shard` / `--n-shards`, for splitting videos across several processes.

    Args:
        parser: parser to add the arguments to.
    """
    parser.add_argument('--shard', type=int, default=0, help='this process\'s shard index')
    parser.add_argument('--n-shards', type=int, default=1, help='total number of shards')


def resolve_results_dir(args: argparse.Namespace) -> Path:
    """Resolve the results directory from the override or the config file.

    The config file is only read when `--results-dir` was not given. Prints an error and exits
    with status 1 if it is needed but unusable.

    Args:
        args: parsed arguments with `results_dir`.

    Returns:
        the results directory.
    """
    if args.results_dir is not None:
        return args.results_dir
    try:
        return load_config().results_dir
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        sys.exit(1)


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the data and results directories from overrides or the config file.

    Prints an error and exits with status 1 if the config file is needed but unusable.

    Args:
        args: parsed arguments with `data_dir` and `results_dir`.

    Returns:
        `(data_dir, results_dir)`.
    """
    if args.data_dir is not None and args.results_dir is not None:
        return args.data_dir, args.results_dir
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        sys.exit(1)
    data_dir = args.data_dir if args.data_dir is not None else config.data_dir
    results_dir = args.results_dir if args.results_dir is not None else config.results_dir
    return data_dir, results_dir


def resolve_model_dir(results_dir: Path, model_name: str, must_exist: bool = True) -> Path:
    """Directory of a named BEAST model.

    Prints an error and exits with status 1 if `must_exist` and the directory is missing.

    Args:
        results_dir: results directory.
        model_name: name of the model.
        must_exist: whether the model directory must already exist. Commands that create the
            model directory (`train`, `embed`) pass False.

    Returns:
        `results_dir/{BEAST_MODELS_RELPATH}/{model_name}`.
    """
    model_dir = results_dir / paths.BEAST_MODELS_RELPATH / model_name
    if must_exist and not model_dir.is_dir():
        print(f'Error: no model found at {model_dir}')
        sys.exit(1)
    return model_dir
