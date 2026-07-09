"""Setup subcommand: configure data_dir and results_dir in the config file."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns.config import DEFAULT_CONFIG_PATH, Config, load_config, save_config


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the setup subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'setup',
        help='configure data_dir and results_dir in ~/.cuttle-patterns/config.yaml',
    )
    parser.set_defaults(handler=cmd_setup)


def cmd_setup(args: argparse.Namespace | None = None) -> None:
    """Prompt for data_dir/results_dir and write them to the config file.

    Args:
        args: parsed command-line arguments (unused, accepted for handler symmetry)
    """
    config_path = DEFAULT_CONFIG_PATH

    if config_path.exists():
        existing = load_config(config_path)
        print(f'Config already exists at {config_path}')
        print(f'Current data_dir: {existing.data_dir}')
        print(f'Current results_dir: {existing.results_dir}')
        answer = input('Overwrite? [y/N] ').strip().lower()
        if answer != 'y':
            print('Aborted.')
            sys.exit(0)

    data_dir_str = input('Enter data_dir (where raw/derived video data lives): ').strip()
    if not data_dir_str:
        print('Error: data_dir cannot be empty.')
        sys.exit(1)

    results_dir_str = input(
        'Enter results_dir (where manifests, embeddings, and checkpoints will be written): '
    ).strip()
    if not results_dir_str:
        print('Error: results_dir cannot be empty.')
        sys.exit(1)

    config = Config(
        data_dir=Path(data_dir_str).expanduser().resolve(),
        results_dir=Path(results_dir_str).expanduser().resolve(),
    )
    save_config(config, config_path)
    print(f'Config written to {config_path}')
