"""Entry point for the cuttle-patterns CLI."""

import argparse
import importlib
import logging
from pathlib import Path


def main() -> None:
    """Entry point for the cuttle-patterns CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
    )

    parser = argparse.ArgumentParser(
        prog='cuttle',
        description='Cuttlefish pattern analysis pipeline.',
    )
    subparsers = parser.add_subparsers(dest='command', metavar='command')
    subparsers.required = True

    # discover and register all cmd_*.py modules in this directory
    cli_dir = Path(__file__).parent
    for module_path in sorted(cli_dir.glob('cmd_*.py')):
        module_name = module_path.stem
        module = importlib.import_module(f'cuttle_patterns.cli.{module_name}')
        module.register(subparsers)

    args = parser.parse_args()
    args.handler(args)
