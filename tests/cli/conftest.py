"""Shared fixtures for cuttle_patterns.cli tests."""

import argparse
from collections.abc import Callable

import pytest


@pytest.fixture
def parse_cli() -> Callable[..., argparse.Namespace]:
    """Return a factory that parses argv with one command's `register` function.

    Returns:
        a callable taking (register, argv) and returning the parsed namespace, including the
        `handler` the command wires up.
    """

    def _parse(register: Callable, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        register(parser.add_subparsers(dest='command'))
        return parser.parse_args(argv)

    return _parse


@pytest.fixture
def no_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make config loading fail as if ~/.cuttle-patterns/config.yaml did not exist."""

    def _raise() -> None:
        raise FileNotFoundError('no config file found at ~/.cuttle-patterns/config.yaml')

    monkeypatch.setattr('cuttle_patterns.cli.common.load_config', _raise)
