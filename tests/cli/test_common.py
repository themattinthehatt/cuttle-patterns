"""Tests for cuttle_patterns.cli.common."""

import argparse
from pathlib import Path

import pytest

from cuttle_patterns.cli.common import resolve_dirs, resolve_results_dir


def _args(data_dir: Path | None = None, results_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(data_dir=data_dir, results_dir=results_dir)


class TestResolveResultsDir:
    """Test the function resolve_results_dir."""

    def test_resolve_results_dir_override_skips_config(self, no_config):
        assert resolve_results_dir(_args(results_dir=Path('/r'))) == Path('/r')

    def test_resolve_results_dir_missing_config_exits(self, no_config, capsys):
        with pytest.raises(SystemExit) as exc_info:
            resolve_results_dir(_args())
        assert exc_info.value.code == 1
        assert 'Error:' in capsys.readouterr().out


class TestResolveDirs:
    """Test the function resolve_dirs."""

    def test_resolve_dirs_both_overrides_skip_config(self, no_config):
        assert resolve_dirs(_args(Path('/d'), Path('/r'))) == (Path('/d'), Path('/r'))

    def test_resolve_dirs_missing_config_exits(self, no_config):
        with pytest.raises(SystemExit):
            resolve_dirs(_args(data_dir=Path('/d')))
