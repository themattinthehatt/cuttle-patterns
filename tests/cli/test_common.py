"""Tests for cuttle_patterns.cli.common."""

import argparse
from pathlib import Path

import pytest

from cuttle_patterns import paths
from cuttle_patterns.cli.common import resolve_dirs, resolve_model_dir, resolve_results_dir


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


class TestResolveModelDir:
    """Test the function resolve_model_dir."""

    def test_resolve_model_dir_joins_models_path_and_name(self):
        out = resolve_model_dir(Path('/r'), 'm1', must_exist=False)
        assert out == Path('/r') / paths.BEAST_MODELS_RELPATH / 'm1'

    def test_resolve_model_dir_missing_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            resolve_model_dir(tmp_path, 'm1')
        assert exc_info.value.code == 1
        assert 'no model found' in capsys.readouterr().out

    def test_resolve_model_dir_missing_allowed(self, tmp_path):
        out = resolve_model_dir(tmp_path, 'm1', must_exist=False)
        assert out == tmp_path / paths.BEAST_MODELS_RELPATH / 'm1'

    def test_resolve_model_dir_existing_returned(self, tmp_path):
        (tmp_path / paths.BEAST_MODELS_RELPATH / 'm1').mkdir(parents=True)
        assert resolve_model_dir(tmp_path, 'm1').name == 'm1'
