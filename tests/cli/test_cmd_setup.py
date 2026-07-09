"""Tests for cuttle_patterns.cli.cmd_setup."""

from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_setup import cmd_setup
from cuttle_patterns.config import load_config


class TestCmdSetup:
    """Test the function cmd_setup."""

    def test_cmd_setup_writes_new_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        monkeypatch.setattr('cuttle_patterns.cli.cmd_setup.DEFAULT_CONFIG_PATH', config_path)
        data_dir = tmp_path / 'data'
        results_dir = tmp_path / 'results'
        inputs = iter([str(data_dir), str(results_dir)])
        monkeypatch.setattr('builtins.input', lambda _prompt='': next(inputs))

        # Act
        cmd_setup()

        # Assert
        config = load_config(config_path)
        assert config.data_dir == data_dir
        assert config.results_dir == results_dir
        assert f'Config written to {config_path}' in capsys.readouterr().out

    def test_cmd_setup_empty_data_dir_exits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        monkeypatch.setattr('cuttle_patterns.cli.cmd_setup.DEFAULT_CONFIG_PATH', config_path)
        monkeypatch.setattr('builtins.input', lambda _prompt='': '')

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_setup()
        assert exc_info.value.code == 1

    def test_cmd_setup_declining_overwrite_aborts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('data_dir: /old/data\nresults_dir: /old/results\n')
        monkeypatch.setattr('cuttle_patterns.cli.cmd_setup.DEFAULT_CONFIG_PATH', config_path)
        monkeypatch.setattr('builtins.input', lambda _prompt='': 'n')

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_setup()
        assert exc_info.value.code == 0
        assert 'Aborted.' in capsys.readouterr().out

    def test_cmd_setup_overwrite_replaces_existing_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('data_dir: /old/data\nresults_dir: /old/results\n')
        monkeypatch.setattr('cuttle_patterns.cli.cmd_setup.DEFAULT_CONFIG_PATH', config_path)
        new_data_dir = tmp_path / 'new_data'
        new_results_dir = tmp_path / 'new_results'
        inputs = iter(['y', str(new_data_dir), str(new_results_dir)])
        monkeypatch.setattr('builtins.input', lambda _prompt='': next(inputs))

        # Act
        cmd_setup()

        # Assert
        config = load_config(config_path)
        assert config.data_dir == new_data_dir
        assert config.results_dir == new_results_dir
