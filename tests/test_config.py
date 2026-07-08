"""Tests for cuttle_patterns.config."""

from pathlib import Path

import pytest

from cuttle_patterns.config import load_config


class TestLoadConfig:
    """Test the function load_config."""

    def test_load_config_success(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('data_dir: /data\nresults_dir: /results\n')

        # Act
        config = load_config(config_path)

        # Assert
        assert config.data_dir == Path('/data')
        assert config.results_dir == Path('/results')

    def test_load_config_expands_user(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('data_dir: ~/data\nresults_dir: ~/results\n')

        # Act
        config = load_config(config_path)

        # Assert
        assert config.data_dir == Path.home() / 'data'
        assert config.results_dir == Path.home() / 'results'

    def test_load_config_missing_file(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'does_not_exist.yaml'

        # Act & Assert
        with pytest.raises(FileNotFoundError, match='no config file found'):
            load_config(config_path)

    def test_load_config_missing_required_key(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config_path.write_text('data_dir: /data\n')

        # Act & Assert
        with pytest.raises(ValueError, match='missing required keys'):
            load_config(config_path)
