"""Tests for cuttle_patterns.config."""

from pathlib import Path

import pytest
import yaml

from cuttle_patterns.config import Config, load_config, save_config


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


class TestSaveConfig:
    """Test the function save_config."""

    def test_save_config_success(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config = Config(data_dir=Path('/data'), results_dir=Path('/results'))

        # Act
        save_config(config, config_path)

        # Assert
        raw = yaml.safe_load(config_path.read_text())
        assert raw == {'data_dir': '/data', 'results_dir': '/results'}

    def test_save_config_creates_parent_dirs(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'nested' / 'dir' / 'config.yaml'
        config = Config(data_dir=Path('/data'), results_dir=Path('/results'))

        # Act
        save_config(config, config_path)

        # Assert
        assert config_path.exists()

    def test_save_config_round_trips_with_load_config(self, tmp_path: Path):
        # Arrange
        config_path = tmp_path / 'config.yaml'
        config = Config(data_dir=Path('/data'), results_dir=Path('/results'))

        # Act
        save_config(config, config_path)
        loaded = load_config(config_path)

        # Assert
        assert loaded == config
