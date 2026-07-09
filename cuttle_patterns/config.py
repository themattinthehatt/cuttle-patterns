"""Per-machine configuration for data and results locations.

Every machine that runs this code is expected to have a local config file at
`~/.cuttle-patterns/config.yaml` pointing at where raw/derived data and results live on
that machine (mount points can differ machine to machine).
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path.home() / '.cuttle-patterns' / 'config.yaml'

REQUIRED_KEYS = ('data_dir', 'results_dir')


@dataclass
class Config:
    """Machine-local paths used throughout the project.

    Attributes:
        data_dir: root directory for raw and derived video/frame data.
        results_dir: root directory for manifests, embeddings, checkpoints, and figures.
    """

    data_dir: Path
    results_dir: Path


def load_config(config_path: Path | None = None) -> Config:
    """Load machine-local configuration from a yaml file.

    Args:
        config_path: path to the config file. Defaults to `DEFAULT_CONFIG_PATH`.

    Returns:
        parsed configuration.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the config file is missing required keys.
    """
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f'no config file found at {path}; create one with contents like:\n\n'
            f'data_dir: /path/to/data\n'
            f'results_dir: /path/to/results\n'
        )

    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    missing_keys = [key for key in REQUIRED_KEYS if key not in raw]
    if missing_keys:
        raise ValueError(f'config file {path} is missing required keys: {missing_keys}')

    return Config(
        data_dir=Path(raw['data_dir']).expanduser(),
        results_dir=Path(raw['results_dir']).expanduser(),
    )


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Write configuration to a yaml file, creating parent directories as needed.

    Args:
        config: configuration to write.
        config_path: path to write to. Defaults to `DEFAULT_CONFIG_PATH`.
    """
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = {
        'data_dir': str(config.data_dir),
        'results_dir': str(config.results_dir),
    }
    with path.open('w') as f:
        yaml.safe_dump(raw, f)
