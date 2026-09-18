"""Shared fixtures for eval/tests."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def make_latents_dir() -> Callable[[Path, dict[str, dict[int, np.ndarray]]], None]:
    """Return a factory that writes a fake `cuttle predict --save-latents` tree.

    Returns:
        a callable `(model_dir, video_frames) -> None`, where `video_frames` maps
        `video_name -> {frame_number: vector}`, matching
        `cuttle_patterns.embeddings.load_latents`'s expected on-disk layout.
    """

    def _make_latents_dir(model_dir: Path, video_frames: dict[str, dict[int, np.ndarray]]) -> None:
        latents_dir = model_dir / 'image_predictions' / 'beast_frames' / 'latents'
        for video_name, frames in video_frames.items():
            video_dir = latents_dir / video_name
            video_dir.mkdir(parents=True, exist_ok=True)
            for frame_number, vector in frames.items():
                np.save(video_dir / f'img{frame_number}.npy', vector)

    return _make_latents_dir


@pytest.fixture
def write_model_config() -> Callable[..., None]:
    """Return a factory that writes a minimal model `config.yaml`.

    Returns:
        a callable `(model_dir, model_class, **model_params) -> None`, matching the
        subset of `beast train`'s own `config.yaml` that
        `cuttle_patterns.embeddings.split_latent_spaces` reads.
    """

    def _write_model_config(model_dir: Path, model_class: str, **model_params) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        config = {'model': {'model_class': model_class, 'model_params': model_params}}
        with (model_dir / 'config.yaml').open('w') as f:
            yaml.safe_dump(config, f)

    return _write_model_config
