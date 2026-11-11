"""Shared fixtures for cuttle_patterns.preprocessing.extract tests."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cuttle_patterns.preprocessing.align import CORNER_COLUMNS

DIM = 8


@pytest.fixture
def write_pose_csv() -> Callable[..., Path]:
    """Return a factory writing a multi-header tail/neck pose CSV.

    Returns:
        a callable taking (path, rows) where each row is a dict with keys neck_x, neck_y,
        neck_likelihood, tail_x, tail_y, tail_likelihood, and returning path.
    """

    def _write(path: Path, rows: list[dict]) -> Path:
        columns = pd.MultiIndex.from_tuples(
            [
                ('m', 'neck', 'x'), ('m', 'neck', 'y'), ('m', 'neck', 'likelihood'),
                ('m', 'tail', 'x'), ('m', 'tail', 'y'), ('m', 'tail', 'likelihood'),
            ],
            names=['scorer', 'bodyparts', 'coords'],
        )
        data = [
            [r['neck_x'], r['neck_y'], r['neck_likelihood'],
             r['tail_x'], r['tail_y'], r['tail_likelihood']]
            for r in rows
        ]
        pd.DataFrame(data, columns=columns).to_csv(path)
        return path

    return _write


@pytest.fixture
def write_rect_csv() -> Callable[..., Path]:
    """Return a factory writing a `cuttle inscribe`-style corner geometry CSV.

    Returns:
        a callable taking (path, n_frames, long_edge, short_edge) and returning path; the same
        axis-aligned rectangle is written for every frame.
    """

    def _write(path: Path, n_frames: int, long_edge: float = 50.0, short_edge: float = 10.0):
        corners = {
            'corner_tl_x': 0.0, 'corner_tl_y': 0.0,
            'corner_tr_x': long_edge, 'corner_tr_y': 0.0,
            'corner_br_x': long_edge, 'corner_br_y': short_edge,
            'corner_bl_x': 0.0, 'corner_bl_y': short_edge,
        }
        df = pd.DataFrame([corners] * n_frames, columns=CORNER_COLUMNS)
        df.insert(0, 'frame_idx', range(n_frames))
        df['is_interpolated'] = False
        df.to_csv(path, index=False)
        return path

    return _write


def pose_row(
    neck: tuple[float, float] = (0.0, 0.0),
    tail: tuple[float, float] = (100.0, 0.0),
    likelihood: float = 0.99,
) -> dict:
    """Build one pose row.

    Args:
        neck: neck (x, y).
        tail: tail (x, y).
        likelihood: likelihood of both keypoints.

    Returns:
        dict with neck_x, neck_y, neck_likelihood, tail_x, tail_y, tail_likelihood.
    """
    return {
        'neck_x': neck[0], 'neck_y': neck[1], 'neck_likelihood': likelihood,
        'tail_x': tail[0], 'tail_y': tail[1], 'tail_likelihood': likelihood,
    }


@pytest.fixture
def constant_frames() -> Callable[..., list[np.ndarray]]:
    """Return a factory for constant-colour frames.

    Returns:
        a callable taking (values, size) and returning one uint8 (size, size, 3) frame per
        value, filled with that value.
    """

    def _frames(values: list[int], size: int = 32) -> list[np.ndarray]:
        return [np.full((size, size, 3), v, dtype=np.uint8) for v in values]

    return _frames


def make_pipeline_dirs(tmp_path: Path, n_videos: int = 3, n_frames: int = 400) -> dict:
    """Write eligibility, classifier and embedding files for synthetic videos."""
    elig_dir, clf_dir, emb_dir = tmp_path / 'elig', tmp_path / 'clf', tmp_path / 'emb'
    for d in (elig_dir, clf_dir, emb_dir):
        d.mkdir()
    rng = np.random.default_rng(0)
    stems = [f'Day{2 + i}_Tank1_Cuttle1_Resident_Crop' for i in range(n_videos)]
    centers = np.eye(DIM)[:4] * 3
    for stem in stems:
        # class A (70%) uses blobs 0-1, class B (30%) blobs 2-3
        labels = rng.random(n_frames) < 0.3
        blob = rng.integers(0, 2, n_frames) + 2 * labels
        emb = centers[blob] + 0.3 * rng.standard_normal((n_frames, DIM))
        np.save(emb_dir / f'{stem}.npy', emb.astype(np.float16))
        pd.DataFrame({
            'frame_idx': np.arange(n_frames),
            'eligible': np.arange(n_frames) % 5 != 0,  # every 5th frame is ineligible
        }).to_parquet(elig_dir / f'{stem}.parquet')
        pd.DataFrame({
            'frame_idx': np.arange(n_frames),
            'predicted_pattern': np.where(labels, 'B', 'A'),
            'confidence': rng.random(n_frames).astype(np.float32),
        }).to_parquet(clf_dir / f'{stem}.parquet')
    return dict(stems=stems, elig_dir=elig_dir, clf_dir=clf_dir, emb_dir=emb_dir)
