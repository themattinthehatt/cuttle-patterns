"""Shared helpers for the exploratory scripts in `scratch/data-analysis/`.

Everything here is a thin layer over `cuttle_patterns` — latent loading (with an npz cache
for the slow per-frame layout), per-frame metadata (individual, session), and the
`analysis/` output tree — so each analysis script stays focused on its own question.

Exploratory outputs live under `beast_models/{model}/analysis/`, kept separate from the
canonical `reduce/` and `clusters/` outputs other tools read:

    beast_models/{model}/analysis/
    ├── clusters/{method}_{hparams}/      # per-clustering summary figures
    ├── image_composites/{video_name}/    # background_swap_composites.py
    └── subsets/{subset_name}/            # subset_analysis.py; mirrors a model dir
        ├── subset.json
        ├── reduce/
        ├── clusters/
        └── figures/
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cuttle_patterns import paths
from cuttle_patterns.latents import (
    load_latents,
    parse_video_name,
    select_cluster_latents,
    split_latent_spaces,
)
from cuttle_patterns.metadata import attach_individual_column

logger = logging.getLogger(__name__)

RESULTS_DIR = Path('/media/mattw/CUTTLE/results')
CACHE_DIR = Path(__file__).parent / '.cache'
ANALYSIS_RELPATH = Path('analysis')
SUBSETS_RELPATH = ANALYSIS_RELPATH / 'subsets'
PREDICTIONS_NAME = 'beast_frames'


def get_model_dir(model: str) -> Path:
    """Path to `results_dir/beast_models/{model}`."""
    return RESULTS_DIR / paths.BEAST_MODELS_RELPATH / model


def get_analysis_dir(model: str) -> Path:
    """Path to `results_dir/beast_models/{model}/analysis`."""
    return get_model_dir(model) / ANALYSIS_RELPATH


def add_frame_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """Add `day`, `tank`, `role`, `individual`, and `session` columns from `video_name`.

    Args:
        meta: per-frame table with a `video_name` column.

    Returns:
        a copy of `meta` with the added columns; `session` is `D{day}-T{tank}`.
    """
    parsed = pd.DataFrame([parse_video_name(v) for v in meta['video_name']], index=meta.index)
    out = meta.drop(columns=parsed.columns, errors='ignore').join(parsed)
    out = attach_individual_column(out)
    out['session'] = 'D' + out['day'].astype(str) + '-T' + out['tank'].astype(str)
    return out


def load_model_latents(model_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load a model's full latents plus per-frame metadata.

    The per-frame `cuttle predict` layout (one tiny file per frame) is slow to read, so it
    is cached as npz under `.cache/` after the first load; the combined `cuttle embed`
    layout is already a single file and is read directly.

    Args:
        model_dir: `results_dir/beast_models/{model_name}`.

    Returns:
        (X, meta): `X` is the full latent array (for MSPS models, `concat(z_u, z_b)`);
        `meta` is row-aligned with columns `video_name`, `frame_number`, `day`, `tank`,
        `role`, `individual`, `session`.
    """
    latents_dir = model_dir / 'image_predictions' / PREDICTIONS_NAME / 'latents'
    cache_path = CACHE_DIR / f'{model_dir.name}_latents.npz'
    if (latents_dir / 'embeddings.npy').is_file():
        X, meta = load_latents(latents_dir)
    elif cache_path.is_file():
        logger.info(f'loading cached latents from {cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        X = cached['X']
        meta = pd.DataFrame({
            'video_name': cached['video_name'],
            'frame_number': cached['frame_number'],
        })
    else:
        logger.info(f'loading per-frame latents from {latents_dir} (slow, cached after)')
        X, meta = load_latents(latents_dir)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            X=X,
            video_name=meta['video_name'].to_numpy(),
            frame_number=meta['frame_number'].to_numpy(),
        )
    meta = add_frame_metadata(meta[['video_name', 'frame_number']].reset_index(drop=True))
    return X, meta


def get_cluster_space(X: np.ndarray, model_dir: Path) -> np.ndarray:
    """The latent subspace `cuttle cluster` clusters on (`z_u` for MSPS models)."""
    return select_cluster_latents(split_latent_spaces(X, model_dir))


def effective_count(labels: pd.Series | np.ndarray) -> float:
    """`exp(entropy)` of a label distribution: the effective number of distinct labels."""
    p = pd.Series(labels).value_counts(normalize=True).to_numpy()
    return float(np.exp(-(p * np.log(p)).sum()))


def setup_logging() -> None:
    """Configure plain INFO-level logging to stdout for a script entry point."""
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
