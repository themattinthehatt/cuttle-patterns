"""Project per-frame BEAST embeddings to 2D via UMAP (Phase 5).

Reads the latents `cuttle_patterns.embeddings.load_latents` returns and runs UMAP over
them. `cuttle cluster` (Phase 6) reads the same latents directly rather than this
module's output, since clustering happens in the raw embedding space, not the 2D
projection.
"""

import numpy as np
import pandas as pd
import umap

DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_METRIC = 'euclidean'
DEFAULT_RANDOM_STATE = 42

UMAP_OUTPUT_COLUMNS = ['umap_x', 'umap_y', 'day', 'tank', 'role', 'frame_number', 'video_name']


def run_umap(
    X: np.ndarray,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    metric: str = DEFAULT_METRIC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Project embeddings to 2D with UMAP.

    Args:
        X: embeddings, shape (n_frames, latent_dim).
        n_neighbors: UMAP's `n_neighbors`.
        min_dist: UMAP's `min_dist`.
        metric: UMAP's `metric`.
        random_state: UMAP's `random_state`, for reproducibility.

    Returns:
        array of shape (n_frames, 2).
    """
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def hparams_to_str(n_neighbors: int, min_dist: float) -> str:
    """Build the `{hparams}` filename suffix for a given `n_neighbors`/`min_dist`.

    Args:
        n_neighbors: UMAP's `n_neighbors`.
        min_dist: UMAP's `min_dist`.

    Returns:
        string like `nn15_md0.1`.
    """
    return f'nn{n_neighbors}_md{min_dist}'


def build_umap_dataframe(meta: pd.DataFrame, umap_xy: np.ndarray) -> pd.DataFrame:
    """Attach UMAP coordinates to per-frame metadata.

    Args:
        meta: per-frame metadata from `cuttle_patterns.embeddings.load_latents`, with
            columns `video_name`, `day`, `tank`, `role`, `frame_number`.
        umap_xy: array of shape (len(meta), 2), row-aligned with meta.

    Returns:
        DataFrame with columns `umap_x`, `umap_y`, `day`, `tank`, `role`,
        `frame_number`, `video_name`.
    """
    out = meta.copy()
    out['umap_x'] = umap_xy[:, 0]
    out['umap_y'] = umap_xy[:, 1]
    return out[UMAP_OUTPUT_COLUMNS]
