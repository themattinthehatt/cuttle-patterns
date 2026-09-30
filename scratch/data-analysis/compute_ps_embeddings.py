"""Portilla-Simoncelli texture statistics as a per-frame embedding.

Computes plenoptic's `PortillaSimoncelli` statistics on grayscale frames (resized to
`--height` x `--width`, which must be divisible by 2 `--n-scales` times and leave the
coarsest scale at least `--spatial-corr-width` pixels), plus a small color block (Lab
L/a/b mean and std over the full-resolution frame), since PS is grayscale-only.

The raw statistics mix variances, correlations, and higher moments in very different
units, and groups of very different sizes (e.g. 6 pixel statistics vs ~150
autocorrelation entries), so each variant's embedding is:

1. log of the variance-type statistics (pixel variance, `magnitude_std`,
   `std_reconstructed`, `var_highpass_residual`);
2. z-scored per statistic across all frames;
3. scaled by `1 / sqrt(group size)` so every statistic group contributes equal total
   variance.

Missing values are imputed with the column mean (0 after z-scoring): a handful of
frames are entirely black (every correlation is 0/0), so their whole row is treated as
missing, and a few textured frames get NaN low-pass autocorrelations.

Each variant is written in the `cuttle embed` output layout (`config.yaml` +
`image_predictions/beast_frames/latents/{embeddings.npy,manifest.parquet}`), so
`cuttle reduce`/`cuttle cluster`/the eval harness read it with no PS-specific code.
Variants (model names use `ps_s{scales}o{orientations}w{corr width}_{width}`):

- `{base}_lab`: PS groups + Lab color group
- `{base}`: PS groups only (grayscale)
- `{base}_nopix`: PS groups minus `pixel_statistics` (no mean/contrast/black level)

Raw (untransformed) statistics and their group labels are cached under `.cache/`.

Example:
    python scratch/data-analysis/compute_ps_embeddings.py --n-scales 3 --height 96 --width 192
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plenoptic as po
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE_DIR, PREDICTIONS_NAME, RESULTS_DIR, get_model_dir, setup_logging  # noqa: E402
from cuttle_patterns import paths  # noqa: E402
from cuttle_patterns.embed import MODEL_CLASS, _parse_frame_path, find_frame_paths  # noqa: E402

logger = logging.getLogger(__name__)

LAB_GROUP = 'lab_color'
PIXEL_GROUP = 'pixel_statistics'
# groups whose entries are all variances/standard deviations (log-transformed)
LOG_GROUPS = ('magnitude_std', 'std_reconstructed', 'var_highpass_residual')
# `pixel_statistics` order is (mean, var, skew, kurtosis, min, max); only var is logged
PIXEL_VAR_OFFSET = 1
LOG_EPS = 1e-12


def load_frame(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load one frame as a resized grayscale image plus its full-resolution Lab stats.

    Args:
        path: frame PNG path.
        shape: (height, width) to resize the grayscale image to.

    Returns:
        (gray, lab_stats): float64 grayscale in [0, 1] of shape `shape`; float64 array
        of Lab (L, a, b) means followed by (L, a, b) standard deviations.
    """
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, shape[::-1], interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_stats = np.concatenate([lab.mean(axis=0), lab.std(axis=0)]).astype(np.float64)
    return gray.astype(np.float64) / 255, lab_stats


def get_stat_groups(model: po.models.PortillaSimoncelli, n_stats: int) -> np.ndarray:
    """Group name of each entry of the PS representation vector.

    Feeds `arange(n_stats)` through `convert_to_dict`, so each dict entry holds the
    vector indices it's built from (redundant entries come back as NaN and are skipped).

    Returns:
        array of `n_stats` group names.
    """
    rep = model.convert_to_dict(torch.arange(n_stats, dtype=torch.float64)[None, None])
    groups = np.empty(n_stats, dtype=object)
    for key, value in rep.items():
        idx = value.flatten().cpu().numpy()
        groups[idx[~np.isnan(idx)].astype(int)] = key
    if any(g is None for g in groups):
        raise ValueError('some PS statistics were not assigned to a group')
    return groups


def compute_raw_stats(
    frame_paths: list[Path],
    shape: tuple[int, int],
    n_scales: int,
    n_orientations: int,
    spatial_corr_width: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw PS statistics and Lab color stats for every frame.

    Returns:
        (stats, groups): `stats` is (n_frames, n_ps + 6) float64, PS entries first then
        the Lab block; `groups` names each column's group.
    """
    model = po.models.PortillaSimoncelli(
        shape,
        n_scales=n_scales,
        n_orientations=n_orientations,
        spatial_corr_width=spatial_corr_width,
    ).to(device)

    batches = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for start in tqdm(range(0, len(frame_paths), batch_size), desc='PS stats'):
            loaded = list(pool.map(
                lambda p: load_frame(p, shape), frame_paths[start:start + batch_size],
            ))
            gray = torch.from_numpy(np.stack([g for g, _ in loaded]))[:, None].to(device)
            with torch.no_grad():
                ps = model(gray)[:, 0].cpu().numpy()
            batches.append(np.concatenate([ps, np.stack([s for _, s in loaded])], axis=1))
    stats = np.concatenate(batches)
    n_ps = stats.shape[1] - 6
    groups = np.concatenate([get_stat_groups(model.cpu(), n_ps), [LAB_GROUP] * 6])
    return stats, groups


def transform_stats(
    stats: np.ndarray,
    groups: np.ndarray,
    is_blank: np.ndarray,
) -> np.ndarray:
    """Log variance-type stats, z-score each column, impute, and equalize group variance.

    Args:
        stats: raw statistics, shape (n_frames, n_stats).
        groups: group name per column.
        is_blank: bool per frame; blank frames' rows are imputed entirely.

    Returns:
        float32 embedding, shape (n_frames, n_kept); constant columns are dropped.
    """
    X = stats.copy()
    X[~np.isfinite(X)] = np.nan
    X[is_blank] = np.nan
    logger.info(
        f'imputing {np.isnan(X).any(axis=1).sum()} frames with missing statistics '
        f'({is_blank.sum()} blank)'
    )
    is_log = np.isin(groups, LOG_GROUPS)
    idx_pixel = np.flatnonzero(groups == PIXEL_GROUP)
    if len(idx_pixel):
        is_log[idx_pixel[PIXEL_VAR_OFFSET]] = True
    X[:, is_log] = np.log(np.maximum(X[:, is_log], LOG_EPS))

    std = np.nanstd(X, axis=0)
    keep = std > 0
    if (~keep).any():
        logger.info(f'dropping {(~keep).sum()} constant statistics')
    X = (X[:, keep] - np.nanmean(X[:, keep], axis=0)) / std[keep]
    X = np.nan_to_num(X, nan=0.0)
    groups = groups[keep]
    for group in np.unique(groups):
        X[:, groups == group] /= np.sqrt((groups == group).sum())
    return X.astype(np.float32)


def write_variant(
    name: str,
    X: np.ndarray,
    meta: pd.DataFrame,
    model_params: dict,
) -> None:
    """Write one variant in the `cuttle embed` model-directory layout."""
    model_dir = get_model_dir(name)
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / 'config.yaml').open('w') as f:
        yaml.safe_dump(
            {'model': {'model_class': MODEL_CLASS, 'model_params': {
                **model_params, 'embed_dim': int(X.shape[1]),
            }}},
            f,
        )
    latents_dir = model_dir / 'image_predictions' / PREDICTIONS_NAME / 'latents'
    latents_dir.mkdir(parents=True, exist_ok=True)
    np.save(latents_dir / 'embeddings.npy', X)
    meta.to_parquet(latents_dir / 'manifest.parquet', index=False)
    logger.info(f'wrote {name}: {X.shape}')


def main() -> None:
    """Compute raw PS stats (cached), then write each embedding variant."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--n-scales', type=int, default=3)
    parser.add_argument('--n-orientations', type=int, default=4)
    parser.add_argument('--spatial-corr-width', type=int, default=7)
    parser.add_argument('--height', type=int, default=96)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    setup_logging()

    base = f'ps_s{args.n_scales}o{args.n_orientations}w{args.spatial_corr_width}_{args.width}'
    shape = (args.height, args.width)
    frame_paths = find_frame_paths(RESULTS_DIR / paths.BEAST_FRAMES_RELPATH)
    meta = pd.DataFrame([_parse_frame_path(p) for p in frame_paths])

    cache_path = CACHE_DIR / f'{base}_raw_stats.npz'
    if cache_path.is_file():
        logger.info(f'loading cached raw stats from {cache_path}')
        cached = np.load(cache_path, allow_pickle=True)
        stats, groups = cached['stats'], cached['groups']
    else:
        stats, groups = compute_raw_stats(
            frame_paths, shape, args.n_scales, args.n_orientations,
            args.spatial_corr_width, args.batch_size, args.device,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, stats=stats, groups=groups)
    if len(stats) != len(meta):
        raise ValueError(f'cached stats have {len(stats)} rows, expected {len(meta)}')
    # zero pixel variance = a constant (in practice all-black) frame
    is_blank = stats[:, np.flatnonzero(groups == PIXEL_GROUP)[PIXEL_VAR_OFFSET]] == 0
    logger.info('group sizes: ' + ', '.join(
        f'{g}={n}' for g, n in zip(*np.unique(groups, return_counts=True))
    ))

    model_params = {
        'backbone': 'portilla_simoncelli',
        'n_scales': args.n_scales,
        'n_orientations': args.n_orientations,
        'spatial_corr_width': args.spatial_corr_width,
        'resolution': [args.height, args.width],
        'plenoptic_version': po.__version__,
        'extraction_timestamp': datetime.now(UTC).isoformat(),
    }
    variants = {
        f'{base}_lab': groups != '',
        base: groups != LAB_GROUP,
        f'{base}_nopix': ~np.isin(groups, [LAB_GROUP, PIXEL_GROUP]),
    }
    for name, cols in variants.items():
        X = transform_stats(stats[:, cols], groups[cols], is_blank)
        write_variant(name, X, meta, {
            **model_params, 'groups': sorted(set(groups[cols].tolist())),
        })


if __name__ == '__main__':
    main()
