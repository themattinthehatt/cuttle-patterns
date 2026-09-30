"""Quantify which low-level image statistics are video/session-specific.

Two complementary analyses:

1. **Composites** (output of `background_swap_composites.py`): crops every synthesized tile
   out of a random sample of composites and computes low-level statistics per tile. Since
   every tile in a composite shares the source's `z_u`, the fraction of variance explained
   by the *target video* (eta squared) measures how consistently the background latent
   `z_b` controls each statistic, across different source frames.
2. **Raw frames** (`beast_frames/`): computes the same statistics on a sample of real
   frames, and reports eta squared for video, session (day/tank), and individual, both
   pooled and *within* supervised-classifier pattern classes, so that videos which merely
   show more of some pattern are not mistaken for lighting/camera differences.

Example:
    python scratch/data-analysis/analyze_composite_lowlevel.py --n-composites 100
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import color

sys.path.insert(0, str(Path(__file__).parent))
from background_swap_composites import (  # noqa: E402
    RESULTS_DIR,
    TILE_GAP,
    TILE_HEIGHT,
    TILE_WIDTH,
    compute_grid_layout,
    load_frame_table,
)
from common import ANALYSIS_RELPATH, setup_logging  # noqa: E402
from cuttle_patterns import paths  # noqa: E402

logger = logging.getLogger(__name__)

CLASSIFIER_NAME = 'iter-1.1_classifier_d512'
# label boxes sit in the upper-right corner; exclude that region from statistics
LABEL_EXCLUDE_WIDTH = 70
LABEL_EXCLUDE_HEIGHT = 18


def image_stats(img: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    """Compute low-level statistics of an RGB image.

    Args:
        img: (H, W, 3) float RGB array in [0, 1].
        valid: optional (H, W) bool mask of pixels to include.

    Returns:
        dict of statistic name to value.
    """
    if valid is None:
        valid = np.ones(img.shape[:2], dtype=bool)
    lab = color.rgb2lab(img)
    lum = lab[..., 0]
    hsv = color.rgb2hsv(img)
    h, w = lum.shape

    # sharpness: variance of the Laplacian of luminance; fine texture: residual after blur
    lap = ndimage.laplace(lum)
    fine = lum - ndimage.gaussian_filter(lum, sigma=2)
    coarse = ndimage.gaussian_filter(lum, sigma=2) - ndimage.gaussian_filter(lum, sigma=8)

    # edge leakage: bottom strip / left strip luminance relative to the central region
    center = np.zeros_like(valid)
    center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    bottom = np.zeros_like(valid)
    bottom[int(0.85 * h):, :] = True
    top = np.zeros_like(valid)
    top[:int(0.15 * h), :] = True
    lum_center = lum[center & valid].mean()

    v = valid
    return {
        'R': img[..., 0][v].mean(),
        'G': img[..., 1][v].mean(),
        'B': img[..., 2][v].mean(),
        'L_mean': lum[v].mean(),
        'L_std': lum[v].std(),
        'L_p05': np.percentile(lum[v], 5),
        'L_p95': np.percentile(lum[v], 95),
        'a_mean': lab[..., 1][v].mean(),
        'b_mean': lab[..., 2][v].mean(),
        'sat_mean': hsv[..., 1][v].mean(),
        'log_lap_var': np.log(lap[v].var() + 1e-6),
        'fine_std': fine[v].std(),
        'coarse_std': coarse[v].std(),
        'fine_to_coarse': fine[v].std() / (coarse[v].std() + 1e-6),
        'bottom_minus_center': lum[bottom & v].mean() - lum_center,
        'top_minus_center': lum[top & v].mean() - lum_center,
    }


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Fraction of variance in `values` explained by group means (one-way ANOVA eta^2)."""
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    means = values.groupby(groups).transform('mean')
    ss_between = ((means - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def within_class_eta_squared(df: pd.DataFrame, stat: str, group: str, cls: str) -> float:
    """Eta^2 of `group` pooled within `cls` strata: sum SS_between / sum SS_total."""
    ss_between = 0.0
    ss_total = 0.0
    for _, sub in df.groupby(cls):
        grand = sub[stat].mean()
        ss_total += ((sub[stat] - grand) ** 2).sum()
        means = sub[stat].groupby(sub[group]).transform('mean')
        ss_between += ((means - grand) ** 2).sum()
    return ss_between / ss_total


def analyze_composites(
    model_dir: Path,
    umap_name: str,
    n_composites: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Crop tiles from random composites and compute per-tile statistics.

    Returns:
        one row per tile with columns `composite`, `source_video`, `target_video`,
        `is_source`, plus one column per statistic.
    """
    meta, _, _ = load_frame_table(model_dir, umap_name)
    centroids = meta.groupby('video_name')[['umap_x', 'umap_y']].mean()
    cell_arr, _, _ = compute_grid_layout(centroids.to_numpy(), shape=(8, 5))
    cells = dict(zip(centroids.index, map(tuple, cell_arr)))

    all_paths = sorted((model_dir / ANALYSIS_RELPATH / 'image_composites').glob('*/*.png'))
    idx = rng.choice(len(all_paths), size=min(n_composites, len(all_paths)), replace=False)

    valid = np.ones((TILE_HEIGHT, TILE_WIDTH), dtype=bool)
    valid[:LABEL_EXCLUDE_HEIGHT, -LABEL_EXCLUDE_WIDTH:] = False
    # stay clear of the source's green border too
    valid[:6, :] = valid[-6:, :] = False
    valid[:, :6] = valid[:, -6:] = False

    rows = []
    for i in idx:
        path = all_paths[i]
        composite = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255
        source = path.parent.name
        for video_name, (row, col) in cells.items():
            x = TILE_GAP + col * (TILE_WIDTH + TILE_GAP)
            y = TILE_GAP + row * (TILE_HEIGHT + TILE_GAP)
            tile = composite[y:y + TILE_HEIGHT, x:x + TILE_WIDTH]
            stats = image_stats(tile, valid)
            stats.update({
                'composite': path.name,
                'source_video': source,
                'target_video': video_name,
                'is_source': video_name == source,
            })
            rows.append(stats)
    return pd.DataFrame(rows)


def analyze_raw_frames(
    meta: pd.DataFrame,
    n_per_video: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Compute statistics on a random sample of real frames per video.

    Returns:
        one row per frame with metadata, `predicted_pattern`, and statistic columns.
    """
    frames_dir = RESULTS_DIR / paths.BEAST_FRAMES_RELPATH
    classes = pd.read_parquet(
        RESULTS_DIR / paths.CLASSIFICATIONS_RELPATH / f'{CLASSIFIER_NAME}.parquet',
        columns=['video_name', 'frame_number', 'predicted_pattern'],
    )
    meta = meta.merge(classes, on=['video_name', 'frame_number'], how='left')

    rows = []
    for _, group in meta.groupby('video_name'):
        n = min(n_per_video, len(group))
        for _, r in group.loc[rng.choice(group.index, size=n, replace=False)].iterrows():
            path = frames_dir / r['video_name'] / f'img{r["frame_number"]:08d}.png'
            img = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64) / 255
            stats = image_stats(img)
            stats.update(r[[
                'video_name', 'frame_number', 'day', 'tank', 'role', 'individual',
                'predicted_pattern',
            ]].to_dict())
            rows.append(stats)
    df = pd.DataFrame(rows)
    df['session'] = 'D' + df['day'].astype(str) + '-T' + df['tank'].astype(str)
    return df


def main() -> None:
    """Run both analyses and print summary tables."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default='iter-1.1_msps-vae_d16')
    parser.add_argument('--umap', default='umap_nn50_md0.1_background')
    parser.add_argument('--n-composites', type=int, default=100)
    parser.add_argument('--n-raw-per-video', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', 30)

    rng = np.random.default_rng(args.seed)
    model_dir = RESULTS_DIR / paths.BEAST_MODELS_RELPATH / args.model
    out_dir = Path(__file__).parent / '.cache'

    tiles = analyze_composites(model_dir, args.umap, args.n_composites, rng)
    tiles.to_parquet(out_dir / f'{args.model}_composite_tile_stats.parquet')
    stat_names = [c for c in tiles.columns if c not in (
        'composite', 'source_video', 'target_video', 'is_source',
    )]
    summary = pd.DataFrame({
        'eta2_target_video': [eta_squared(tiles[s], tiles['target_video']) for s in stat_names],
        'eta2_composite': [eta_squared(tiles[s], tiles['composite']) for s in stat_names],
    }, index=stat_names)
    print(f'\n=== composites: {tiles["composite"].nunique()} composites, {len(tiles)} tiles ===')
    print(summary.round(3).sort_values('eta2_target_video', ascending=False))

    meta, _, _ = load_frame_table(model_dir, args.umap)
    raw = analyze_raw_frames(meta, args.n_raw_per_video, rng)
    raw.to_parquet(out_dir / 'raw_frame_stats.parquet')
    summary_raw = pd.DataFrame({
        'video': [eta_squared(raw[s], raw['video_name']) for s in stat_names],
        'session': [eta_squared(raw[s], raw['session']) for s in stat_names],
        'individual': [eta_squared(raw[s], raw['individual']) for s in stat_names],
        'pattern': [eta_squared(raw[s], raw['predicted_pattern']) for s in stat_names],
        'video|pattern': [
            within_class_eta_squared(raw, s, 'video_name', 'predicted_pattern')
            for s in stat_names
        ],
        'session|pattern': [
            within_class_eta_squared(raw, s, 'session', 'predicted_pattern')
            for s in stat_names
        ],
        'indiv|pattern': [
            within_class_eta_squared(raw, s, 'individual', 'predicted_pattern')
            for s in stat_names
        ],
    }, index=stat_names)
    print(f'\n=== raw frames: {len(raw)} frames, eta^2 by grouping ===')
    print(summary_raw.round(3).sort_values('video|pattern', ascending=False))

    print('\n=== raw frames: per-session means (sorted by b_mean) ===')
    cols = ['L_mean', 'L_std', 'a_mean', 'b_mean', 'sat_mean', 'log_lap_var', 'fine_std',
            'bottom_minus_center']
    print(raw.groupby('session')[cols].mean().round(2).sort_values('b_mean'))
    print('\n=== raw frames: pattern class counts per session ===')
    print(pd.crosstab(raw['session'], raw['predicted_pattern']))


if __name__ == '__main__':
    main()
