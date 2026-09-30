"""Background-latent swap composites for an MSPS-VAE model.

For a source video, picks a few random frames from each k-means cluster it participates in.
For each source frame, keeps its unsupervised latent `z_u` fixed and swaps in, for every
video in the dataset ("target"), the target frame whose background latent `z_b` is nearest
(euclidean) to the source frame's `z_b`, then decodes `concat(z_u_src, z_b_target)`. When
source == target this is just the model's reconstruction of the source frame.

The resulting per-video synthesized frames are tiled onto a grid that approximates the
layout of each video's centroid in the 2D background UMAP (optimal assignment of centroids
to grid cells), so the layout is identical across every composite. The source tile gets a
green border; each tile gets a small `D{day}-T{tank}-{individual}` label in its upper right.

Saved latents (`image_predictions/beast_frames/latents/`) are reused rather than
re-encoding frames; `--verify` re-encodes a handful of frames once to confirm the saved
latents match this checkpoint + preprocessing.

Example:
    python scratch/data-analysis/background_swap_composites.py \
        --video Day1_Tank2_Cuttle1_Resident_Crop --limit 1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    RESULTS_DIR,
    get_analysis_dir,
    get_model_dir,
    load_model_latents,
    setup_logging,
)
from cuttle_patterns import paths  # noqa: E402
from cuttle_patterns.latents import split_latent_spaces  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = 'iter-1.1_msps-vae-mask_d16'
CLUSTERS_NAME = 'kmeans_k16'
DEFAULT_UMAP_NAME = 'umap_nn50_md0.1_euclidean_background'

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_IMAGE_SIZE = 224

TILE_WIDTH = 200
TILE_HEIGHT = 100
TILE_GAP = 4
SOURCE_BORDER_COLOR = (0, 255, 0)
SOURCE_BORDER_WIDTH = 4
SAME_INDIVIDUAL_COLOR = (0, 255, 0)
SAME_SESSION_COLOR = (255, 105, 180)
LABEL_OUTLINE_WIDTH = 2
LABEL_FONT_PATH = Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')
LABEL_FONT_SIZE = 10


def load_frame_table(
    model_dir: Path,
    umap_name: str,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Load saved latents joined with cluster labels, background UMAP coords, and identity.

    Args:
        model_dir: `results_dir/beast_models/{model_name}`.
        umap_name: background-UMAP parquet stem under the model's `reduce/` directory.

    Returns:
        (meta, X, num_latents_unsupervised): `meta` has one row per frame with columns
        `video_name`, `day`, `tank`, `role`, `frame_number`, `individual`, `cluster`,
        `umap_x`, `umap_y`; `X` is the row-aligned `concat(z_u, z_b)` latent array.
    """
    X, meta = load_model_latents(model_dir)
    num_latents_unsupervised = split_latent_spaces(X, model_dir)['unsupervised'].shape[1]

    keys = ['video_name', 'frame_number']
    clusters = pd.read_parquet(model_dir / paths.CLUSTERS_RELPATH / f'{CLUSTERS_NAME}.parquet')
    umap = pd.read_parquet(model_dir / paths.REDUCE_RELPATH / f'{umap_name}.parquet')
    meta = (
        meta
        .merge(clusters[keys + ['cluster']], on=keys, how='left', validate='one_to_one')
        .merge(umap[keys + ['umap_x', 'umap_y']], on=keys, how='left', validate='one_to_one')
    )
    if meta[['cluster', 'umap_x']].isna().any().any():
        raise ValueError('some latents have no matching cluster label or UMAP coordinate')
    return meta, X, num_latents_unsupervised


def compute_grid_layout(
    centroids: np.ndarray,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Assign 2D points to cells of a grid of `TILE_WIDTH` x `TILE_HEIGHT` tiles.

    Unless `shape` is given, the grid shape is the smallest (by cell count) whose pixel
    aspect ratio best matches the points' bounding-box aspect ratio; points are then
    assigned to cells by minimizing total squared displacement (Hungarian algorithm).

    Args:
        centroids: (n, 2) array of (x, y) points, y pointing up (UMAP convention).
        shape: optional explicit `(n_rows, n_cols)`; must have at least n cells.

    Returns:
        (cells, n_rows, n_cols): `cells` is an (n, 2) int array of (row, col) per point,
        with row 0 at the top.

    Raises:
        ValueError: if `shape` has fewer cells than points.
    """
    n = len(centroids)
    lo = centroids.min(axis=0)
    span = centroids.max(axis=0) - lo
    aspect_target = span[0] / span[1]

    # candidate shapes with at most a few spare cells; pick best aspect match
    best = None
    for n_cols in range(1, n + 1):
        n_rows = int(np.ceil(n / n_cols))
        if n_rows * n_cols - n > max(n_cols, n_rows):
            continue
        aspect = (n_cols * TILE_WIDTH) / (n_rows * TILE_HEIGHT)
        score = abs(np.log(aspect / aspect_target))
        if best is None or score < best[0]:
            best = (score, n_rows, n_cols)
    _, n_rows, n_cols = best
    if shape is not None:
        n_rows, n_cols = shape
        if n_rows * n_cols < n:
            raise ValueError(f'grid {n_rows}x{n_cols} has fewer than {n} cells')

    # normalize points into grid coordinates; flip y so UMAP "up" is image "up"
    unit = (centroids - lo) / span
    col_pos = unit[:, 0] * (n_cols - 1)
    row_pos = (1 - unit[:, 1]) * (n_rows - 1)

    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing='ij')
    cell_rc = np.stack([rows.ravel(), cols.ravel()], axis=1)
    cost = (
        (row_pos[:, None] - cell_rc[None, :, 0]) ** 2
        + (col_pos[:, None] - cell_rc[None, :, 1]) ** 2
    )
    idx_point, idx_cell = linear_sum_assignment(cost)
    cells = np.empty((n, 2), dtype=int)
    cells[idx_point] = cell_rc[idx_cell]
    return cells, n_rows, n_cols


def load_decoder(model_dir: Path, device: str) -> torch.nn.Module:
    """Load the model's best checkpoint via beast's API, in eval mode on `device`."""
    from beast.api.model import Model

    model = Model.from_dir(model_dir).model
    return model.to(device).eval()


def to_display_image(tensor: torch.Tensor) -> Image.Image:
    """Un-normalize a (3, 224, 224) model-space tensor and resize to the frame aspect."""
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    arr = ((tensor * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode='RGB').resize(
        (TILE_WIDTH, TILE_HEIGHT), resample=Image.Resampling.LANCZOS,
    )


@torch.no_grad()
def decode(model: torch.nn.Module, latents: np.ndarray) -> list[Image.Image]:
    """Decode `concat(z_u, z_b)` latents into display-sized images."""
    z = torch.from_numpy(latents).float().to(next(model.parameters()).device)
    xhat = model.decoder(model.latents_to_decoder(z))
    return [to_display_image(x) for x in xhat]


@torch.no_grad()
def verify_saved_latents(
    model: torch.nn.Module,
    meta: pd.DataFrame,
    X: np.ndarray,
    n_frames: int = 8,
    seed: int = 0,
) -> None:
    """Re-encode a few frames and compare to saved latents; logs max absolute difference."""
    frames_dir = RESULTS_DIR / paths.BEAST_FRAMES_RELPATH
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    idx = np.random.default_rng(seed).choice(len(meta), size=n_frames, replace=False)
    images = []
    for i in idx:
        row = meta.iloc[i]
        path = frames_dir / row['video_name'] / f'img{row["frame_number"]:08d}.png'
        images.append(preprocess(Image.open(path).convert('RGB')))
    x = torch.stack(images).to(next(model.parameters()).device)
    _, z_u, z_b = model(x)
    z = torch.cat([z_u, z_b], dim=1).cpu().numpy()
    diff = np.abs(z - X[idx]).max()
    scale = np.abs(X[idx]).max()
    logger.info(f'verify: max |encoded - saved| = {diff:.2e} (max |saved| = {scale:.2f})')


def draw_label(
    tile: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    inset: int = 0,
    outline: tuple[int, int, int] | None = None,
) -> None:
    """Draw `text` in white on a small black box in the tile's upper-right corner.

    Args:
        tile: image to draw on, in place.
        text: label text.
        font: label font.
        inset: offset of the box from the top and right edges, e.g. to clear a border.
        outline: optional RGB color for a `LABEL_OUTLINE_WIDTH` outline around the box.
    """
    draw = ImageDraw.Draw(tile)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    # pad includes room for the outline in every box, so all labels are the same size
    pad = 2 + LABEL_OUTLINE_WIDTH
    box_w = right - left + 2 * pad
    box_h = bottom - top + 2 * pad
    x0 = tile.width - inset - box_w
    y0 = inset
    draw.rectangle(
        [x0, y0, x0 + box_w - 1, y0 + box_h - 1],
        fill=(0, 0, 0),
        outline=outline,
        width=LABEL_OUTLINE_WIDTH,
    )
    draw.text((x0 + pad - left, y0 + pad - top), text, fill=(255, 255, 255), font=font)


def build_composite(
    tiles: dict[str, Image.Image],
    labels: dict[str, str],
    label_outlines: dict[str, tuple[int, int, int]],
    cells: dict[str, tuple[int, int]],
    n_rows: int,
    n_cols: int,
    video_source: str,
    font: ImageFont.FreeTypeFont,
) -> Image.Image:
    """Paste labeled tiles onto a black canvas at their grid cells.

    Args:
        tiles: synthesized image per video name.
        labels: label text per video name.
        label_outlines: label-box outline color per video name; videos absent get none.
        cells: (row, col) grid cell per video name.
        n_rows: number of grid rows.
        n_cols: number of grid columns.
        video_source: source video name, whose tile gets a thick green border.
        font: label font.

    Returns:
        the composite image.
    """
    width = n_cols * TILE_WIDTH + (n_cols + 1) * TILE_GAP
    height = n_rows * TILE_HEIGHT + (n_rows + 1) * TILE_GAP
    canvas = Image.new('RGB', (width, height), (0, 0, 0))
    for video_name, tile in tiles.items():
        tile = tile.copy()
        is_source = video_name == video_source
        if is_source:
            ImageDraw.Draw(tile).rectangle(
                [0, 0, tile.width - 1, tile.height - 1],
                outline=SOURCE_BORDER_COLOR,
                width=SOURCE_BORDER_WIDTH,
            )
        draw_label(
            tile,
            labels[video_name],
            font,
            inset=SOURCE_BORDER_WIDTH if is_source else 0,
            outline=label_outlines.get(video_name),
        )
        row, col = cells[video_name]
        x = TILE_GAP + col * (TILE_WIDTH + TILE_GAP)
        y = TILE_GAP + row * (TILE_HEIGHT + TILE_GAP)
        canvas.paste(tile, (x, y))
    return canvas


def main() -> None:
    """Parse args and write composites for one source video."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        '--umap',
        default=DEFAULT_UMAP_NAME,
        help='background-UMAP parquet stem under the model reduce/ dir (euclidean)',
    )
    parser.add_argument('--video', default='Day1_Tank2_Cuttle1_Resident_Crop')
    parser.add_argument('--frames-per-cluster', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None, help='max composites to write')
    parser.add_argument('--grid-rows', type=int, default=8, help='0 for auto shape')
    parser.add_argument('--grid-cols', type=int, default=5, help='0 for auto shape')
    parser.add_argument('--verify', action='store_true', help='check saved vs encoded latents')
    args = parser.parse_args()
    setup_logging()

    model_dir = get_model_dir(args.model)
    out_dir = get_analysis_dir(args.model) / 'image_composites' / args.video
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    meta, X, n_u = load_frame_table(model_dir, args.umap)
    model = load_decoder(model_dir, device)
    if args.verify:
        verify_saved_latents(model, meta, X)

    # fixed per-video layout from background-UMAP centroids
    videos = meta.groupby('video_name')
    centroids = videos[['umap_x', 'umap_y']].mean()
    cell_arr, n_rows, n_cols = compute_grid_layout(
        centroids.to_numpy(),
        shape=(args.grid_rows, args.grid_cols) if args.grid_rows and args.grid_cols else None,
    )
    cells = {v: tuple(rc) for v, rc in zip(centroids.index, cell_arr)}
    first = videos.first()
    labels = {
        v: f'D{r.day}-T{r.tank}-{r.individual}' for v, r in first.iterrows()
    }
    # label outlines: same individual (green) takes precedence over same day/tank (pink)
    src = first.loc[args.video]
    label_outlines = {}
    for v, r in first.iterrows():
        if v == args.video:
            continue
        if r.individual == src.individual:
            label_outlines[v] = SAME_INDIVIDUAL_COLOR
        elif (r.day, r.tank) == (src.day, src.tank):
            label_outlines[v] = SAME_SESSION_COLOR
    idx_by_video = {v: np.asarray(idx) for v, idx in videos.indices.items()}
    z_b_all = X[:, n_u:]
    logger.info(f'grid: {n_rows} rows x {n_cols} cols for {len(cells)} videos')

    # source frames: `frames_per_cluster` random frames per cluster the source video hits
    rng = np.random.default_rng(args.seed)
    source = meta[meta['video_name'] == args.video]
    if source.empty:
        raise ValueError(f'unknown video: {args.video}')
    picks = []
    for cluster, group in source.groupby('cluster'):
        n = min(args.frames_per_cluster, len(group))
        picks.extend((cluster, i) for i in sorted(rng.choice(group.index, size=n, replace=False)))
    if args.limit is not None:
        picks = picks[:args.limit]

    font = ImageFont.truetype(str(LABEL_FONT_PATH), LABEL_FONT_SIZE)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cluster, idx_src in picks:
        z_src = X[idx_src]
        names = list(cells)
        idx_targets = []
        for v in names:
            idx_v = idx_by_video[v]
            dist = np.linalg.norm(z_b_all[idx_v] - z_src[n_u:], axis=1)
            idx_targets.append(idx_v[np.argmin(dist)])
        latents = np.concatenate(
            [np.repeat(z_src[None, :n_u], len(names), axis=0), X[idx_targets, n_u:]],
            axis=1,
        )
        tiles = dict(zip(names, decode(model, latents)))
        composite = build_composite(
            tiles, labels, label_outlines, cells, n_rows, n_cols, args.video, font,
        )
        frame_number = meta.at[idx_src, 'frame_number']
        out_path = out_dir / f'composite_c{cluster:02d}_frame_{frame_number}.png'
        composite.save(out_path)
        logger.info(f'wrote {out_path}')


if __name__ == '__main__':
    main()
