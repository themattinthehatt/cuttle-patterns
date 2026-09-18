"""Plot candidate radial weight profiles for the MSPS-AE reconstruction-loss experiment.

Visualizes three candidate spatial weight maps for down-weighting reconstruction loss
near the edges/corners of a square crop (full weight at the center, tapering to zero
past the inscribed circle, so corners — farther from center than edge midpoints — are
fully ignored):

  - Gaussian: `exp(-(r/sigma)**2)`.
  - Super-Gaussian: `exp(-(r/sigma)**(2*n))`, a flatter-topped Gaussian.
  - Raised-cosine (Tukey-style): flat at 1.0 out to `r0`, cosine taper to 0 at `r=1`.

`r` is the distance from the crop center, normalized so `r=1` sits at each edge's
midpoint and corners sit at `r=sqrt(2)` — i.e. always past the taper's zero point,
regardless of the chosen parameters.

Writes two kinds of figure to `results_dir/beast_frames_qc/loss_mask/`:
  - `loss_weight_profiles.png`: each profile's 2D weight map plus all three radial
    profiles overlaid on a single axis, so the shapes can be compared abstractly.
  - `frame_overlays_{i:02d}.png`: a random sample of real exported frames — resized to
    square, the same way the autoencoder sees them — with each mask overlaid as a red
    tint (stronger where a mask discards more), one row per frame and one column per
    mask, so the abstract shapes above can be checked against what they'd actually do
    to real crops.

Usage:
    python scratch/plot_loss_weight_profiles.py
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from cuttle_patterns import paths
from cuttle_patterns.config import load_config
from cuttle_patterns.embeddings import FRAME_FILENAME_PATTERN

DEFAULT_SIDE = 128
DEFAULT_SIGMA_GAUSSIAN = 0.4
DEFAULT_SIGMA_SUPER_GAUSSIAN = 0.4
DEFAULT_N_SUPER_GAUSSIAN = 2
DEFAULT_R0_TUKEY = 0.5
DEFAULT_N_FRAMES = 20
DEFAULT_N_ROWS_PER_FIGURE = 4
DEFAULT_SEED = 0
OVERLAY_MAX_ALPHA = 0.6
OVERLAY_COLOR = np.array([1.0, 0.0, 0.0])

CORNER_RADIUS = np.sqrt(2)


def radial_grid(side: int) -> np.ndarray:
    """Build a normalized-radius grid for a square crop.

    Args:
        side: crop side length, in pixels.

    Returns:
        `(side, side)` array of distances from the crop center, normalized so `r=1`
        falls exactly at each edge's midpoint (corners fall at `r=sqrt(2)`).
    """
    y, x = np.mgrid[0:side, 0:side]
    cy, cx = (side - 1) / 2, (side - 1) / 2
    return np.hypot(x - cx, y - cy) / (side / 2)


def gaussian_profile(r: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian radial weight, full weight at the center, no true plateau.

    Args:
        r: normalized radius (see `radial_grid`).
        sigma: standard deviation of the Gaussian, in the same normalized units as `r`.

    Returns:
        weights in `[0, 1]`, same shape as `r`.
    """
    return np.exp(-(r**2) / (2 * sigma**2))


def super_gaussian_profile(r: np.ndarray, sigma: float, n: float) -> np.ndarray:
    """Super-Gaussian radial weight: a Gaussian with a flattened top.

    Args:
        r: normalized radius (see `radial_grid`).
        sigma: controls how far the near-flat core extends.
        n: controls how sharp the shoulder is; `n=1` is a plain Gaussian, larger `n`
            approaches a smoothed top-hat.

    Returns:
        weights in `[0, 1]`, same shape as `r`.
    """
    return np.exp(-((r / sigma) ** (2 * n)))


def tukey_radial_profile(r: np.ndarray, r0: float) -> np.ndarray:
    """Raised-cosine (Tukey-style) radial weight: a literal plateau, then a taper.

    Flat at 1.0 for `r <= r0`, a cosine taper down to exactly 0 (value and slope
    both zero) at `r = 1`, and exactly 0 beyond.

    Args:
        r: normalized radius (see `radial_grid`).
        r0: plateau radius — `r <= r0` gets full weight.

    Returns:
        weights in `[0, 1]`, same shape as `r`.
    """
    w = np.ones_like(r)
    taper = (r > r0) & (r <= 1)
    w[taper] = 0.5 * (1 + np.cos(np.pi * (r[taper] - r0) / (1 - r0)))
    w[r > 1] = 0.0
    return w


def build_mask_specs(
    side: int,
    sigma_gaussian: float,
    sigma_super_gaussian: float,
    n_super_gaussian: float,
    r0_tukey: float,
) -> list[tuple[str, str, np.ndarray]]:
    """Build the (name, params, 2D weight map) triples shared by both figure kinds.

    Args:
        side: crop side length (pixels).
        sigma_gaussian: `sigma` passed to `gaussian_profile`.
        sigma_super_gaussian: `sigma` passed to `super_gaussian_profile`.
        n_super_gaussian: `n` passed to `super_gaussian_profile`.
        r0_tukey: `r0` passed to `tukey_radial_profile`.

    Returns:
        one `(name, params_str, weight_map)` triple per candidate profile.
    """
    r_grid = radial_grid(side)
    return [
        ('Gaussian', f'sigma={sigma_gaussian}', gaussian_profile(r_grid, sigma_gaussian)),
        (
            'Super-Gaussian', f'sigma={sigma_super_gaussian}, n={n_super_gaussian}',
            super_gaussian_profile(r_grid, sigma_super_gaussian, n_super_gaussian),
        ),
        ('Raised-cosine', f'r0={r0_tukey}', tukey_radial_profile(r_grid, r0_tukey)),
    ]


def plot_profiles(mask_specs: list[tuple[str, str, np.ndarray]], side: int) -> plt.Figure:
    """Build the abstract comparison figure: 2D weight maps plus radial profiles.

    Args:
        mask_specs: output of `build_mask_specs`.
        side: crop side length (pixels) the weight maps were built at.

    Returns:
        the assembled figure.
    """
    r_line = np.linspace(0, CORNER_RADIUS, 500)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), height_ratios=[1, 0.7])

    for ax, (name, params, weight_map) in zip(axes[0], mask_specs, strict=True):
        im = ax.imshow(weight_map, vmin=0, vmax=1, cmap='viridis')
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(
            (side - 1) / 2 * (1 + np.cos(theta)), (side - 1) / 2 * (1 + np.sin(theta)),
            color='white', linestyle='--', linewidth=1,
        )
        ax.set_title(f'{name}\n({params})', fontsize=10)
        ax.axis('off')
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label='weight')

    ax = axes[1, 0]
    ax.remove()
    ax = fig.add_subplot(2, 1, 2)
    for name, params, weight_map in mask_specs:
        # re-derive each profile along a dense 1D ray from the same 2D weight map's
        # center, rather than recomputing from parameters, so the line always matches
        # what's actually plotted in the heatmaps above; bilinear (order=1) avoids a
        # staircase artifact from the grid's finite resolution
        side_actual = weight_map.shape[0]
        cy = cx = (side_actual - 1) / 2
        cols = cx + r_line * (side_actual / 2)
        rows = np.full(len(r_line), cy)
        values = map_coordinates(weight_map, [rows, cols], order=1, mode='nearest')
        ax.plot(r_line, values, label=f'{name} ({params})', linewidth=2)
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=1)
    ax.text(1.0, -0.09, 'edge\nmidpoint', ha='center', va='top', fontsize=8, color='gray')
    ax.axvline(CORNER_RADIUS, color='gray', linestyle=':', linewidth=1)
    ax.text(CORNER_RADIUS, -0.09, 'corner', ha='center', va='top', fontsize=8, color='gray')
    ax.set_xlabel('normalized radius r')
    ax.set_ylabel('weight')
    ax.set_xlim(0, CORNER_RADIUS * 1.05)
    ax.set_ylim(-0.22, 1.08)
    ax.legend(loc='upper right', fontsize=9)
    axes[1, 1].remove()
    axes[1, 2].remove()

    fig.suptitle('Candidate loss-weighting radial profiles', fontsize=13)
    return fig


def sample_random_frame_paths(
    beast_frames_dir: Path,
    n_frames: int,
    rng: np.random.Generator,
) -> list[Path]:
    """Randomly sample exported frame PNGs across every video.

    Args:
        beast_frames_dir: path to `results_dir/beast_frames/`.
        n_frames: number of frames to sample.
        rng: random number generator, for reproducibility.

    Returns:
        `n_frames` paths (or fewer, if fewer frames exist), sampled without replacement.

    Raises:
        FileNotFoundError: if no exported frame PNGs are found.
    """
    png_paths = sorted(beast_frames_dir.glob('*/img*.png'))
    if not png_paths:
        raise FileNotFoundError(f'no exported frame PNGs found under {beast_frames_dir}')

    n_frames = min(n_frames, len(png_paths))
    idxs = rng.choice(len(png_paths), size=n_frames, replace=False)
    return [png_paths[i] for i in sorted(idxs)]


def load_square_frame(frame_path: Path, side: int) -> np.ndarray:
    """Load a frame PNG and resize it to a square, the way the autoencoder sees it.

    Args:
        frame_path: path to a `beast_frames/{video_name}/img{frame_number:08d}.png` file.
        side: output side length (pixels).

    Returns:
        an `(side, side, 3)` RGB float array in `[0, 1]`.

    Raises:
        OSError: if the frame image cannot be read.
    """
    image = cv2.imread(str(frame_path))
    if image is None:
        raise OSError(f'could not read frame image: {frame_path}')
    resized = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def overlay_mask_on_frame(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the parts of a frame a mask discards, so the cutoff is visible in context.

    Args:
        frame_rgb: `(side, side, 3)` RGB float array in `[0, 1]`.
        mask: `(side, side)` weight map in `[0, 1]`, same shape as `frame_rgb`'s
            spatial dims — 1 means fully kept, 0 means fully discarded.

    Returns:
        an `(side, side, 3)` RGB float composite, `OVERLAY_COLOR` blended in
        proportional to `1 - mask` (up to `OVERLAY_MAX_ALPHA`).
    """
    alpha = (1 - mask)[..., None] * OVERLAY_MAX_ALPHA
    return frame_rgb * (1 - alpha) + OVERLAY_COLOR * alpha


def plot_frame_overlays(
    frame_paths: list[Path],
    mask_specs: list[tuple[str, str, np.ndarray]],
    side: int,
    n_rows_per_figure: int,
) -> list[plt.Figure]:
    """Build one figure per chunk of frames, each mask overlaid in its own column.

    Args:
        frame_paths: frame PNGs to visualize (see `sample_random_frame_paths`).
        mask_specs: output of `build_mask_specs`.
        side: side length (pixels) frames are resized to, matching the masks' shape.
        n_rows_per_figure: number of frames (rows) per figure.

    Returns:
        one figure per chunk of `n_rows_per_figure` frame paths.
    """
    figures = []
    for chunk_start in range(0, len(frame_paths), n_rows_per_figure):
        chunk = frame_paths[chunk_start:chunk_start + n_rows_per_figure]
        n_rows = len(chunk)
        fig, axes = plt.subplots(
            n_rows, len(mask_specs), figsize=(3.2 * len(mask_specs), 3.2 * n_rows), squeeze=False,
        )

        for row, frame_path in enumerate(chunk):
            frame_rgb = load_square_frame(frame_path, side)
            video_name = frame_path.parent.name
            frame_number = int(FRAME_FILENAME_PATTERN.match(frame_path.stem)['frame_number'])

            for col, (name, params, mask) in enumerate(mask_specs):
                ax = axes[row, col]
                ax.imshow(overlay_mask_on_frame(frame_rgb, mask))
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(f'{name} ({params})', fontsize=9)
                if col == 0:
                    ax.set_ylabel(f'{video_name}\nframe {frame_number}', fontsize=7)

        fig.suptitle(
            f'Loss-weighting masks overlaid on real frames '
            f'({chunk_start + 1}-{chunk_start + n_rows})',
            fontsize=12,
        )
        fig.tight_layout()
        figures.append(fig)

    return figures


def main() -> None:
    """Plot and save the weight-profile comparison and frame-overlay QC figures."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results-dir', type=Path, default=None)
    parser.add_argument('--side', type=int, default=DEFAULT_SIDE)
    parser.add_argument('--sigma-gaussian', type=float, default=DEFAULT_SIGMA_GAUSSIAN)
    parser.add_argument(
        '--sigma-super-gaussian', type=float, default=DEFAULT_SIGMA_SUPER_GAUSSIAN,
    )
    parser.add_argument('--n-super-gaussian', type=float, default=DEFAULT_N_SUPER_GAUSSIAN)
    parser.add_argument('--r0-tukey', type=float, default=DEFAULT_R0_TUKEY)
    parser.add_argument('--n-frames', type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument('--n-rows-per-figure', type=int, default=DEFAULT_N_ROWS_PER_FIGURE)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir is not None else load_config().results_dir
    output_dir = results_dir / paths.LOSS_MASK_QC_RELPATH
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_specs = build_mask_specs(
        args.side,
        args.sigma_gaussian,
        args.sigma_super_gaussian,
        args.n_super_gaussian,
        args.r0_tukey,
    )

    profiles_path = output_dir / 'loss_weight_profiles.png'
    if profiles_path.exists():
        print(f'warning: overwriting existing {profiles_path}')
    plot_profiles(mask_specs, args.side).savefig(profiles_path, dpi=150, bbox_inches='tight')
    print(f'wrote {profiles_path}')

    rng = np.random.default_rng(args.seed)
    frame_paths = sample_random_frame_paths(
        results_dir / paths.BEAST_FRAMES_RELPATH, args.n_frames, rng,
    )
    overlay_figures = plot_frame_overlays(
        frame_paths, mask_specs, args.side, args.n_rows_per_figure,
    )

    n_digits = max(2, len(str(len(overlay_figures))))
    for i, fig in enumerate(overlay_figures, start=1):
        overlay_path = output_dir / f'frame_overlays_{i:0{n_digits}d}.png'
        if overlay_path.exists():
            print(f'warning: overwriting existing {overlay_path}')
        fig.savefig(overlay_path, dpi=150, bbox_inches='tight')
        print(f'wrote {overlay_path}')


if __name__ == '__main__':
    main()
