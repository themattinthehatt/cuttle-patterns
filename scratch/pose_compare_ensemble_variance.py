"""Compare cross-seed ensemble variance between two pose-model training iterations.

For each video, loads tail/neck predictions from all seeds of two iterations (e.g.
iter-1.0 and iter-1.1, see docs/pose_estimation.md), computes per-frame ensemble variance
(var(x across seeds) + var(y across seeds)) within each iteration, keeps only frames where
the *new* iteration's max-across-seeds likelihood is >= --likelihood-thresh, and plots a
box-plot grid (one subplot per video, one box per keypoint x iteration) so it's easy to see
whether the newer iteration's predictions have gotten more consistent.

Not part of the installed package -- rerun anytime new iterations finish training.

Usage:
    python scratch/pose_compare_ensemble_variance.py \
        --old-iteration iter-1.0 --new-iteration iter-1.1
"""

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuttle_patterns.config import load_config
from cuttle_patterns.ingest import build_manifest
from cuttle_patterns.preprocessing.pose import DEFAULT_LIKELIHOOD_THRESH, load_pose_predictions

DEFAULT_PROJECT_DIR = Path('/media/mattw/CUTTLE/pose-estimation/cuttle-test')
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_BACKBONE = 'vits-dino'
DEFAULT_N_COLS = 6

# order the two keypoint groups appear left-to-right within each subplot
PLOT_KEYPOINT_ORDER = ('neck', 'tail')
OLD_COLOR = 'tab:blue'
NEW_COLOR = 'tab:orange'
# variance floor so log-scale plotting never hits exactly zero
MIN_PLOTTED_VARIANCE = 1e-6


def model_name(iteration: str, backbone: str, seed: int) -> str:
    """Build a model directory name from its iteration/backbone/seed."""
    return f'{iteration}_{backbone}_seed-{seed}'


def load_iteration_predictions(
    project_dir: Path,
    iteration: str,
    backbone: str,
    seeds: list[int],
    video_name: str,
) -> list[pd.DataFrame] | None:
    """Load one video's predictions from every seed of one iteration.

    Args:
        project_dir: Lightning Pose project root (contains models/).
        iteration: iteration prefix, e.g. 'iter-1.0'.
        backbone: backbone name used in the model directory naming convention.
        seeds: seed numbers expected for this iteration.
        video_name: video stem to look up under each model's video_preds/.

    Returns:
        one tidy prediction frame (see pose.load_pose_predictions) per seed, in seed
        order, or None if any seed is missing a video_preds CSV for this video.

    Raises:
        ValueError: if the loaded predictions don't all have the same number of frames.
    """
    predictions = []
    for seed in seeds:
        csv_path = (
            project_dir / 'models' / model_name(iteration, backbone, seed)
            / 'video_preds' / f'{video_name}.csv'
        )
        if not csv_path.exists():
            return None
        predictions.append(load_pose_predictions(csv_path))

    lengths = {len(df) for df in predictions}
    if len(lengths) > 1:
        raise ValueError(f'{video_name}: prediction length mismatch across seeds: {lengths}')

    return predictions


def compute_ensemble_variance(predictions: list[pd.DataFrame], keypoint: str) -> np.ndarray:
    """Compute per-frame cross-seed variance for one keypoint (var(x) + var(y))."""
    x_stack = np.stack([df[f'{keypoint}_x'].to_numpy() for df in predictions])
    y_stack = np.stack([df[f'{keypoint}_y'].to_numpy() for df in predictions])
    return x_stack.var(axis=0) + y_stack.var(axis=0)


def compute_max_likelihood(predictions: list[pd.DataFrame], keypoint: str) -> np.ndarray:
    """Compute per-frame max-across-seeds likelihood for one keypoint."""
    stack = np.stack([df[f'{keypoint}_likelihood'].to_numpy() for df in predictions])
    return stack.max(axis=0)


def collect_video_variances(
    old_predictions: list[pd.DataFrame],
    new_predictions: list[pd.DataFrame],
    likelihood_thresh: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pair up old/new ensemble variance per keypoint, filtered by new-iteration confidence.

    A frame is kept for a given keypoint only if the new iteration's max-across-seeds
    likelihood for that keypoint is >= likelihood_thresh; the same frames are then used
    for both the old and new variance arrays, so the comparison is apples-to-apples.

    Args:
        old_predictions: per-seed prediction frames for the old iteration.
        new_predictions: per-seed prediction frames for the new iteration.
        likelihood_thresh: minimum new-iteration max-across-seeds likelihood to keep a
            frame.

    Returns:
        keypoint -> (old_variance, new_variance) arrays, one entry per surviving frame.
    """
    result = {}
    for keypoint in PLOT_KEYPOINT_ORDER:
        old_variance = compute_ensemble_variance(old_predictions, keypoint)
        new_variance = compute_ensemble_variance(new_predictions, keypoint)
        keep = compute_max_likelihood(new_predictions, keypoint) >= likelihood_thresh
        result[keypoint] = (old_variance[keep], new_variance[keep])

    return result


def plot_video_comparison(ax: plt.Axes, video_name: str, variances: dict) -> None:
    """Draw one subplot's 4-box comparison (neck old/new, tail old/new) for one video.

    Args:
        ax: axes to draw into.
        video_name: used as the subplot title.
        variances: keypoint -> (old_variance, new_variance), from collect_video_variances.
    """
    data = []
    colors = []
    positions = []
    group_centers = []
    pos = 1
    for keypoint in PLOT_KEYPOINT_ORDER:
        old_variance, new_variance = variances[keypoint]
        data.append(np.maximum(old_variance, MIN_PLOTTED_VARIANCE))
        colors.append(OLD_COLOR)
        positions.append(pos)
        data.append(np.maximum(new_variance, MIN_PLOTTED_VARIANCE))
        colors.append(NEW_COLOR)
        positions.append(pos + 1)
        group_centers.append(pos + 0.5)
        pos += 3

    boxplot = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True)
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(PLOT_KEYPOINT_ORDER)
    ax.set_yscale('log')
    ax.set_title(video_name, fontsize=8)
    ax.tick_params(axis='both', labelsize=7)


def main() -> None:
    """Parse arguments, compute per-video ensemble variances, and save the comparison figure."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help='Lightning Pose project root (contains models/)',
    )
    parser.add_argument('--old-iteration', default='iter-1.0', help='baseline iteration prefix')
    parser.add_argument('--new-iteration', default='iter-1.1', help='newer iteration prefix')
    parser.add_argument(
        '--backbone',
        default=DEFAULT_BACKBONE,
        help='backbone name used in the model directory naming convention',
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='seed numbers to load',
    )
    parser.add_argument(
        '--likelihood-thresh',
        type=float,
        default=DEFAULT_LIKELIHOOD_THRESH,
        help='minimum new-iteration max-across-seeds likelihood to keep a frame',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='raw video directory; defaults to data_dir from the cuttle config',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='defaults to {project_dir}/qc/ensemble_variance_comparison.png',
    )
    parser.add_argument(
        '--n-cols', type=int, default=DEFAULT_N_COLS, help='number of subplot columns',
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else load_config().data_dir
    output_path = (
        args.output if args.output is not None
        else args.project_dir / 'qc' / 'ensemble_variance_comparison.png'
    )

    video_variances = {}
    manifest = build_manifest(data_dir)
    for _, row in manifest.iterrows():
        video_name = Path(row['video_path']).stem

        old_predictions = load_iteration_predictions(
            args.project_dir, args.old_iteration, args.backbone, args.seeds, video_name,
        )
        new_predictions = load_iteration_predictions(
            args.project_dir, args.new_iteration, args.backbone, args.seeds, video_name,
        )
        if old_predictions is None or new_predictions is None:
            missing = args.old_iteration if old_predictions is None else args.new_iteration
            print(f'{video_name}: missing one or more {missing} seeds, skipping')
            continue

        variances = collect_video_variances(
            old_predictions, new_predictions, args.likelihood_thresh,
        )
        video_variances[video_name] = variances

        counts = ', '.join(f'{kp} n={len(old)}' for kp, (old, _) in variances.items())
        print(f'{video_name}: {counts}')

    if not video_variances:
        print('no videos had predictions for both iterations, nothing to plot')
        return

    video_names = sorted(video_variances)
    n_cols = args.n_cols
    n_rows = -(-len(video_names) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)

    for idx, video_name in enumerate(video_names):
        ax = axes[idx // n_cols][idx % n_cols]
        plot_video_comparison(ax, video_name, video_variances[video_name])

    for idx in range(len(video_names), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis('off')

    legend_handles = [
        mpatches.Patch(color=OLD_COLOR, label=args.old_iteration),
        mpatches.Patch(color=NEW_COLOR, label=args.new_iteration),
    ]
    fig.suptitle(
        f'ensemble variance by keypoint, frames filtered to {args.new_iteration} '
        f'max likelihood >= {args.likelihood_thresh}',
        y=0.998,
    )
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f'wrote {output_path}')


if __name__ == '__main__':
    main()
