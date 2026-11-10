"""Run the single-view Ensemble Kalman Smoother (EKS) over several pose models' predictions.

For every video that has a `video_preds` CSV in all of the given models (see
docs/pose_estimation.md), feeds those CSVs to `eks.fit_eks_singlecam` with a fixed smoothing
parameter and writes the smoothed tail/neck predictions to
`{output_dir}/video_preds/{video_name}.csv`, in the same 3-header format as the inputs.

With a large `--smooth-param` (default 1000) the Kalman smoother barely smooths, so the
result is close to a per-frame median over the ensemble members.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI; run directly, e.g.:

    python scripts/pose_run_eks.py \
        --models iter-1.2_vits-dino_seed-0 iter-1.2_vits-dino_seed-1 iter-1.2_vits-dino_seed-2 \
        --output-name iter-1.2_eks
"""

import argparse
import logging
from pathlib import Path

from eks.singlecam_smoother import fit_eks_singlecam

logger = logging.getLogger(__name__)

DEFAULT_PROJECT_DIR = Path('/media/mattw/CUTTLE/pose-estimation/cuttle-test')
DEFAULT_SMOOTH_PARAM = 1000.0
# per-video files written by Lightning Pose alongside the predictions; not model outputs
SKIP_SUFFIX = '_temporal_norm'


def find_video_names(dir_video_preds: list[Path]) -> list[str]:
    """Find videos with predictions in every model's video_preds directory.

    Args:
        dir_video_preds: one `video_preds` directory per ensemble member.

    Returns:
        Sorted video names (CSV stems) present in all directories, excluding the
        `_temporal_norm` companion files.
    """
    names_per_model = [
        {p.stem for p in d.glob('*.csv') if not p.stem.endswith(SKIP_SUFFIX)}
        for d in dir_video_preds
    ]
    names_all = set.union(*names_per_model)
    names_common = set.intersection(*names_per_model)
    if names_all != names_common:
        logger.warning(
            f'skipping {len(names_all - names_common)} videos missing from some models: '
            f'{sorted(names_all - names_common)}'
        )
    return sorted(names_common)


def run_eks(
    dir_models: list[Path],
    dir_output: Path,
    smooth_param: float,
    overwrite: bool,
) -> None:
    """Run EKS on every video shared by the given models.

    Args:
        dir_models: model directories, each containing `video_preds/`.
        dir_output: output model directory; results go to `{dir_output}/video_preds`.
        smooth_param: EKS smoothing parameter (smaller values smooth more).
        overwrite: recompute videos whose output CSV already exists.
    """
    dir_video_preds = [d / 'video_preds' for d in dir_models]
    for d in dir_video_preds:
        if not d.is_dir():
            raise FileNotFoundError(f'no video_preds directory at {d}')

    video_names = find_video_names(dir_video_preds)
    logger.info(f'running EKS on {len(video_names)} videos from {len(dir_models)} models')

    for idx, video_name in enumerate(video_names):
        path_save = dir_output / 'video_preds' / f'{video_name}.csv'
        if path_save.exists() and not overwrite:
            logger.info(f'[{idx + 1}/{len(video_names)}] {video_name}: exists, skipping')
            continue
        logger.info(f'[{idx + 1}/{len(video_names)}] {video_name}')
        fit_eks_singlecam(
            input_source=[str(d / f'{video_name}.csv') for d in dir_video_preds],
            save_file=str(path_save),
            smooth_param=smooth_param,
        )


def main() -> None:
    """Parse arguments and run EKS over the given models' predictions."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help=f'Lightning Pose project root (contains models/); default {DEFAULT_PROJECT_DIR}',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        metavar='NAME',
        help='ensemble member directory names under {project_dir}/models',
    )
    parser.add_argument(
        '--output-name',
        required=True,
        help='output directory name under {project_dir}/models (e.g. iter-1.2_eks)',
    )
    parser.add_argument(
        '--smooth-param',
        type=float,
        default=DEFAULT_SMOOTH_PARAM,
        help=f'EKS smoothing parameter; default {DEFAULT_SMOOTH_PARAM}',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='recompute videos whose output CSV already exists (default: skip them)',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    dir_models = args.project_dir / 'models'
    run_eks(
        dir_models=[dir_models / name for name in args.models],
        dir_output=dir_models / args.output_name,
        smooth_param=args.smooth_param,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
