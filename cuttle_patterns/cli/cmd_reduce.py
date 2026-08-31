"""Reduce subcommand: project per-frame BEAST embeddings to 2D via UMAP."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.config import load_config
from cuttle_patterns.embeddings import load_latents
from cuttle_patterns.reduce import (
    DEFAULT_METRIC,
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_RANDOM_STATE,
    build_umap_dataframe,
    hparams_to_str,
    run_umap,
)

DEFAULT_PREDICTIONS_NAME = 'beast_frames'


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the reduce subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'reduce',
        help='project per-frame BEAST embeddings to 2D via UMAP',
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        metavar='PATH',
        help='override the results directory from config',
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help=f'name of a model previously trained with cuttle train and predicted on '
        f'with cuttle predict --save-latents; looked up at '
        f'results_dir/{paths.BEAST_MODELS_RELPATH}/{{model_name}}',
    )
    parser.add_argument(
        '--predictions-name',
        default=DEFAULT_PREDICTIONS_NAME,
        help='name of the predicted frame set to reduce, matching the stem of the '
        '--input-dir passed to cuttle predict',
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=DEFAULT_N_NEIGHBORS,
        help='UMAP n_neighbors',
    )
    parser.add_argument(
        '--min-dist',
        type=float,
        default=DEFAULT_MIN_DIST,
        help='UMAP min_dist',
    )
    parser.add_argument(
        '--metric',
        default=DEFAULT_METRIC,
        help='UMAP metric',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help='UMAP random_state',
    )
    parser.set_defaults(handler=cmd_reduce)


def cmd_reduce(args: argparse.Namespace) -> None:
    """Run UMAP over a model's latents and write the projection to a parquet file.

    Args:
        args: parsed command-line arguments
    """
    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        try:
            config = load_config()
        except (FileNotFoundError, ValueError) as e:
            print(f'Error: {e}')
            sys.exit(1)
        results_dir = config.results_dir

    model_dir = results_dir / paths.BEAST_MODELS_RELPATH / args.model_name
    latents_dir = model_dir / 'image_predictions' / args.predictions_name / 'latents'

    try:
        X, meta = load_latents(latents_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        sys.exit(1)

    umap_xy = run_umap(
        X,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    df = build_umap_dataframe(meta, umap_xy)

    hparams = hparams_to_str(args.n_neighbors, args.min_dist)
    output_dir = model_dir / paths.REDUCE_RELPATH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'umap_{hparams}.parquet'
    if output_path.exists():
        print(f'Warning: overwriting existing {output_path}')

    df.to_parquet(output_path, index=False)
    print(f'Wrote {len(df)} rows to {output_path}')
