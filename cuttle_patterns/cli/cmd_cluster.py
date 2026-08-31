"""Cluster subcommand: assign a discrete cluster label to every frame."""

import argparse
import sys
from pathlib import Path

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.cluster import (
    DEFAULT_RANDOM_STATE,
    build_cluster_dataframe,
    hparams_to_str,
    run_kmeans,
)
from cuttle_patterns.config import load_config
from cuttle_patterns.embeddings import load_latents

DEFAULT_PREDICTIONS_NAME = 'beast_frames'

METHODS = {
    'kmeans': run_kmeans,
}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the cluster subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'cluster',
        help='assign a discrete cluster label to every frame',
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
        help='name of the predicted frame set to cluster, matching the stem of the '
        '--input-dir passed to cuttle predict',
    )
    parser.add_argument(
        '--method',
        choices=sorted(METHODS),
        default='kmeans',
        help='clustering method',
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        required=True,
        help='number of clusters',
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help='clustering method random_state',
    )
    parser.set_defaults(handler=cmd_cluster)


def cmd_cluster(args: argparse.Namespace) -> None:
    """Cluster a model's latents and write per-frame labels to a parquet file.

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

    run_method = METHODS[args.method]
    labels = run_method(X, n_clusters=args.n_clusters, random_state=args.random_state)
    df = build_cluster_dataframe(meta, labels)

    hparams = hparams_to_str(args.n_clusters)
    output_dir = model_dir / paths.CLUSTERS_RELPATH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{args.method}_{hparams}.parquet'
    if output_path.exists():
        print(f'Warning: overwriting existing {output_path}')

    df.to_parquet(output_path, index=False)
    print(f'Wrote {len(df)} rows to {output_path}')
