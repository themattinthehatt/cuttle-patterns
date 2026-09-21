"""Clusterview subcommand: plot a grid of representative frames for each cluster."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.config import load_config
from cuttle_patterns.visualization.cluster_frames import (
    DEFAULT_N_COLS,
    DEFAULT_N_FRAMES,
    DEFAULT_SEED,
    plot_cluster_grid,
    sample_cluster_frames,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the clusterview subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'clusterview',
        help='plot a grid of representative frames for each cluster',
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
        help='BEAST model directory name, e.g. iter-1.1_msps-vae_d16',
    )
    parser.add_argument(
        '--cluster-run',
        required=True,
        help='stem of the cluster parquet file written by cuttle cluster, e.g. kmeans_k16',
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=DEFAULT_N_FRAMES,
        help='number of frames to sample per cluster',
    )
    parser.add_argument(
        '--n-cols',
        type=int,
        default=DEFAULT_N_COLS,
        help='number of grid columns; rows are derived from --n-frames',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='random seed for per-cluster frame sampling',
    )
    parser.set_defaults(handler=cmd_clusterview)


def cmd_clusterview(args: argparse.Namespace) -> None:
    """Sample and plot representative frames for every cluster in a cluster run.

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

    cluster_path = (
        results_dir
        / paths.BEAST_MODELS_RELPATH
        / args.model_name
        / paths.CLUSTERS_RELPATH
        / f'{args.cluster_run}.parquet'
    )
    if not cluster_path.exists():
        print(f'Error: no cluster file found at {cluster_path}')
        sys.exit(1)
    df = pd.read_parquet(cluster_path)

    output_dir = (
        results_dir
        / paths.BEAST_MODELS_RELPATH
        / args.model_name
        / paths.CLUSTERS_RELPATH
        / args.cluster_run
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cluster_width = max(2, len(str(int(df['cluster'].max()))))

    for cluster_label in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_label]
        sample_df = sample_cluster_frames(cluster_df, args.n_frames, rng)

        fig = plot_cluster_grid(sample_df, results_dir, cluster_label, args.n_cols)
        output_path = output_dir / f'cluster_{cluster_label:0{cluster_width}d}.png'
        if output_path.exists():
            print(f'Warning: overwriting existing {output_path}')
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f'cluster {cluster_label}: {len(cluster_df)} members, wrote {output_path}')

    print('next: cuttle serve to visualize')
