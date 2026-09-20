"""Embed subcommand: run a frozen pretrained embedder over exported frames."""

import argparse
import sys
from pathlib import Path

import torch

from cuttle_patterns import paths
from cuttle_patterns.cli import DefaultsHelpFormatter
from cuttle_patterns.config import load_config
from cuttle_patterns.embed import (
    DEFAULT_BATCH_SIZE,
    build_embedder,
    find_frame_paths,
    run_embed,
    write_embedder_output,
)
from cuttle_patterns.embedders.dinov3 import ARCH_TO_HF_ID
from cuttle_patterns.embedders.readouts import READOUTS_BY_NAME

DEFAULT_RESOLUTION = 224


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the embed subcommand.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'embed',
        help='run a frozen pretrained embedder (e.g. DINOv3) over exported frames',
        formatter_class=DefaultsHelpFormatter,
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        metavar='PATH',
        help='override the results directory from config',
    )
    parser.add_argument(
        '--backbone',
        choices=list(ARCH_TO_HF_ID),
        required=True,
        help='DINOv3 architecture to embed with',
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=DEFAULT_RESOLUTION,
        help='square input side length, in pixels; must be a multiple of 16',
    )
    parser.add_argument(
        '--readout',
        choices=list(READOUTS_BY_NAME),
        required=True,
        help='readout applied to the backbone tokens',
    )
    parser.add_argument(
        '--model-name',
        default=None,
        help='defaults to {backbone}_{resolution}_{readout}, e.g. dinov3_vitb16_224_cls; '
        f'written to results_dir/{paths.BEAST_MODELS_RELPATH}/{{model_name}}',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help=f'directory of per-video frame subdirectories to embed; defaults to '
        f'results_dir/{paths.BEAST_FRAMES_RELPATH}',
    )
    parser.add_argument(
        '--predictions-name',
        default=None,
        help='names the image_predictions/{predictions_name} subdirectory written into; '
        'defaults to the stem of --input-dir, matching cuttle predict\'s convention',
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help='number of frames per forward pass',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='defaults to cuda if available, else cpu',
    )
    parser.set_defaults(handler=cmd_embed)


def cmd_embed(args: argparse.Namespace) -> None:
    """Run a frozen pretrained embedder over every frame under --input-dir.

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

    input_dir = (
        args.input_dir if args.input_dir is not None else results_dir / paths.BEAST_FRAMES_RELPATH
    )
    device = torch.device(
        args.device if args.device is not None
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model_name = (
        args.model_name if args.model_name is not None
        else f'dinov3_{args.backbone}_{args.resolution}_{args.readout}'
    )
    predictions_name = (
        args.predictions_name if args.predictions_name is not None else input_dir.stem
    )

    try:
        frame_paths = find_frame_paths(input_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f'Error: {e}')
        sys.exit(1)
    print(f'found {len(frame_paths)} frames under {input_dir}')

    print(f'device: {device}')
    try:
        embedder = build_embedder(args.backbone, args.resolution, args.readout, device)
    except ValueError as e:
        print(f'Error: {e}')
        sys.exit(1)

    embeddings, meta = run_embed(embedder, frame_paths, batch_size=args.batch_size)

    model_dir = results_dir / paths.BEAST_MODELS_RELPATH / model_name
    latents_dir = write_embedder_output(embeddings, meta, embedder, model_dir, predictions_name)
    print(
        f'wrote {len(meta)} embeddings ({embedder.dim}-d) to {latents_dir}\n'
        f'next: cuttle reduce --model-name {model_name} (and optionally cuttle cluster '
        f'--model-name {model_name} --n-clusters K)'
    )
