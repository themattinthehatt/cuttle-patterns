"""Extract command: prints the frame-selection pipeline stages (each is its own command)."""

import argparse

STAGES = (
    ('extract-eligibility', 'which frames of each aligned video are usable'),
    ('extract-classify', 'skin-pattern class of every frame (crop videos)'),
    ('extract-dino-cls', 'DINOv3 CLS features of every frame (rectangle videos)'),
    ('extract-kmeans-video', 'stage 1: per-video, per-class k-means, with class quotas'),
    ('extract-kmeans-pooled', 'stage 2: pooled k-means across videos -> selection manifest'),
    ('extract-report', 'diagnostics for a selection (coverage, allocation, contact sheets)'),
    ('extract-export', 'write the selected frames as PNGs for BEAST training'),
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the extract command.

    Args:
        subparsers: the subparsers action from the root argument parser
    """
    parser = subparsers.add_parser(
        'extract',
        help='frame selection for BEAST training (prints the stages; run them in order)',
    )
    parser.set_defaults(handler=cmd_extract)


def cmd_extract(args: argparse.Namespace) -> None:
    """Print the stages of the frame-selection pipeline.

    Args:
        args: parsed command-line arguments
    """
    print('Frame selection for BEAST training is a sequence of stages; run them in order')
    print('(`cuttle <stage> --help` for each stage\'s options):\n')
    width = max(len(name) for name, _ in STAGES)
    for idx, (name, description) in enumerate(STAGES, start=1):
        print(f'  {idx}. cuttle {name:<{width}}  {description}')
    print('\nEach stage is resumable (finished videos are skipped) and prints the next stage')
    print('when it finishes. See README.md ("Frame selection") for an example run.')
