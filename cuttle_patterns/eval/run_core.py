"""Command-line entry point for the eval harness's core (v1) scoreboard.

Deliberately not wired into the `cuttle` CLI — see `docs/eval_plan.md`. Run it
directly from the repo root, e.g.:

    python -m cuttle_patterns.eval.run_core \\
        --classifier-name iter-1.1_classifier_d512 \\
        --model-name iter-1.1_resnet-18_d16:ae \\
        --model-name iter-1.1_msps-vae_d16:msps_vae

See `cuttle_patterns/eval/README.md` for the full walkthrough.
"""

import argparse
from pathlib import Path

from cuttle_patterns.config import load_config
from cuttle_patterns.eval.build_manifest import build_eval_manifest
from cuttle_patterns.eval.load_embeddings import EmbedderSpec
from cuttle_patterns.eval.report import run_report, write_report
from cuttle_patterns.paths import BEAST_MODELS_RELPATH, EVAL_RELPATH

MSPS_VAE_SUBSPACES = ('unsupervised', 'background', 'all')
MSPS_VAE_SUBSPACE_LABELS = {'unsupervised': 'z_u', 'background': 'z_b', 'all': 'z_all'}


def build_specs(results_dir: Path, model_arg: str) -> list[EmbedderSpec]:
    """Build one or more `EmbedderSpec`s from a `--model-name` argument.

    Args:
        results_dir: root results directory.
        model_arg: `{model_name}:{kind}`, where `kind` is `ae` for a single-subspace
            model (its latents loaded as-is, scoreboard id = `model_name`) or
            `msps_vae` to expand into its three subspaces (`z_u`, `z_b`, `z_all`) as
            separate scoreboard rows.

    Returns:
        one or more `EmbedderSpec`s for `model_arg`.

    Raises:
        ValueError: if `model_arg` isn't `{model_name}:{kind}`, or `kind` isn't
            recognized.
    """
    try:
        model_name, kind = model_arg.split(':')
    except ValueError as error:
        raise ValueError(f'expected "{{model_name}}:{{kind}}", got {model_arg!r}') from error
    model_dir = results_dir / BEAST_MODELS_RELPATH / model_name

    if kind == 'ae':
        return [EmbedderSpec(id=model_name, model_dir=model_dir, subspace='all')]
    if kind == 'msps_vae':
        return [
            EmbedderSpec(
                id=f'{model_name}_{MSPS_VAE_SUBSPACE_LABELS[subspace]}',
                model_dir=model_dir,
                subspace=subspace,
            )
            for subspace in MSPS_VAE_SUBSPACES
        ]
    raise ValueError(f'unrecognized model kind {kind!r}; expected "ae" or "msps_vae"')


def main() -> None:
    """Build the eval manifest, score every requested model, and write the scoreboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--classifier-name', required=True,
        help='stem of results_dir/classifications/{name}.parquet to join as the pattern proxy.',
    )
    parser.add_argument(
        '--model-name', action='append', required=True, dest='model_args',
        help='{model_name}:{kind}, kind is "ae" or "msps_vae"; repeatable.',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=None,
        help='defaults to results_dir/eval/ (see cuttle_patterns.paths.EVAL_RELPATH).',
    )
    parser.add_argument('--n-clusters', type=int, default=16)
    args = parser.parse_args()

    config = load_config()
    manifest = build_eval_manifest(config.results_dir, args.classifier_name)

    specs = [
        spec
        for model_arg in args.model_args
        for spec in build_specs(config.results_dir, model_arg)
    ]

    out_dir = args.out_dir if args.out_dir is not None else config.results_dir / EVAL_RELPATH
    scoreboard = run_report(specs, manifest, n_clusters=args.n_clusters)
    write_report(scoreboard, out_dir)
    print(scoreboard.to_string(index=False))


if __name__ == '__main__':
    main()
