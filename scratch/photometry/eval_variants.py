"""Score one or more models across the Step 4 variant results_dirs with the eval harness.

For each variant (a results_dir under `.iter-1.0-photometry/variants/`) builds the eval manifest
(same frames, fixed iter-1.0 classifier labels) and runs the five core metrics for every
`--model-name`, then prints one table with a row per (model, variant). Per-variant scoreboards go
to `variants/{variant}/eval/`.

Example:
    python scratch/photometry/eval_variants.py --model-name dinov3_vitb16_224_cls:ae \\
        --variants orig raw final wrong norm
"""

import argparse
from pathlib import Path

import pandas as pd

from cuttle_patterns.eval.build_manifest import build_eval_manifest
from cuttle_patterns.eval.report import run_report, write_report
from cuttle_patterns.eval.run_core import build_specs

VARIANTS_ROOT = Path('/media/mattw/CUTTLE/results/.iter-1.0-photometry/variants')
CLASSIFIER = 'iter-1.1_classifier_d512'
SHOW = [
    'ami_cluster_video_within_class_mean', 'effective_videos_per_cluster_mean',
    'cross_video_knn_accuracy', 'linear_probe_accuracy', 'ami_cluster_class_mean',
]


def main() -> None:
    """Run the eval for every variant and print the combined table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', action='append', required=True, dest='model_args')
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--exclude', nargs='*', default=[], help='skip embedder ids ending with these')
    parser.add_argument('--tag', default='', help='suffix for the per-variant eval subfolder')
    args = parser.parse_args()
    rows = []
    for variant in args.variants:
        results_dir = VARIANTS_ROOT / variant
        manifest = build_eval_manifest(results_dir, CLASSIFIER)
        specs = [
            s for m in args.model_args for s in build_specs(results_dir, m)
            if not any(s.id.endswith(x) for x in args.exclude)
        ]
        board = run_report(specs, manifest, n_clusters=16)
        write_report(board, results_dir / 'eval' / args.tag if args.tag else results_dir / 'eval')
        rows.append(board.assign(variant=variant))
    table = pd.concat(rows)
    table = table.set_index(['embedder', 'variant'])
    print(table[SHOW + ['cross_video_knn_accuracy']].loc[:, SHOW].round(3).to_string())
    print(f'\n{len(manifest)} frames per variant')


if __name__ == '__main__':
    main()
