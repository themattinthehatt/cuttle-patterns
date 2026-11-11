"""Day / fog split of the clustering metrics across Step 4 variants.

Two ways, both on saved embeddings (no GPU):
- refit: k-means refit on each subset's frames (`score_embedder`'s clustering recipe)
- fixed: k-means fit once on all frames (as in the full eval); the subset's frames are scored under
  those assignments

Metrics: AMI(cluster, video) within class, effective videos per cluster, AMI(cluster, class).
kNN / linear probe are not recomputed here (slow, and flat across variants in the full runs).

Example:
    python scratch/photometry/eval_split.py out.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cuttle_patterns.eval.build_manifest import build_eval_manifest
from cuttle_patterns.eval.clustering import prepare_for_clustering, run_kmeans_multiseed
from cuttle_patterns.eval.load_embeddings import load_embedder_matrix
from cuttle_patterns.eval.metrics import (
    ami_cluster_vs_class,
    ami_cluster_vs_video_within_class,
    effective_videos_per_cluster,
)
from cuttle_patterns.eval.run_core import build_specs

VARIANTS_ROOT = Path('/media/mattw/CUTTLE/results/.iter-1.0-photometry/variants')
VARIANTS = ['orig', 'raw', 'final', 'wrong', 'norm']
MODELS = [
    'vgg19_1_448_gram_k64_taper:ae', 'vgg19_12345_448_gram_k28_taper:ae',
    'iter-1.1_msps-vae-mask_d16:msps_vae', 'dinov3_vitb16_224_gram_k64_taper:ae',
    'iter-1.1_resnet-18_d16:ae',
]
FOGGED = ('Day1_Tank2', 'Day1_Tank6')  # fog ~0.02 in the fitted params; Day1_Tank4 is excluded


def subsets(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean frame masks per subset."""
    day1 = manifest.session_id.str.startswith('Day1').to_numpy()
    fog = manifest.session_id.isin(FOGGED).to_numpy()
    return {
        'all': np.ones(len(manifest), bool), 'day1': day1, 'day23': ~day1,
        'fogged': fog, 'nofog': ~fog, 'day1_nofog': day1 & ~fog,
    }


def score(labels_per_seed: list[np.ndarray], mask: np.ndarray, manifest: pd.DataFrame) -> dict:
    """Mean over seeds of the three cluster metrics on the masked frames."""
    video = manifest.video_name.to_numpy()[mask]
    cls = manifest.predicted_pattern.to_numpy()[mask]
    out = {'ami_video': [], 'eff_videos': [], 'ami_class': []}
    for labels in labels_per_seed:
        lab = labels[mask]
        out['ami_video'].append(ami_cluster_vs_video_within_class(lab, video, cls))
        out['eff_videos'].append(effective_videos_per_cluster(lab, video))
        out['ami_class'].append(ami_cluster_vs_class(lab, cls))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main(out_csv: Path) -> None:
    """Score all models/variants/subsets and write one tidy csv."""
    rows = []
    for variant in VARIANTS:
        results_dir = VARIANTS_ROOT / variant
        manifest = build_eval_manifest(results_dir, 'iter-1.1_classifier_d512')
        masks = subsets(manifest)
        for model_arg in MODELS:
            for spec in build_specs(results_dir, model_arg):
                if spec.id.endswith('_z_all'):
                    continue
                X = load_embedder_matrix(spec, manifest)
                reduced, _ = prepare_for_clustering(X)
                full_labels = run_kmeans_multiseed(reduced, n_clusters=16)
                for name, mask in masks.items():
                    fixed = score(full_labels, mask, manifest)
                    sub = manifest[mask].reset_index(drop=True)
                    r_sub, _ = prepare_for_clustering(X[mask])
                    refit = score(
                        run_kmeans_multiseed(r_sub, n_clusters=16),
                        np.ones(len(sub), bool), sub,
                    )
                    for mode, res in (('fixed', fixed), ('refit', refit)):
                        rows.append({
                            'embedder': spec.id, 'variant': variant, 'subset': name,
                            'mode': mode, 'n_frames': int(mask.sum()),
                            'n_videos': int(manifest.video_name[mask].nunique()), **res,
                        })
                print(f'{variant} {spec.id} done', flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
