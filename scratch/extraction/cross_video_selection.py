"""Cross-video two-stage selection: per-video flat k-means, then k-means on pooled centroids.

Stage 1: per video, flat spherical k-means with `fine_mult * per_video` clusters on DINOv3 CLS.
Stage 2: pooled stage-1 centroids from all videos are clustered into `n_groups` groups, an
equal number of centroids is drawn per group (water-filled), and one random member frame is
taken per chosen centroid. Per-video allocation is therefore whatever the structure gives.

Compared on pooled metrics against pooled-uniform random and per-video two-level (1k each).

Usage:
    python cross_video_selection.py --proto-dir <dir> --out <json>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kmeans_utils import (
    normalize,
    random_member_per_cluster,
    select_two_level,
    spherical_kmeans,
    water_fill,
)

_logger = logging.getLogger(__name__)

CLASSES = ['Black border', 'Dark aggression', 'Leopard', 'Pale aggression', 'White',
           'Yellow-brown']
PER_VIDEO = 1000
FINE_MULT = 3
N_COARSE = 50
SEEDS = (0, 1, 2)


def select_cross_video(
    xs: list[torch.Tensor],
    n_target: int,
    n_groups: int,
    seed: int,
) -> list[np.ndarray]:
    """Run the two-stage cross-video selection.

    Args:
        xs: per-video normalized embeddings (GPU).
        n_target: total number of frames to select.
        n_groups: number of stage-2 groups.
        seed: rng seed.

    Returns:
        per-video sorted arrays of selected row indices.
    """
    rng = np.random.default_rng(seed)
    cents, owner, labels_v = [], [], []
    for v, x in enumerate(xs):
        k = min(FINE_MULT * PER_VIDEO, len(x))
        labels, c = spherical_kmeans(x, k, seed=seed)
        labels = labels.cpu().numpy()
        counts = np.bincount(labels, minlength=k)
        keep = np.where(counts > 0)[0]
        cents.append(c[torch.from_numpy(keep).to(c.device)])
        owner.append(np.stack([np.full(len(keep), v), keep], axis=1))
        labels_v.append(labels)
    cents = torch.cat(cents)
    owner = np.concatenate(owner)
    g_labels, _ = spherical_kmeans(cents, n_groups, seed=seed)
    g_labels = g_labels.cpu().numpy()
    quota = water_fill(np.bincount(g_labels, minlength=n_groups), n_target)
    chosen = np.concatenate([
        rng.choice(np.where(g_labels == g)[0], quota[g], replace=False)
        for g in range(n_groups) if quota[g] > 0
    ])
    sel = []
    for v in range(len(xs)):
        clusters = owner[chosen][owner[chosen][:, 0] == v][:, 1]
        sel.append(np.sort(random_member_per_cluster(labels_v[v], clusters, rng)))
    return sel


def pool_metrics(
    sel: list[np.ndarray],
    x_all: torch.Tensor,
    offsets: np.ndarray,
    pred_all: np.ndarray,
    coarse: np.ndarray,
) -> dict:
    """Pooled metrics for a selection.

    Args:
        sel: per-video selected row indices.
        x_all: pooled normalized embeddings (GPU).
        offsets: start row of each video in the pooled array.
        pred_all: pooled predicted class.
        coarse: pooled coarse-state ids.

    Returns:
        metrics dict.
    """
    idx = np.concatenate([s + o for s, o in zip(sel, offsets)])
    it = torch.from_numpy(idx).to(x_all.device)
    d = torch.cat([1 - (x_all[i:i + 20000] @ x_all[it].T).max(1).values
                   for i in range(0, len(x_all), 20000)]).cpu().numpy()
    cls = np.bincount(pred_all[idx], minlength=len(CLASSES)) / len(idx)
    sizes = np.bincount(coarse, minlength=N_COARSE)
    pc = np.bincount(coarse[idx], minlength=N_COARSE)
    n = len(x_all)
    rare = np.argsort(sizes)[:10]
    per_video = np.array([len(s) for s in sel])
    return {
        'n': int(len(idx)),
        'cls': cls.tolist(),
        'class_entropy': float(-(cls[cls > 0] * np.log(cls[cls > 0])).sum()),
        'cov_mean': float(d.mean()),
        'cov_p999': float(np.percentile(d, 99.9)),
        'dens_ratio': float((pc[sizes.argmax()] / len(idx)) / (sizes.max() / n)),
        'rare_ratio': float((pc[rare].sum() / len(idx)) / (sizes[rare].sum() / n)),
        'per_video_min': int(per_video.min()),
        'per_video_max': int(per_video.max()),
        'per_video': per_video.tolist(),
    }


def main() -> None:
    """Run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proto-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda')

    names, xs, preds = [], [], []
    for vdir in sorted(args.proto_dir.iterdir()):
        if (vdir / 'clf_prob.npy').exists():
            names.append(vdir.name)
            xs.append(normalize(torch.from_numpy(np.load(vdir / 'dino_cls.npy')).float()
                                .to(device)))
            preds.append(np.load(vdir / 'clf_prob.npy').argmax(1))
    offsets = np.concatenate([[0], np.cumsum([len(x) for x in xs])[:-1]])
    x_all = torch.cat(xs)
    pred_all = np.concatenate(preds)
    coarse = spherical_kmeans(x_all, N_COARSE, seed=0)[0].cpu().numpy()
    n_target = PER_VIDEO * len(xs)
    all_cls = np.bincount(pred_all, minlength=len(CLASSES)) / len(pred_all)
    _logger.info(f'{len(xs)} videos, {len(x_all)} frames, target {n_target}')

    methods = {
        'random_pooled': lambda s: [
            np.sort(r) for r in np.split(
                np.sort(np.random.default_rng(s).choice(len(x_all), n_target, replace=False)),
                np.searchsorted(np.sort(np.random.default_rng(s).choice(
                    len(x_all), n_target, replace=False)), offsets[1:]))
        ],
        'pervideo_twolevel': lambda s: [select_two_level(x, PER_VIDEO, s) for x in xs],
    }
    for g in (100, 400, 800):
        methods[f'xvid_g{g}'] = lambda s, g=g: select_cross_video(xs, n_target, g, s)

    results = {'all_cls': all_cls.tolist(), 'names': names, 'n_frames': [len(x) for x in xs]}
    # random_pooled returns offset-based splits; fix to per-video local indices
    def local(sel_list):
        return [s - o if s.size and s.min() >= o else s for s, o in zip(sel_list, offsets)]

    for name, fn in methods.items():
        ms = []
        for s in SEEDS:
            if name == 'random_pooled':
                flat = np.sort(np.random.default_rng(s).choice(len(x_all), n_target,
                                                               replace=False))
                sel = [flat[(flat >= o) & (flat < o + len(x))] - o
                       for o, x in zip(offsets, xs)]
            else:
                sel = fn(s)
            ms.append(pool_metrics(sel, x_all, offsets, pred_all, coarse))
        results[name] = {k: np.mean([m[k] for m in ms], axis=0).tolist() for k in ms[0]}
        _logger.info(f'{name} done')
    args.out.write_text(json.dumps(results, indent=1))

    print('eligible pooled class share:', dict(zip(CLASSES, np.round(all_cls, 3))))
    print('per-video eligible counts:', results['n_frames'])
    for name in methods:
        r = results[name]
        print(f'\n{name}: dark={r["cls"][1]:.3f} entropy={r["class_entropy"]:.3f} '
              f'cov_mean={r["cov_mean"]:.4f} cov_p999={r["cov_p999"]:.3f} '
              f'dens={r["dens_ratio"]:.2f} rare={r["rare_ratio"]:.2f} '
              f'per_video={np.round(r["per_video"]).astype(int).tolist()}')
        print('   classes', dict(zip(CLASSES, np.round(r['cls'], 3))))


if __name__ == '__main__':
    main()
