"""Compare frame-selection methods on embedded eligible frames, 1k picks per video.

Methods: `baseline` (old recipe: top-50% BEAST motion energy, PCA-32 on 32x32 pixels, flat
k-means, nearest to centroid; restricted to eligible frames), `random` (uniform), `flat`
(flat k-means on DINOv3 CLS, random member), `twolevel` (balanced hierarchical k-means,
random member). Metrics are per video, averaged over videos (and seeds for stochastic
methods): classifier-class shares of the pool vs. all eligible frames, coverage
(nearest-selected cosine distance), densest/rarest coarse-state ratios, isolated-frame
retention. Also checks how strongly embeddings separate by video.

Usage:
    python compare_selection.py --proto-dir <dir> --elig-dir <dir> --cache-dir <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from beast.video import compute_video_motion_energy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cuttle_patterns.config import load_config
from kmeans_utils import normalize, select_flat, select_two_level, spherical_kmeans

_logger = logging.getLogger(__name__)

N_TARGET = 1000
SEEDS = (0, 1, 2)
CLASSES = ['Black border', 'Dark aggression', 'Leopard', 'Pale aggression', 'White',
           'Yellow-brown']
N_COARSE = 50
K_ISO = 10


def select_baseline(video_path: Path, idx_elig: np.ndarray, cache_path: Path) -> np.ndarray:
    """Old extraction recipe restricted to eligible frames.

    Args:
        video_path: crop video.
        idx_elig: eligible frame indices.
        cache_path: .npy cache of selected frame indices.

    Returns:
        selected frame indices (values from `idx_elig`).
    """
    if cache_path.exists():
        return np.load(cache_path)
    me, frames = compute_video_motion_energy(video_path, resize_dims=32, return_frames=True)
    me_c = me[idx_elig]
    idx_high = idx_elig[me_c > np.percentile(me_c, 50)]
    emb = PCA(n_components=32).fit_transform(frames[idx_high])
    km = KMeans(N_TARGET, n_init='auto', random_state=0).fit(emb)
    d = np.linalg.norm(emb[:, :, None] - km.cluster_centers_.T[None], axis=1)
    sel = np.unique(idx_high[np.argmin(d, axis=0)])
    np.save(cache_path, sel)
    return sel


def evaluate(
    sel: np.ndarray,
    x: torch.Tensor,
    prob_argmax: np.ndarray,
    coarse: np.ndarray,
    iso_top: np.ndarray,
    nn_sim: torch.Tensor | None = None,
) -> dict:
    """Compute selection metrics for one pool.

    Args:
        sel: selected row indices into the eligible array.
        x: (n, d) normalized embeddings, GPU.
        prob_argmax: (n,) predicted class per eligible frame.
        coarse: (n,) coarse-state id per eligible frame.
        iso_top: row indices of the 1% most isolated frames.
        nn_sim: unused placeholder.

    Returns:
        dict of metrics.
    """
    sel_t = torch.from_numpy(sel).to(x.device)
    sim = torch.cat([x[i:i + 20000] @ x[sel_t].T for i in range(0, len(x), 20000)])
    dist = 1.0 - sim.max(dim=1).values
    dist = dist.cpu().numpy()
    n = len(x)
    cls_pool = np.bincount(prob_argmax[sel], minlength=len(CLASSES)) / len(sel)
    sizes = np.bincount(coarse, minlength=N_COARSE)
    pool_c = np.bincount(coarse[sel], minlength=N_COARSE)
    dens = int(np.argmax(sizes))
    order = np.argsort(sizes)
    rare = order[:10]
    return {
        'n_sel': int(len(sel)),
        'cls_pool': cls_pool.tolist(),
        'cov_mean': float(dist.mean()),
        'cov_p95': float(np.percentile(dist, 95)),
        'cov_p999': float(np.percentile(dist, 99.9)),
        'dens_ratio': float((pool_c[dens] / len(sel)) / (sizes[dens] / n)),
        'rare_ratio': float((pool_c[rare].sum() / len(sel)) / (sizes[rare].sum() / n)),
        'iso_kept': float(np.isin(iso_top, sel).mean()),
    }


def main() -> None:
    """Run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proto-dir', type=Path, required=True)
    parser.add_argument('--elig-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rect_dir = load_config().results_dir / 'rectangles'
    device = torch.device('cuda')

    results = {}
    pooled = {}
    for vdir in sorted(args.proto_dir.iterdir()):
        if not (vdir / 'clf_prob.npy').exists():
            continue
        stem = vdir.name
        idx_elig = np.load(vdir / 'frame_idx.npy')
        x = normalize(torch.from_numpy(np.load(vdir / 'dino_cls.npy')).float().to(device))
        prob = np.load(vdir / 'clf_prob.npy')
        pred = prob.argmax(1)
        n = len(x)
        _logger.info(f'{stem}: {n} eligible frames')

        # coarse states and isolation, shared by all methods
        coarse = spherical_kmeans(x, N_COARSE, seed=0)[0].cpu().numpy()
        iso = torch.cat([
            (x[i:i + 4000] @ x.T).topk(K_ISO + 1, dim=1).values[:, 1:].mean(dim=1)
            for i in range(0, n, 4000)
        ])
        iso_top = torch.argsort(iso)[:max(n // 100, 1)].cpu().numpy()  # lowest mean sim

        pools = {}
        base = select_baseline(rect_dir / f'{stem}.mp4', idx_elig,
                               args.cache_dir / f'{stem}_baseline.npy')
        pools['baseline'] = [np.searchsorted(idx_elig, base)]
        pools['random'] = [np.sort(np.random.default_rng(s).choice(n, N_TARGET, replace=False))
                           for s in SEEDS]
        pools['flat'] = [select_flat(x, N_TARGET, s) for s in SEEDS]
        pools['twolevel'] = [select_two_level(x, N_TARGET, s) for s in SEEDS]

        res = {'all_cls': (np.bincount(pred, minlength=len(CLASSES)) / n).tolist(), 'n': n}
        for name, sels in pools.items():
            ms = [evaluate(s, x, pred, coarse, iso_top) for s in sels]
            res[name] = {k: (np.mean([m[k] for m in ms], axis=0).tolist()) for k in ms[0]}
        results[stem] = res
        pooled[stem] = (x.cpu(), idx_elig)

    # video-identity check: nearest neighbor (other-video-or-far-in-time) same video?
    ident = {}
    rng = np.random.default_rng(0)
    xs, vid, fr = [], [], []
    for v, (x, idx) in enumerate(pooled.values()):
        s = rng.choice(len(x), 2000, replace=False)
        xs.append(x[s]); vid += [v] * len(s); fr += idx[s].tolist()
    xs = torch.cat(xs).to(device); vid = np.array(vid); fr = np.array(fr)
    for label, feats in [
        ('raw', xs),
        ('video_centered', normalize(torch.cat([
            xs[vid == v] - xs[vid == v].mean(0, keepdim=True) for v in range(len(pooled))]))),
    ]:
        v_sorted = np.concatenate([np.where(vid == v)[0] for v in range(len(pooled))])
        f = feats  # rows already ordered by video (concatenated above)
        sim = f @ f.T
        same = torch.from_numpy(vid[:, None] == vid[None, :]).to(device)
        near = torch.from_numpy(np.abs(fr[:, None] - fr[None, :]) < 100).to(device)
        sim = sim.masked_fill(same & near, -2)  # exclude self and temporal neighbors
        nn = sim.argmax(dim=1).cpu().numpy()
        ident[label] = float((vid[nn] == vid).mean())
    ident['chance'] = float(1 / len(pooled))
    results['_video_identity_nn_same_video'] = ident
    args.out.write_text(json.dumps(results, indent=1))

    # summary tables
    stems = [s for s in results if not s.startswith('_')]
    print(f'videos: {len(stems)}, identity check: {ident}')
    print('\nclass share of eligible (mean over videos):',
          dict(zip(CLASSES, np.mean([results[s]['all_cls'] for s in stems], 0).round(3))))
    for name in ['baseline', 'random', 'flat', 'twolevel']:
        cls = np.mean([results[s][name]['cls_pool'] for s in stems], 0).round(3)
        print(f'{name:9s} class share', dict(zip(CLASSES, cls)))
    print('\nmethod    cov_mean  cov_p95  cov_p999  dens_ratio  rare_ratio  iso_kept')
    for name in ['baseline', 'random', 'flat', 'twolevel']:
        m = {k: np.mean([results[s][name][k] for s in stems])
             for k in ['cov_mean', 'cov_p95', 'cov_p999', 'dens_ratio', 'rare_ratio', 'iso_kept']}
        print(f'{name:9s} ' + '  '.join(f'{v:8.3f}' for v in m.values()))


if __name__ == '__main__':
    main()
