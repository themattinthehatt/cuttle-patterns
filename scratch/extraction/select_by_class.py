"""Class-quota frame selection with per-video-first hierarchical k-means.

Status: exploratory; produces the candidate `beast_frames` manifest (see README.md).

1. Quotas: `--budget` frames split across classifier classes with shares proportional to
   `class_count ** --quota-power` (0.5 = square root, 0 = uniform, 1 = proportional),
   rounded by largest remainder. A class never gets more than it has.
2. Stage 1 (per video, per class): spherical k-means on DINOv3 CLS of the video's eligible
   frames of that class, with about `--fine-mult` times the video's proportional share of
   the class quota as the cluster count (all frames kept if the video has fewer).
3. Stage 2 (per class, across videos): pool all stage-1 centroids, spherical k-means into
   groups (about `quota / --picks-per-group` groups), draw an equal number of centroids
   per group (water-filled), and take one random member frame per chosen centroid.

Inputs: `eligibility_final/*.parquet` (from `final_eligibility.py`) and
`{results_dir}/embeddings/dinov3_vits16/{video}.npy`. Videos without an embedding file
yet are skipped with a warning (or error with --strict). Output (`--out-dir`):
`manifest.parquet` (one row per selected frame) and `summary.json`.

Usage:
    python select_by_class.py --budget 132000 --out-dir <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cuttle_patterns.config import load_config
from kmeans_utils import normalize, spherical_kmeans, water_fill

_logger = logging.getLogger(__name__)

EXTRACTION_DIR = Path('/media/mattw/CUTTLE/results/.extraction')
CLASSES = ['Dark aggression', 'Leopard', 'Pale aggression', 'White', 'Yellow-brown']
EMBEDDING_NAME = 'dinov3_vits16'


def compute_quotas(counts: np.ndarray, budget: int, power: float) -> np.ndarray:
    """Split a budget across classes proportionally to count**power, capped at availability.

    Args:
        counts: available frames per class.
        budget: total frames to select (clipped to counts.sum()).
        power: exponent on the class counts (0.5 = square root).

    Returns:
        integer quota per class, summing to min(budget, counts.sum()).
    """
    budget = min(budget, int(counts.sum()))
    quota = np.zeros(len(counts), dtype=int)
    open_cls = counts > 0
    remaining = budget
    while remaining > 0:
        w = np.where(open_cls, counts.astype(float) ** power, 0.0)
        ideal = w / w.sum() * remaining
        add = np.floor(ideal).astype(int)
        leftover = remaining - int(add.sum())
        add[np.argsort(-(ideal - add))[:leftover]] += 1  # largest remainder
        add = np.minimum(add, counts - quota)
        quota += add
        remaining -= int(add.sum())
        open_cls &= quota < counts
        if add.sum() == 0:
            break
    return quota


def stage1(
    x: torch.Tensor,
    k: int,
    seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Cluster one video's frames of one class.

    Args:
        x: (n, d) normalized embeddings on the GPU.
        k: target cluster count; if >= n every frame is its own cluster.
        seed: rng seed.

    Returns:
        `(centroids, labels)`: (k', d) centroids on the GPU and (n,) cluster ids on the CPU.
    """
    n = len(x)
    if k >= n:
        return x, np.arange(n)
    labels, centroids = spherical_kmeans(x, k, seed=seed)
    labels = labels.cpu().numpy()
    counts = np.bincount(labels, minlength=k)
    keep = np.where(counts > 0)[0]
    remap = np.full(k, -1)
    remap[keep] = np.arange(len(keep))
    return centroids[torch.from_numpy(keep).to(x.device)], remap[labels]


def sample_member(
    labels: np.ndarray,
    cluster: int,
    rng: np.random.Generator,
) -> int:
    """Pick a random member (as an index into `labels`) of one cluster.

    Args:
        labels: (n,) cluster ids.
        cluster: cluster to sample from.
        rng: numpy generator.

    Returns:
        index into `labels`.
    """
    return int(rng.choice(np.flatnonzero(labels == cluster)))


def select_stage2(
    centroids: torch.Tensor,
    quota: int,
    picks_per_group: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster pooled centroids into groups and draw a balanced number per group.

    Args:
        centroids: (m, d) pooled stage-1 centroids (GPU).
        quota: number of centroids to choose.
        picks_per_group: target centroids drawn per group; sets the group count.
        seed: rng seed.

    Returns:
        `(chosen, group)`: chosen centroid indices, and the stage-2 group id of each chosen
        centroid.
    """
    m = len(centroids)
    rng = np.random.default_rng(seed)
    n_groups = max(min(quota // picks_per_group, m), 1)
    g_labels = spherical_kmeans(centroids, n_groups, seed=seed)[0].cpu().numpy()
    sizes = np.bincount(g_labels, minlength=n_groups)
    alloc = water_fill(sizes, quota)
    chosen, groups = [], []
    for g in np.flatnonzero(alloc):
        pick = rng.choice(np.flatnonzero(g_labels == g), alloc[g], replace=False)
        chosen.append(pick)
        groups.append(np.full(len(pick), g))
    return np.concatenate(chosen), np.concatenate(groups)


def main() -> None:
    """Run class-quota selection."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--quota-power', type=float, default=0.5)
    parser.add_argument('--fine-mult', type=float, default=3.0)
    parser.add_argument('--picks-per-group', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--videos', nargs='*', default=None, help='restrict to these stems')
    parser.add_argument('--strict', action='store_true',
                        help='error (instead of skipping) when a video has no embedding yet')
    parser.add_argument('--elig-dir', type=Path, default=EXTRACTION_DIR / 'eligibility_final')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = load_config().results_dir / 'embeddings' / EMBEDDING_NAME
    device = torch.device('cuda')

    # load tables, keep eligible frames with embeddings available
    tables = {}
    for p in sorted(args.elig_dir.glob('*.parquet')):
        if args.videos and p.stem not in args.videos:
            continue
        if not (emb_dir / f'{p.stem}.npy').exists():
            if args.strict:
                raise FileNotFoundError(f'no embeddings for {p.stem}')
            _logger.warning(f'{p.stem}: no embeddings yet, skipping')
            continue
        t = pd.read_parquet(p)
        tables[p.stem] = t[t['eligible']].reset_index(drop=True)
    if not tables:
        raise ValueError('no videos to process')
    videos = list(tables)

    counts = np.array([
        sum(int((t['predicted_pattern'] == c).sum()) for t in tables.values()) for c in CLASSES
    ])
    quota = compute_quotas(counts, args.budget, args.quota_power)
    _logger.info(f'{len(videos)} videos; class counts {dict(zip(CLASSES, counts))}')
    _logger.info(f'quotas {dict(zip(CLASSES, quota))}')

    # stage 1 over every (video, class) block
    blocks = {c: [] for c in CLASSES}  # per class: list of dicts
    for v_idx, stem in enumerate(tqdm(videos, desc='stage 1', unit='video')):
        t = tables[stem]
        emb = np.load(emb_dir / f'{stem}.npy', mmap_mode='r')
        for c_idx, cls in enumerate(CLASSES):
            rows = np.flatnonzero(t['predicted_pattern'].to_numpy() == cls)
            if len(rows) == 0:
                continue
            frames = t['frame_idx'].to_numpy()[rows]
            x = normalize(torch.from_numpy(np.asarray(emb[frames])).float().to(device))
            k = int(np.ceil(args.fine_mult * quota[c_idx] * len(rows) / counts[c_idx]))
            seed = int(np.random.SeedSequence([args.seed, v_idx, c_idx]).generate_state(1)[0])
            cents, labels = stage1(x, max(k, 1), seed)
            blocks[cls].append({
                'video': stem, 'frames': frames, 'rows': rows, 'labels': labels,
                'centroids': cents,
            })

    # stage 2 per class
    selected = []
    summary = {'quota': dict(zip(CLASSES, quota.tolist())), 'counts': dict(zip(CLASSES,
               counts.tolist())), 'args': {k: str(v) for k, v in vars(args).items()}}
    for c_idx, cls in enumerate(CLASSES):
        bl = blocks[cls]
        cents = torch.cat([b['centroids'] for b in bl])
        owner = np.concatenate([np.full(len(b['centroids']), i) for i, b in enumerate(bl)])
        local = np.concatenate([np.arange(len(b['centroids'])) for b in bl])
        n_pick = min(int(quota[c_idx]), len(cents))
        if n_pick < quota[c_idx]:
            _logger.warning(f'{cls}: only {len(cents)} centroids for quota {quota[c_idx]}')
        chosen, groups = select_stage2(cents, n_pick, args.picks_per_group,
                                       seed=args.seed + 1000 + c_idx)
        rng = np.random.default_rng(args.seed + 2000 + c_idx)
        for cid, grp in zip(chosen, groups):
            b = bl[owner[cid]]
            j = sample_member(b['labels'], int(local[cid]), rng)
            t = tables[b['video']]
            row = b['rows'][j]
            selected.append((b['video'], int(b['frames'][j]), cls,
                             float(t['confidence'].iloc[row]), int(grp)))
        _logger.info(f'{cls}: {len(cents)} stage-1 centroids -> {len(chosen)} selected')
        del cents

    manifest = pd.DataFrame(selected, columns=['video', 'frame_idx', 'predicted_pattern',
                                               'confidence', 'stage2_group'])
    assert not manifest.duplicated(['video', 'frame_idx']).any()
    manifest.to_parquet(args.out_dir / 'manifest.parquet')
    per_video = manifest.groupby('video').size()
    summary['selected'] = int(len(manifest))
    summary['per_class_selected'] = manifest['predicted_pattern'].value_counts().to_dict()
    summary['per_video'] = {k: float(v) for k, v in per_video.describe().items()}
    (args.out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    _logger.info(f'{len(manifest)} frames; per-video {summary["per_video"]}')


if __name__ == '__main__':
    main()
