"""Class-quota frame selection with per-video-first hierarchical k-means.

Two stages, each its own command so they are resumable and tunable independently:

1. `cuttle extract-kmeans-video` (`run_kmeans_video`): the `--budget` is split across the
   classifier's classes with shares proportional to `class_count ** quota_power` (0.5 = square
   root, 0 = uniform, 1 = proportional), never exceeding what a class has. Then, per video and
   per class, spherical k-means on the video's eligible frames' DINOv3 CLS embeddings with about
   `fine_mult` times the video's proportional share of the class quota as the cluster count
   (every frame is kept as its own cluster if the video has fewer). Saves centroids, per-frame
   cluster labels and confidences to `{video}.npz`.
2. `cuttle extract-kmeans-pooled` (`run_kmeans_pooled`): per class, pools every video's
   stage-1 centroids, clusters them into groups (about `quota / picks_per_group` groups), draws
   an equal number of centroids per group (water-filled, so small groups give everything) and
   takes one random member frame per chosen centroid. Per-video allocation is whatever the
   cluster structure gives; there is no per-video cap.

A random member per cluster is as good as the one nearest the centroid and far better than the
farthest (Vo et al. 2024), and is what the mighty-mouse pretraining pipeline uses.
"""

import json
import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cuttle_patterns.preprocessing.extract.common import save_parquet_atomic

_logger = logging.getLogger(__name__)

DEFAULT_QUOTA_POWER = 0.5
DEFAULT_FINE_MULT = 3.0
DEFAULT_PICKS_PER_GROUP = 10
CONFIG_FILENAME = 'config.yaml'
MANIFEST_FILENAME = 'manifest.parquet'
SUMMARY_FILENAME = 'summary.json'
N_ITER = 30


def normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows.

    Args:
        x: (n, d) tensor.

    Returns:
        row-normalized tensor.
    """
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def assign(x: torch.Tensor, centroids: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """Assign each row to the centroid with the highest dot product.

    Args:
        x: (n, d) tensor.
        centroids: (k, d) tensor.
        chunk: rows per matmul chunk.

    Returns:
        (n,) long tensor of cluster ids.
    """
    parts = [(x[i:i + chunk] @ centroids.T).argmax(dim=1) for i in range(0, len(x), chunk)]
    return torch.cat(parts)


def segment_sums(x: torch.Tensor, labels: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Sum rows of `x` per label in a fixed order (deterministic, unlike `index_add_` on GPU).

    Sorts rows by label and differences a float64 cumulative sum at the segment boundaries.

    Args:
        x: (n, d) tensor.
        labels: (n,) long tensor of cluster ids.
        counts: (k,) number of rows per cluster (`bincount(labels, minlength=k)`).

    Returns:
        (k, d) float32 tensor of per-cluster sums (zeros for empty clusters).
    """
    order = torch.argsort(labels, stable=True)
    csum = torch.cumsum(x[order].double(), dim=0)
    csum = torch.cat([torch.zeros(1, x.shape[1], dtype=csum.dtype, device=x.device), csum])
    ends = torch.cumsum(counts, dim=0)
    starts = ends - counts
    return (csum[ends] - csum[starts]).float()


def spherical_kmeans(
    x: torch.Tensor,
    k: int,
    n_iter: int = N_ITER,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm on the unit sphere with random-sample init (deterministic given seed).

    Args:
        x: (n, d) row-normalized tensor.
        k: number of clusters (must be <= n).
        n_iter: maximum number of iterations.
        seed: rng seed.

    Returns:
        `(labels, centroids)`: (n,) long labels and (k, d) normalized centroids. Empty clusters
        are re-seeded from random points each iteration.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n = len(x)
    centroids = x[torch.randperm(n, generator=gen)[:k].to(x.device)].clone()
    labels = assign(x, centroids)
    for _ in range(n_iter):
        counts = torch.bincount(labels, minlength=k)
        centroids = normalize(segment_sums(x, labels, counts))
        empty = counts == 0
        if empty.any():
            reseed = torch.randperm(n, generator=gen)[:int(empty.sum())].to(x.device)
            centroids[empty] = x[reseed]
        new_labels = assign(x, centroids)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    return labels, centroids


def water_fill(sizes: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` draws across groups as equally as possible, capped at group sizes.

    Args:
        sizes: (g,) number of available items per group.
        total: total number of draws (clipped to sizes.sum()).

    Returns:
        (g,) integer quota per group.
    """
    total = min(total, int(sizes.sum()))
    quota = np.zeros(len(sizes), dtype=int)
    open_groups = np.ones(len(sizes), dtype=bool)
    remaining = total
    while remaining > 0:
        share = max(remaining // int(open_groups.sum()), 1)
        for g in np.where(open_groups)[0]:
            give = min(share, sizes[g] - quota[g], remaining)
            quota[g] += give
            remaining -= give
            if quota[g] >= sizes[g]:
                open_groups[g] = False
            if remaining == 0:
                break
    return quota


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
    while remaining > 0 and open_cls.any():
        weights = np.where(open_cls, counts.astype(float) ** power, 0.0)
        ideal = weights / weights.sum() * remaining
        add = np.floor(ideal).astype(int)
        add[np.argsort(-(ideal - add))[:remaining - int(add.sum())]] += 1  # largest remainder
        add = np.minimum(add, counts - quota)
        if add.sum() == 0:
            break
        quota += add
        remaining -= int(add.sum())
        open_cls &= quota < counts
    return quota


def cluster_block(x: torch.Tensor, k: int, seed: int) -> tuple[torch.Tensor, np.ndarray]:
    """Cluster one video's frames of one class (stage 1).

    Args:
        x: (n, d) normalized embeddings.
        k: target cluster count; if >= n every frame is its own cluster.
        seed: rng seed.

    Returns:
        `(centroids, labels)`: (k', d) centroids (empty clusters dropped) and (n,) cluster ids
        in [0, k') as a numpy array.
    """
    n = len(x)
    if k >= n:
        return x, np.arange(n)
    labels, centroids = spherical_kmeans(x, k, seed=seed)
    labels = labels.cpu().numpy()
    keep = np.flatnonzero(np.bincount(labels, minlength=k) > 0)
    remap = np.full(k, -1)
    remap[keep] = np.arange(len(keep))
    return centroids[torch.from_numpy(keep).to(x.device)], remap[labels]


def _block_seed(seed: int, stem: str, class_idx: int) -> int:
    """Stable per-(video, class) seed that doesn't change when other videos are added.

    Args:
        seed: base seed.
        stem: video stem.
        class_idx: class index.

    Returns:
        an integer seed.
    """
    return int(np.random.SeedSequence([seed, zlib.crc32(stem.encode()), class_idx])
               .generate_state(1)[0])


def load_eligible_labels(elig_path: Path, clf_path: Path) -> pd.DataFrame:
    """Load a video's eligible frames with their classifier labels.

    Args:
        elig_path: eligibility table (`cuttle extract-eligibility`).
        clf_path: per-frame classifier predictions (`cuttle extract-classify`).

    Returns:
        rows for eligible frames with `frame_idx`, `predicted_pattern` and `confidence`.

    Raises:
        ValueError: if the two tables differ in length.
    """
    elig = pd.read_parquet(elig_path)
    clf = pd.read_parquet(clf_path)
    if len(elig) != len(clf):
        raise ValueError(
            f'{elig_path.stem}: eligibility has {len(elig)} frames, classifier {len(clf)}'
        )
    out = clf.loc[elig['eligible'].to_numpy(), ['frame_idx', 'predicted_pattern', 'confidence']]
    return out.reset_index(drop=True)


def run_kmeans_video(
    stems: list[str],
    elig_dir: Path,
    clf_dir: Path,
    emb_dir: Path,
    out_dir: Path,
    classes: list[str],
    budget: int,
    classifier_name: str,
    embedding_name: str,
    quota_power: float = DEFAULT_QUOTA_POWER,
    fine_mult: float = DEFAULT_FINE_MULT,
    seed: int = 0,
    overwrite: bool = False,
    device: torch.device | None = None,
) -> dict:
    """Stage 1: cluster every video's eligible frames within each class and save the results.

    Class counts (hence quotas and cluster counts) come from all `stems` together, so pass the
    full set of videos in one call. Existing per-video files are skipped unless `overwrite` is
    set, but only if they were made with the same settings and quotas; otherwise this raises,
    since stage 1 results depend on the quotas.

    Args:
        stems: video stems to include (each needs eligibility, classifier and embedding files).
        elig_dir: directory of eligibility tables.
        clf_dir: directory of classifier prediction tables.
        emb_dir: directory of per-video DINOv3 `.npy` files (row i is frame i).
        out_dir: output directory (`{video}.npz`, `config.yaml`).
        classes: class names, in classifier output order.
        budget: total number of frames to select across all videos.
        classifier_name: name of the classifier output directory (recorded for later stages).
        embedding_name: name of the embedding output directory (recorded for later stages).
        quota_power: exponent on class counts for the quotas (0.5 = square root).
        fine_mult: stage-1 clusters per class per video = fine_mult x the video's share of the
            class quota.
        seed: base rng seed.
        overwrite: recompute existing per-video files.
        device: torch device; defaults to CUDA if available.

    Returns:
        the config dict written to `config.yaml` (settings, class counts and quotas).

    Raises:
        ValueError: if existing outputs were made with different settings and `overwrite` is
            not set.
    """
    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {s: load_eligible_labels(elig_dir / f'{s}.parquet', clf_dir / f'{s}.parquet')
              for s in stems}
    counts = np.array([
        sum(int((t['predicted_pattern'] == c).sum()) for t in tables.values()) for c in classes
    ])
    quotas = compute_quotas(counts, budget, quota_power)
    config = {
        'classes': list(classes), 'classifier_name': classifier_name,
        'embedding_name': embedding_name, 'budget': budget, 'quota_power': quota_power,
        'fine_mult': fine_mult, 'seed': seed, 'n_videos': len(stems),
        'counts': counts.tolist(), 'quotas': quotas.tolist(),
    }
    config_path = out_dir / CONFIG_FILENAME
    if config_path.exists() and not overwrite:
        old = yaml.safe_load(config_path.read_text())
        if old != config:
            raise ValueError(
                f'{config_path} was made with different settings or videos; stage 1 depends on '
                f'the class quotas, so rerun with --overwrite (or choose a new --name)'
            )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    _logger.info(f'class counts {dict(zip(classes, counts.tolist(), strict=True))}')
    _logger.info(f'quotas {dict(zip(classes, quotas.tolist(), strict=True))}')

    for stem in tqdm(stems, desc='kmeans-video', unit='video'):
        out_path = out_dir / f'{stem}.npz'
        if out_path.exists() and not overwrite:
            continue
        table = tables[stem]
        emb = np.load(emb_dir / f'{stem}.npy', mmap_mode='r')
        arrays = {}
        for c_idx, cls in enumerate(classes):
            sub = table[table['predicted_pattern'] == cls]
            frames = sub['frame_idx'].to_numpy()
            arrays[f'frames_{c_idx}'] = frames.astype(np.int32)
            arrays[f'conf_{c_idx}'] = sub['confidence'].to_numpy().astype(np.float16)
            if len(frames) == 0:
                arrays[f'labels_{c_idx}'] = np.zeros(0, dtype=np.int32)
                arrays[f'centroids_{c_idx}'] = np.zeros((0, emb.shape[1]), dtype=np.float16)
                continue
            x = normalize(torch.from_numpy(np.asarray(emb[frames])).float().to(device))
            k = max(int(np.ceil(fine_mult * quotas[c_idx] * len(frames) / counts[c_idx])), 1)
            centroids, labels = cluster_block(x, k, _block_seed(seed, stem, c_idx))
            arrays[f'labels_{c_idx}'] = labels.astype(np.int32)
            arrays[f'centroids_{c_idx}'] = centroids.cpu().numpy().astype(np.float16)
        tmp = out_path.with_name(f'{out_path.name}.tmp')
        with open(tmp, 'wb') as f:
            np.savez(f, **arrays)
        tmp.replace(out_path)
    return config


def select_pooled(
    centroids: torch.Tensor,
    quota: int,
    picks_per_group: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2 core: group pooled centroids and draw a balanced number from each group.

    Args:
        centroids: (m, d) pooled stage-1 centroids of one class.
        quota: number of centroids to choose (clipped to m).
        picks_per_group: target centroids drawn per group; sets the group count.
        seed: rng seed.

    Returns:
        `(chosen, group)`: chosen centroid indices, and the group id of each chosen centroid.
    """
    m = len(centroids)
    quota = min(quota, m)
    rng = np.random.default_rng(seed)
    n_groups = max(min(quota // picks_per_group, m), 1)
    centroids = normalize(centroids)
    g_labels = spherical_kmeans(centroids, n_groups, seed=seed)[0].cpu().numpy()
    alloc = water_fill(np.bincount(g_labels, minlength=n_groups), quota)
    chosen, groups = [], []
    for g in np.flatnonzero(alloc):
        pick = rng.choice(np.flatnonzero(g_labels == g), alloc[g], replace=False)
        chosen.append(pick)
        groups.append(np.full(len(pick), g))
    return np.concatenate(chosen), np.concatenate(groups)


def run_kmeans_pooled(
    kmeans_dir: Path,
    out_dir: Path,
    picks_per_group: int = DEFAULT_PICKS_PER_GROUP,
    seed: int = 0,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Stage 2: pool stage-1 centroids per class, select balanced centroids, pick members.

    Args:
        kmeans_dir: stage-1 output directory (`{video}.npz`, `config.yaml`).
        out_dir: output directory (`manifest.parquet`, `summary.json`).
        picks_per_group: target centroids drawn per stage-2 group.
        seed: rng seed.
        device: torch device; defaults to CUDA if available.

    Returns:
        the selection manifest: one row per selected frame with `video`, `frame_idx`,
        `predicted_pattern`, `confidence` and `stage2_group`.

    Raises:
        FileNotFoundError: if `kmeans_dir` has no stage-1 config or per-video files.
    """
    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    config_path = kmeans_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f'no stage-1 config at {config_path}; run extract-kmeans-video')
    config = yaml.safe_load(config_path.read_text())
    classes, quotas = config['classes'], config['quotas']
    paths = sorted(kmeans_dir.glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'no stage-1 files in {kmeans_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {p.stem: dict(np.load(p)) for p in paths}
    rows = []
    for c_idx, cls in enumerate(classes):
        owners = [s for s in data if len(data[s][f'centroids_{c_idx}'])]
        cents = torch.cat([
            torch.from_numpy(data[s][f'centroids_{c_idx}']).float() for s in owners
        ]).to(device)
        owner = np.concatenate([np.full(len(data[s][f'centroids_{c_idx}']), i)
                                for i, s in enumerate(owners)])
        local = np.concatenate([np.arange(len(data[s][f'centroids_{c_idx}'])) for s in owners])
        if len(cents) < quotas[c_idx]:
            _logger.warning(f'{cls}: only {len(cents)} centroids for quota {quotas[c_idx]}')
        chosen, groups = select_pooled(cents, quotas[c_idx], picks_per_group, seed + 1000 + c_idx)
        rng = np.random.default_rng(seed + 2000 + c_idx)
        for cid, group in zip(chosen, groups, strict=True):
            stem = owners[owner[cid]]
            members = np.flatnonzero(data[stem][f'labels_{c_idx}'] == local[cid])
            j = int(rng.choice(members))
            rows.append((stem, int(data[stem][f'frames_{c_idx}'][j]), cls,
                         float(data[stem][f'conf_{c_idx}'][j]), int(group)))
        _logger.info(f'{cls}: {len(cents)} centroids -> {len(chosen)} selected')

    manifest = pd.DataFrame(
        rows, columns=['video', 'frame_idx', 'predicted_pattern', 'confidence', 'stage2_group'],
    )
    if manifest.duplicated(['video', 'frame_idx']).any():
        raise RuntimeError('selection produced duplicate frames')
    save_parquet_atomic(out_dir / MANIFEST_FILENAME, manifest)
    per_video = manifest.groupby('video').size()
    (out_dir / SUMMARY_FILENAME).write_text(json.dumps({
        'quotas': dict(zip(classes, quotas, strict=True)),
        'selected': int(len(manifest)),
        'per_class_selected': manifest['predicted_pattern'].value_counts().to_dict(),
        'per_video': {k: float(v) for k, v in per_video.describe().items()},
        'picks_per_group': picks_per_group,
        'seed': seed,
        'stage1': config,
    }, indent=1))
    return manifest
