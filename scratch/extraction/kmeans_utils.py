"""GPU spherical k-means and frame-selection strategies (flat and two-level balanced).

Selection follows the recipe in mighty-mouse's `pretrain_frame_selection.md`: one random
member per chosen cluster, and for the two-level variant, fine clusters are grouped on their
centroids and an equal number of fine clusters is drawn from each group (water-filled, so
small groups give everything and the surplus goes to larger groups).
"""

import numpy as np
import torch


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
    out = [(x[i:i + chunk] @ centroids.T).argmax(dim=1) for i in range(0, len(x), chunk)]
    return torch.cat(out)


def spherical_kmeans(
    x: torch.Tensor,
    k: int,
    n_iter: int = 30,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's algorithm on the unit sphere with random-sample init.

    Args:
        x: (n, d) row-normalized tensor.
        k: number of clusters (must be <= n).
        n_iter: number of iterations.
        seed: rng seed.

    Returns:
        (labels, centroids): (n,) long labels and (k, d) normalized centroids. Empty clusters
        are re-seeded from random points each iteration.
    """
    gen = torch.Generator(device='cpu').manual_seed(seed)
    n = len(x)
    centroids = x[torch.randperm(n, generator=gen)[:k].to(x.device)].clone()
    labels = assign(x, centroids)
    for _ in range(n_iter):
        sums = torch.zeros_like(centroids).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k)
        empty = counts == 0
        centroids = normalize(sums)
        if empty.any():
            reseed = torch.randperm(n, generator=gen)[:int(empty.sum())].to(x.device)
            centroids[empty] = x[reseed]
        new_labels = assign(x, centroids)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    return labels, centroids


def random_member_per_cluster(
    labels: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick one random member from each listed cluster.

    Args:
        labels: (n,) cluster id per point.
        clusters: cluster ids to sample from; empty clusters are skipped.
        rng: numpy generator.

    Returns:
        indices into the point array, one per non-empty listed cluster.
    """
    order = np.argsort(labels, kind='stable')
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, clusters, side='left')
    ends = np.searchsorted(sorted_labels, clusters, side='right')
    keep = ends > starts
    starts, ends = starts[keep], ends[keep]
    picks = starts + (rng.random(len(starts)) * (ends - starts)).astype(int)
    return order[picks]


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
        n_open = int(open_groups.sum())
        share = max(remaining // n_open, 1)
        for g in np.where(open_groups)[0]:
            give = min(share, sizes[g] - quota[g], remaining)
            quota[g] += give
            remaining -= give
            if quota[g] >= sizes[g]:
                open_groups[g] = False
            if remaining == 0:
                break
    return quota


def select_flat(x: torch.Tensor, n_target: int, seed: int) -> np.ndarray:
    """Flat k-means with `n_target` clusters, one random member per cluster.

    Args:
        x: (n, d) normalized embeddings on the GPU.
        n_target: number of frames to select.
        seed: rng seed.

    Returns:
        sorted array of selected row indices.
    """
    n = len(x)
    if n <= n_target:
        return np.arange(n)
    labels, _ = spherical_kmeans(x, n_target, seed=seed)
    rng = np.random.default_rng(seed)
    picks = random_member_per_cluster(labels.cpu().numpy(), np.arange(n_target), rng)
    return np.sort(_top_up(picks, n, n_target, rng))


def select_two_level(
    x: torch.Tensor,
    n_target: int,
    seed: int,
    fine_mult: int = 3,
    group_div: int = 10,
) -> np.ndarray:
    """Two-level k-means with balanced sampling, one random member per chosen fine cluster.

    Args:
        x: (n, d) normalized embeddings on the GPU.
        n_target: number of frames to select.
        seed: rng seed.
        fine_mult: number of fine clusters = fine_mult * n_target.
        group_div: number of groups = n_target // group_div.

    Returns:
        sorted array of selected row indices.
    """
    n = len(x)
    if n <= n_target:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    k_fine = min(n, fine_mult * n_target)
    labels, centroids = spherical_kmeans(x, k_fine, seed=seed)
    labels = labels.cpu().numpy()
    # drop empty fine clusters before grouping
    counts = np.bincount(labels, minlength=k_fine)
    nonempty = np.where(counts > 0)[0]
    n_groups = max(min(n_target // group_div, len(nonempty)), 1)
    g_labels, _ = spherical_kmeans(centroids[torch.from_numpy(nonempty).to(x.device)],
                                   n_groups, seed=seed)
    g_labels = g_labels.cpu().numpy()
    sizes = np.bincount(g_labels, minlength=n_groups)
    quota = water_fill(sizes, n_target)
    chosen = []
    for g in range(n_groups):
        members = nonempty[g_labels == g]
        if quota[g] > 0:
            chosen.append(rng.choice(members, quota[g], replace=False))
    chosen = np.concatenate(chosen)
    picks = random_member_per_cluster(labels, chosen, rng)
    return np.sort(_top_up(picks, n, n_target, rng))


def _top_up(picks: np.ndarray, n: int, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Add random unselected points if fewer than `n_target` were picked.

    Args:
        picks: selected indices.
        n: total number of points.
        n_target: desired count.
        rng: numpy generator.

    Returns:
        indices with length min(n_target, n).
    """
    if len(picks) >= n_target:
        return picks
    rest = np.setdiff1d(np.arange(n), picks)
    return np.concatenate([picks, rng.choice(rest, n_target - len(picks), replace=False)])
