"""Diagnostics for a `select_by_class.py` manifest.

Status: exploratory; companion to `select_by_class.py`.

Everything is computed over the videos present in the manifest, against the eligible frames of
those videos (`eligibility_final/`) and their DINOv3 CLS embeddings. For each class, the
selection is compared with a random baseline of the same size drawn uniformly from that class's
eligible frames. Reports (printed, and written to `report.json` / `report.md`):

- class counts vs quotas (from `summary.json`) and mean classifier confidence, selected vs
  eligible, plus the share with confidence < 0.6;
- per-video allocation (min / p10 / median / p90 / max, videos under 500 / over 2000, most and
  least represented videos) and per-class concentration (top-5 video share, contributing videos);
- within-class coverage: cosine distance from sampled eligible frames to the nearest selected
  frame of the same class (mean, p95, p99.9), pooled across videos;
- densest-state and rare-state ratios on a per-class k-means partition (selected share /
  eligible share of the largest cluster and of the 10 smallest);
- temporal redundancy: share of frames with another selected frame from the same video within
  10 / 100 frames, selected vs random baseline (random matters: at ~1k picks per video the
  100-frame figure is high even for random);
- contact sheets per class (aligned rectangle frames): 50 random picks and the 50 most isolated
  picks (largest mean cosine distance to their 5 nearest selected neighbors).

Usage:
    python evaluate_selection.py --selection-dir <dir with manifest.parquet> [--sample 200000]
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cuttle_patterns.config import load_config
from jump_contact_sheets import read_frames
from kmeans_utils import assign, normalize, spherical_kmeans

_logger = logging.getLogger(__name__)

EXTRACTION_DIR = Path('/media/mattw/CUTTLE/results/.extraction')
CLASSES = ['Dark aggression', 'Leopard', 'Pale aggression', 'White', 'Yellow-brown']
EMBEDDING_NAME = 'dinov3_vits16'
N_COARSE = 50
N_RARE = 10
LOW_CONF = 0.6
N_SHEET = 50
TILE_W, TILE_H = 256, 128
N_COLS = 10
CHUNK = 20000


def load_pools(
    videos: list[str],
    elig_dir: Path,
    emb_dir: Path,
) -> dict[str, dict]:
    """Load eligible frames and embeddings of the given videos, grouped by class.

    Args:
        videos: video stems.
        elig_dir: directory of final eligibility tables.
        emb_dir: directory of per-video DINOv3 `.npy` files.

    Returns:
        per class: `video` (int video index per frame), `frame` (frame index), `conf`, and `emb`
        (float16 CPU tensor, row-aligned).
    """
    parts = {c: {'video': [], 'frame': [], 'conf': [], 'emb': []} for c in CLASSES}
    for v_idx, stem in enumerate(tqdm(videos, desc='loading', unit='video')):
        t = pd.read_parquet(elig_dir / f'{stem}.parquet')
        t = t[t['eligible']]
        emb = np.load(emb_dir / f'{stem}.npy', mmap_mode='r')
        for cls in CLASSES:
            sub = t[t['predicted_pattern'] == cls]
            if len(sub) == 0:
                continue
            frames = sub['frame_idx'].to_numpy()
            parts[cls]['video'].append(np.full(len(frames), v_idx, dtype=np.int32))
            parts[cls]['frame'].append(frames.astype(np.int64))
            parts[cls]['conf'].append(sub['confidence'].to_numpy())
            parts[cls]['emb'].append(torch.from_numpy(np.asarray(emb[frames])))
    return {
        c: {k: (torch.cat(v) if k == 'emb' else np.concatenate(v)) for k, v in p.items()}
        for c, p in parts.items()
    }


def nearest_distance(
    queries: torch.Tensor,
    refs: torch.Tensor,
    exclude_self: bool = False,
    k: int = 1,
) -> torch.Tensor:
    """Mean cosine distance from each query to its k nearest references.

    Args:
        queries: (q, d) normalized embeddings on the GPU.
        refs: (r, d) normalized embeddings on the GPU.
        exclude_self: if True, queries are rows of refs (same order) and each query's own row is
            ignored.
        k: number of neighbors to average over.

    Returns:
        (q,) tensor of distances.
    """
    out = []
    for i in range(0, len(queries), CHUNK):
        sim = queries[i:i + CHUNK] @ refs.T
        if exclude_self:
            idx = torch.arange(i, min(i + CHUNK, len(queries)), device=sim.device)
            sim[torch.arange(len(idx)), idx] = -2
        out.append(1 - sim.topk(k, dim=1).values.mean(dim=1))
    return torch.cat(out)


def pool_metrics(
    sel_emb: torch.Tensor,
    sample_emb: torch.Tensor,
    centroids: torch.Tensor,
    sample_sizes: np.ndarray,
    rare: np.ndarray,
) -> dict:
    """Coverage and state-share metrics for one selected set against a class sample.

    Args:
        sel_emb: (s, d) normalized embeddings of the selected frames (GPU).
        sample_emb: (n, d) normalized embeddings of the eligible sample (GPU).
        centroids: (N_COARSE, d) coarse-state centroids.
        sample_sizes: sample frames per coarse state.
        rare: indices of the rarest coarse states.

    Returns:
        dict with coverage and state-ratio metrics.
    """
    dist = nearest_distance(sample_emb, sel_emb).cpu().numpy()
    sel_counts = np.bincount(assign(sel_emb, centroids).cpu().numpy(), minlength=N_COARSE)
    dens = int(np.argmax(sample_sizes))
    n, s = sample_sizes.sum(), sel_counts.sum()
    return {
        'cov_mean': float(dist.mean()),
        'cov_p95': float(np.percentile(dist, 95)),
        'cov_p999': float(np.percentile(dist, 99.9)),
        'dens_ratio': float((sel_counts[dens] / s) / (sample_sizes[dens] / n)),
        'rare_ratio': float((sel_counts[rare].sum() / s) / (sample_sizes[rare].sum() / n)),
    }


def near_fraction(df: pd.DataFrame, gap: int) -> float:
    """Share of frames with another frame from the same video within `gap` frames.

    Args:
        df: rows with `video` and `frame_idx`.
        gap: maximum frame distance.

    Returns:
        fraction of rows that have such a neighbor.
    """
    hit = 0
    for _, g in df.groupby('video'):
        f = np.sort(g['frame_idx'].to_numpy())
        near = np.zeros(len(f), dtype=bool)
        near[1:] |= np.diff(f) <= gap
        near[:-1] |= np.diff(f) <= gap
        hit += int(near.sum())
    return hit / len(df)


def make_sheet(
    picks: pd.DataFrame,
    rect_dir: Path,
    out_path: Path,
    extra: np.ndarray | None = None,
) -> None:
    """Write a contact sheet of selected frames from the aligned rectangle videos.

    Args:
        picks: rows with `video`, `frame_idx` and `confidence`.
        rect_dir: directory of rectangle videos.
        out_path: output jpg.
        extra: optional per-row number shown in the label (e.g. isolation).
    """
    tiles = []
    for (video, group) in picks.groupby('video', sort=False):
        frames = read_frames(rect_dir / f'{video}.mp4', group['frame_idx'].tolist())
        for pos, (_, r) in enumerate(group.iterrows()):
            img = cv2.resize(frames[int(r['frame_idx'])], (TILE_W, TILE_H),
                             interpolation=cv2.INTER_AREA)
            label = f'{video.split("_Crop")[0][3:]} {int(r["frame_idx"])} c={r["confidence"]:.2f}'
            cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append((r.name, img))
    tiles = [img for _, img in sorted(tiles, key=lambda x: picks.index.get_loc(x[0]))]
    tiles += [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (-len(tiles) % N_COLS)
    rows = [np.hstack(tiles[i:i + N_COLS]) for i in range(0, len(tiles), N_COLS)]
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    """Evaluate a selection manifest."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--selection-dir', type=Path, required=True)
    parser.add_argument('--elig-dir', type=Path, default=EXTRACTION_DIR / 'eligibility_final')
    parser.add_argument('--sample', type=int, default=200_000,
                        help='eligible frames per class used for coverage/state metrics')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-sheets', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = load_config()
    emb_dir = config.results_dir / 'embeddings' / EMBEDDING_NAME
    rect_dir = config.results_dir / 'rectangles'
    manifest = pd.read_parquet(args.selection_dir / 'manifest.parquet')
    summary = json.loads((args.selection_dir / 'summary.json').read_text())
    videos = sorted(manifest['video'].unique())
    video_idx = {v: i for i, v in enumerate(videos)}
    device = torch.device('cuda')
    rng = np.random.default_rng(args.seed)
    pools = load_pools(videos, args.elig_dir, emb_dir)
    report = {'n_videos': len(videos), 'classes': {}}

    # per-video allocation and class-level counts
    per_video = manifest.groupby('video').size().sort_values()
    report['per_video'] = {
        'min': int(per_video.min()), 'p10': float(np.percentile(per_video, 10)),
        'median': float(per_video.median()), 'p90': float(np.percentile(per_video, 90)),
        'max': int(per_video.max()), 'under_500': int((per_video < 500).sum()),
        'over_2000': int((per_video > 2000).sum()),
        'lowest': per_video.head(3).to_dict(), 'highest': per_video.tail(3).to_dict(),
    }

    for cls in CLASSES:
        pool = pools[cls]
        sel = manifest[manifest['predicted_pattern'] == cls].reset_index(drop=True)
        n_pool = len(pool['frame'])
        _logger.info(f'{cls}: {len(sel)} selected of {n_pool} eligible')
        # map selected rows to pool rows
        keys = pool['video'].astype(np.int64) * 10**7 + pool['frame']
        order = np.argsort(keys)
        sel_keys = sel['video'].map(video_idx).to_numpy().astype(np.int64) * 10**7 \
            + sel['frame_idx'].to_numpy()
        sel_rows = order[np.searchsorted(keys[order], sel_keys)]
        assert (keys[sel_rows] == sel_keys).all()

        emb = normalize(pool['emb'].float().to(device))
        sample = np.sort(rng.choice(n_pool, min(args.sample, n_pool), replace=False))
        sample_emb = emb[torch.from_numpy(sample).to(device)]
        centroids = spherical_kmeans(sample_emb, min(N_COARSE, len(sample)), seed=args.seed)[1]
        sizes = np.bincount(assign(sample_emb, centroids).cpu().numpy(), minlength=N_COARSE)
        rare = np.argsort(sizes)[:N_RARE]

        entry = {
            'quota': summary['quota'][cls], 'selected': int(len(sel)), 'eligible': int(n_pool),
            'conf_selected': float(sel['confidence'].mean()),
            'conf_eligible': float(pool['conf'].mean()),
            'lowconf_selected': float((sel['confidence'] < LOW_CONF).mean()),
            'lowconf_eligible': float((pool['conf'] < LOW_CONF).mean()),
        }
        counts = sel.groupby('video').size()
        entry['top5_video_share'] = float(counts.sort_values().tail(5).sum() / len(sel))
        entry['videos_contributing'] = int(len(counts))
        entry['videos_with_class'] = int(len(np.unique(pool['video'])))

        sel_emb = emb[torch.from_numpy(sel_rows).to(device)]
        entry['selected_metrics'] = pool_metrics(sel_emb, sample_emb, centroids, sizes, rare)
        base_rows = rng.choice(n_pool, len(sel), replace=False)
        base_emb = emb[torch.from_numpy(base_rows).to(device)]
        entry['random_metrics'] = pool_metrics(base_emb, sample_emb, centroids, sizes, rare)

        # temporal redundancy within a video, selected vs the random baseline
        base = pd.DataFrame({'video': pool['video'][base_rows], 'frame_idx': pool['frame'][base_rows]})
        sel_v = pd.DataFrame({'video': sel['video'].map(video_idx), 'frame_idx': sel['frame_idx']})
        entry['temporal_redundancy'] = {
            f'{name}_within_{gap}': near_fraction(df, gap)
            for gap in (10, 100) for name, df in (('sel', sel_v), ('random', base))
        }
        report['classes'][cls] = entry

        if not args.no_sheets:
            sheet_dir = args.selection_dir / 'eval_sheets'
            sheet_dir.mkdir(exist_ok=True)
            slug = cls.replace(' ', '_')
            pick = sel.sample(min(N_SHEET, len(sel)), random_state=args.seed)
            make_sheet(pick, rect_dir, sheet_dir / f'{slug}_random.jpg')
            if len(sel) > 6:
                iso = nearest_distance(sel_emb, sel_emb, exclude_self=True, k=5).cpu().numpy()
                top = np.argsort(-iso)[:N_SHEET]
                make_sheet(sel.iloc[top], rect_dir, sheet_dir / f'{slug}_isolated.jpg')
        del emb, sample_emb, sel_emb, base_emb
        torch.cuda.empty_cache()

    (args.selection_dir / 'report.json').write_text(json.dumps(report, indent=1))
    lines = [f'videos: {len(videos)}', f'per video: {report["per_video"]}', '']
    hdr = ('class', 'quota', 'sel', 'conf s/e', 'lowconf s/e', 'top5vid', 'vids',
           'cov p999 s/r', 'dens s', 'rare s/r', 'near10 s/r')
    lines.append(' | '.join(hdr))
    for cls, e in report['classes'].items():
        s, r = e['selected_metrics'], e['random_metrics']
        lines.append(' | '.join([
            cls, str(e['quota']), str(e['selected']),
            f'{e["conf_selected"]:.2f}/{e["conf_eligible"]:.2f}',
            f'{e["lowconf_selected"]:.2f}/{e["lowconf_eligible"]:.2f}',
            f'{e["top5_video_share"]:.2f}', f'{e["videos_contributing"]}/{e["videos_with_class"]}',
            f'{s["cov_p999"]:.3f}/{r["cov_p999"]:.3f}', f'{s["dens_ratio"]:.2f}',
            f'{s["rare_ratio"]:.2f}/{r["rare_ratio"]:.2f}',
            f'{e["temporal_redundancy"]["sel_within_10"]:.2f}/{e["temporal_redundancy"]["random_within_10"]:.2f}',
        ]))
    (args.selection_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
