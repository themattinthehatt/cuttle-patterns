"""Diagnostics for a class-quota selection (`cuttle extract-report`).

Everything is computed over the videos present in the selection manifest, against the eligible
frames of those videos and their DINOv3 CLS embeddings. For each class, the selection is compared
with a random baseline of the same size drawn uniformly from that class's eligible frames.
Written to `report.json` / `report.md`, and printed:

- class counts vs quotas and mean classifier confidence, selected vs eligible, plus the share
  with confidence < 0.6;
- per-video allocation (min / p10 / median / p90 / max, videos under 500 / over 2000, most and
  least represented videos) and per-class concentration (top-5 video share, contributing
  videos);
- within-class coverage: cosine distance from sampled eligible frames to the nearest selected
  frame of the same class (mean, p95, p99.9), pooled across videos;
- densest-state and rare-state ratios on a per-class k-means partition (selected share /
  eligible share of the largest cluster and of the 10 smallest);
- temporal redundancy: share of frames with another selected frame from the same video within
  10 / 100 frames, selected vs random baseline;
- contact sheets per class (aligned rectangle frames): 50 random picks and the 50 most
  isolated picks (largest mean cosine distance to their 5 nearest selected neighbors).
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cuttle_patterns.preprocessing.extract.kmeans import (
    MANIFEST_FILENAME,
    SUMMARY_FILENAME,
    assign,
    load_eligible_labels,
    normalize,
    spherical_kmeans,
)

_logger = logging.getLogger(__name__)

N_COARSE = 50
N_RARE = 10
LOW_CONFIDENCE = 0.6
N_SHEET = 50
TILE_W, TILE_H = 256, 128
N_COLS = 10
CHUNK = 20000


def load_pools(
    stems: list[str],
    elig_dir: Path,
    clf_dir: Path,
    emb_dir: Path,
    classes: list[str],
) -> dict[str, dict]:
    """Load eligible frames and embeddings of the given videos, grouped by class.

    Args:
        stems: video stems.
        elig_dir: directory of eligibility tables.
        clf_dir: directory of classifier prediction tables.
        emb_dir: directory of per-video DINOv3 `.npy` files.
        classes: class names.

    Returns:
        per class: `video` (int index into `stems` per frame), `frame` (frame index), `conf`, and
        `emb` (float16 CPU tensor), all row-aligned.
    """
    parts = {c: {'video': [], 'frame': [], 'conf': [], 'emb': []} for c in classes}
    for v_idx, stem in enumerate(tqdm(stems, desc='loading', unit='video')):
        table = load_eligible_labels(elig_dir / f'{stem}.parquet', clf_dir / f'{stem}.parquet')
        emb = np.load(emb_dir / f'{stem}.npy', mmap_mode='r')
        for cls in classes:
            sub = table[table['predicted_pattern'] == cls]
            if len(sub) == 0:
                continue
            frames = sub['frame_idx'].to_numpy()
            parts[cls]['video'].append(np.full(len(frames), v_idx, dtype=np.int32))
            parts[cls]['frame'].append(frames.astype(np.int64))
            parts[cls]['conf'].append(sub['confidence'].to_numpy())
            parts[cls]['emb'].append(torch.from_numpy(np.asarray(emb[frames])))
    return {
        c: {k: (torch.cat(v) if k == 'emb' else np.concatenate(v)) for k, v in p.items()}
        for c, p in parts.items() if p['frame']
    }


def nearest_distance(
    queries: torch.Tensor,
    refs: torch.Tensor,
    exclude_self: bool = False,
    k: int = 1,
) -> torch.Tensor:
    """Mean cosine distance from each query to its k nearest references.

    Args:
        queries: (q, d) normalized embeddings.
        refs: (r, d) normalized embeddings.
        exclude_self: if True, queries are the rows of refs (same order) and each query's own
            row is ignored.
        k: number of neighbors to average over.

    Returns:
        (q,) tensor of distances.
    """
    out = []
    for i in range(0, len(queries), CHUNK):
        sim = queries[i:i + CHUNK] @ refs.T
        if exclude_self:
            rows = torch.arange(i, min(i + CHUNK, len(queries)), device=sim.device)
            sim[torch.arange(len(rows)), rows] = -2
        out.append(1 - sim.topk(min(k, sim.shape[1]), dim=1).values.mean(dim=1))
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
        sel_emb: (s, d) normalized embeddings of the selected frames.
        sample_emb: (n, d) normalized embeddings of the eligible sample.
        centroids: (n_states, d) coarse-state centroids.
        sample_sizes: sample frames per coarse state.
        rare: indices of the rarest coarse states.

    Returns:
        dict with `cov_mean`, `cov_p95`, `cov_p999` (nearest-selected cosine distance over the
        sample), and `dens_ratio` / `rare_ratio` (selected share over eligible share of the
        densest state / the rare states).
    """
    dist = nearest_distance(sample_emb, sel_emb).cpu().numpy()
    sel_counts = np.bincount(assign(sel_emb, centroids).cpu().numpy(), minlength=len(centroids))
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
    for _, group in df.groupby('video'):
        frames = np.sort(group['frame_idx'].to_numpy())
        near = np.zeros(len(frames), dtype=bool)
        near[1:] |= np.diff(frames) <= gap
        near[:-1] |= np.diff(frames) <= gap
        hit += int(near.sum())
    return hit / max(len(df), 1)


def make_sheet(picks: pd.DataFrame, video_dir: Path, out_path: Path) -> None:
    """Write a contact sheet of selected frames from the aligned rectangle videos.

    Args:
        picks: rows with `video`, `frame_idx` and `confidence`; shown in this order.
        video_dir: directory of rectangle videos.
        out_path: output jpg.
    """
    frames = {}
    for video, group in picks.groupby('video'):
        cap = cv2.VideoCapture(str(video_dir / f'{video}.mp4'))
        for frame_idx in group['frame_idx']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                frames[(video, int(frame_idx))] = frame
        cap.release()
    tiles = []
    for _, row in picks.iterrows():
        key = (row['video'], int(row['frame_idx']))
        if key not in frames:
            continue
        img = cv2.resize(frames[key], (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
        label = f'{row["video"].split("_Crop")[0][3:]} {key[1]} c={row["confidence"]:.2f}'
        cv2.putText(img, label, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(img)
    tiles += [np.zeros((TILE_H, TILE_W, 3), np.uint8)] * (-len(tiles) % N_COLS)
    rows = [np.hstack(tiles[i:i + N_COLS]) for i in range(0, len(tiles), N_COLS)]
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run_report(
    selection_dir: Path,
    elig_dir: Path,
    clf_dir: Path,
    emb_dir: Path,
    video_dir: Path,
    sample: int = 200_000,
    seed: int = 0,
    sheets: bool = True,
    device: torch.device | None = None,
) -> dict:
    """Evaluate a selection manifest and write `report.json`, `report.md` and contact sheets.

    Args:
        selection_dir: directory with `manifest.parquet` and `summary.json`; reports are
            written here (sheets under `eval_sheets/`).
        elig_dir: directory of eligibility tables.
        clf_dir: directory of classifier prediction tables.
        emb_dir: directory of per-video DINOv3 `.npy` files.
        video_dir: directory of aligned rectangle videos (for contact sheets).
        sample: eligible frames per class used for coverage and state metrics.
        seed: rng seed.
        sheets: whether to write contact sheets.
        device: torch device; defaults to CUDA if available.

    Returns:
        the report as a dict (also saved to `report.json`).
    """
    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    manifest = pd.read_parquet(selection_dir / MANIFEST_FILENAME)
    summary = json.loads((selection_dir / SUMMARY_FILENAME).read_text())
    classes = summary['stage1']['classes']
    stems = sorted(manifest['video'].unique())
    video_idx = {s: i for i, s in enumerate(stems)}
    rng = np.random.default_rng(seed)
    pools = load_pools(stems, elig_dir, clf_dir, emb_dir, classes)

    per_video = manifest.groupby('video').size().sort_values()
    report = {
        'n_videos': len(stems),
        'per_video': {
            'min': int(per_video.min()), 'p10': float(np.percentile(per_video, 10)),
            'median': float(per_video.median()), 'p90': float(np.percentile(per_video, 90)),
            'max': int(per_video.max()), 'under_500': int((per_video < 500).sum()),
            'over_2000': int((per_video > 2000).sum()),
            'lowest': per_video.head(3).to_dict(), 'highest': per_video.tail(3).to_dict(),
        },
        'classes': {},
    }
    sheet_dir = selection_dir / 'eval_sheets'
    if sheets:
        sheet_dir.mkdir(exist_ok=True)

    for cls in classes:
        sel = manifest[manifest['predicted_pattern'] == cls].reset_index(drop=True)
        if cls not in pools or len(sel) == 0:
            continue
        pool = pools[cls]
        n_pool = len(pool['frame'])
        _logger.info(f'{cls}: {len(sel)} selected of {n_pool} eligible')

        # map selected frames to rows of the class pool
        keys = pool['video'].astype(np.int64) * 10**7 + pool['frame']
        order = np.argsort(keys)
        sel_keys = (sel['video'].map(video_idx).to_numpy().astype(np.int64) * 10**7
                    + sel['frame_idx'].to_numpy())
        sel_rows = order[np.searchsorted(keys[order], sel_keys)]
        if not (keys[sel_rows] == sel_keys).all():
            raise ValueError(f'{cls}: manifest contains frames that are not eligible {cls} frames')

        emb = normalize(pool['emb'].float().to(device))
        sample_idx = np.sort(rng.choice(n_pool, min(sample, n_pool), replace=False))
        sample_emb = emb[torch.from_numpy(sample_idx).to(device)]
        centroids = spherical_kmeans(sample_emb, min(N_COARSE, len(sample_idx)), seed=seed)[1]
        sizes = np.bincount(assign(sample_emb, centroids).cpu().numpy(), minlength=len(centroids))
        rare = np.argsort(sizes)[:N_RARE]

        counts = sel.groupby('video').size()
        entry = {
            'quota': summary['quotas'][cls], 'selected': int(len(sel)), 'eligible': int(n_pool),
            'conf_selected': float(sel['confidence'].mean()),
            'conf_eligible': float(pool['conf'].mean()),
            'lowconf_selected': float((sel['confidence'] < LOW_CONFIDENCE).mean()),
            'lowconf_eligible': float((pool['conf'] < LOW_CONFIDENCE).mean()),
            'top5_video_share': float(counts.sort_values().tail(5).sum() / len(sel)),
            'videos_contributing': int(len(counts)),
            'videos_with_class': int(len(np.unique(pool['video']))),
        }
        sel_emb = emb[torch.from_numpy(sel_rows).to(device)]
        base_rows = rng.choice(n_pool, len(sel), replace=False)
        base_emb = emb[torch.from_numpy(base_rows).to(device)]
        entry['selected_metrics'] = pool_metrics(sel_emb, sample_emb, centroids, sizes, rare)
        entry['random_metrics'] = pool_metrics(base_emb, sample_emb, centroids, sizes, rare)

        base = pd.DataFrame({'video': pool['video'][base_rows],
                             'frame_idx': pool['frame'][base_rows]})
        sel_v = pd.DataFrame({'video': sel['video'].map(video_idx),
                              'frame_idx': sel['frame_idx']})
        entry['temporal_redundancy'] = {
            f'{name}_within_{gap}': near_fraction(df, gap)
            for gap in (10, 100) for name, df in (('sel', sel_v), ('random', base))
        }
        report['classes'][cls] = entry

        if sheets:
            slug = cls.replace(' ', '_')
            make_sheet(sel.sample(min(N_SHEET, len(sel)), random_state=seed), video_dir,
                       sheet_dir / f'{slug}_random.jpg')
            if len(sel) > 6:
                isolation = nearest_distance(sel_emb, sel_emb, exclude_self=True, k=5)
                top = np.argsort(-isolation.cpu().numpy())[:N_SHEET]
                make_sheet(sel.iloc[top], video_dir, sheet_dir / f'{slug}_isolated.jpg')
        del emb, sample_emb, sel_emb, base_emb
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    (selection_dir / 'report.json').write_text(json.dumps(report, indent=1))
    (selection_dir / 'report.md').write_text(format_report(report))
    return report


def format_report(report: dict) -> str:
    """Render a report dict as a short text table.

    Args:
        report: report from `run_report`.

    Returns:
        multi-line text: the per-video allocation, then one row per class.
    """
    lines = [f'videos: {report["n_videos"]}', f'per video: {report["per_video"]}', '']
    header = ('class', 'quota', 'sel', 'conf s/e', 'lowconf s/e', 'top5vid', 'vids',
              'cov p999 s/r', 'dens s', 'rare s/r', 'near10 s/r')
    lines.append(' | '.join(header))
    for cls, e in report['classes'].items():
        s, r, t = e['selected_metrics'], e['random_metrics'], e['temporal_redundancy']
        lines.append(' | '.join([
            cls, str(e['quota']), str(e['selected']),
            f'{e["conf_selected"]:.2f}/{e["conf_eligible"]:.2f}',
            f'{e["lowconf_selected"]:.2f}/{e["lowconf_eligible"]:.2f}',
            f'{e["top5_video_share"]:.2f}',
            f'{e["videos_contributing"]}/{e["videos_with_class"]}',
            f'{s["cov_p999"]:.3f}/{r["cov_p999"]:.3f}', f'{s["dens_ratio"]:.2f}',
            f'{s["rare_ratio"]:.2f}/{r["rare_ratio"]:.2f}',
            f'{t["sel_within_10"]:.2f}/{t["random_within_10"]:.2f}',
        ]))
    return '\n'.join(lines) + '\n'
