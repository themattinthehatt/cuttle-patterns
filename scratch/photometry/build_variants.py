"""Build same-frame variant results_dirs for the Step 4 model comparison.

Each variant under `.iter-1.0-photometry/variants/{name}/` is a self-contained results_dir:
`beast_frames/{video}/img*.png` (the identical frame set in every variant), `manifests/
extract.parquet`, a symlink to the iter-1.0 `classifications/` (labels stay fixed to the original
frames), and an empty `beast_models/`. The frame set is the gate-passing anchors of the clean
videos (current crop video reproduces iter-1.0 frames) that exist in every variant.

Variants: orig (iter-1.0 frames), raw (re-export, no correction), final (tone curve + own-floor
target), wrong (borrowed parameters), norm (per-tank per-channel mean/std matching, animal pixels).

Example:
    python scratch/photometry/build_variants.py scratchpad_dir
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RES = Path('/media/mattw/CUTTLE/results')
OUT = RES / '.iter-1.0-photometry'
SOURCES = {
    'orig': RES / '.iter-1.0' / 'beast_frames',
    'raw': OUT / 'beast_frames_rawreexport_final',
    'final': OUT / 'beast_frames_final',
    'wrong': OUT / 'beast_frames_wrongfinal',
}
N_STAT = 25  # frames per video for the normalization statistics


def animal_pixels(img: np.ndarray) -> np.ndarray:
    """Boolean mask of non-background pixels (same rule as the pixel-stat gate)."""
    return img.sum(2) > 24


def main(qc_dir: Path) -> None:
    """Write all variant directories and report the frame counts."""
    clean = []
    for f in sorted((qc_dir / 'apply').glob('apply_Day*.csv')):
        stat = pd.read_csv(f).groupby('fish_id').mad_orig_vs_rawreexport.mean()
        clean += [f'{f.stem.removeprefix("apply_")}_{k}_Crop' for k, m in stat.items() if m < 10]
    extract = pd.read_parquet(RES / '.iter-1.0' / 'manifests' / 'extract.parquet')
    extract['video'] = extract.session_id + '_' + extract.fish_id + '_Crop'
    extract = extract[extract.video.isin(clean)]
    keep = []
    for video, sub in extract.groupby('video'):
        sets = [
            {int(p.stem[3:]) for p in (src / video).glob('img*.png')} for src in SOURCES.values()
        ]
        common = set.intersection(*sets) & set(sub.frame_idx)
        keep += [(video, f) for f in sorted(common)]
    frames = pd.DataFrame(keep, columns=['video', 'frame_idx'])
    print(f'{len(clean)} clean videos, {len(frames)} common frames')
    extract = extract.merge(frames, on=['video', 'frame_idx'])
    (OUT / 'manifests').mkdir(exist_ok=True)
    frames.to_parquet(OUT / 'manifests' / 'common_frames.parquet')

    def path(variant: str, video: str, frame: int) -> Path:
        return OUT / 'variants' / variant / 'beast_frames' / video / f'img{frame:08d}.png'

    # symlinked variants
    for name, src in SOURCES.items():
        for video, frame in keep_pairs(frames):
            dst = path(name, video, frame)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_symlink():
                dst.symlink_to(src / video / dst.name)

    # per-tank normalization baseline on the original frames
    rng = np.random.default_rng(0)
    stats: dict[int, list] = {}
    for video, sub in frames.groupby('video'):
        tank = int(video.split('Tank')[1][0])
        for frame in rng.choice(sub.frame_idx.to_numpy(), min(N_STAT, len(sub)), replace=False):
            img = cv2.imread(str(SOURCES['orig'] / video / f'img{frame:08d}.png'))
            stats.setdefault(tank, []).append(img[animal_pixels(img)].astype(np.float64))
    mu = {t: np.concatenate(v).mean(0) for t, v in stats.items()}
    sd = {t: np.concatenate(v).std(0) for t, v in stats.items()}
    mu_g, sd_g = np.mean(list(mu.values()), 0), np.mean(list(sd.values()), 0)
    print('per-tank animal mean BGR:', {t: m.round(1).tolist() for t, m in mu.items()})
    for video, frame in keep_pairs(frames):
        tank = int(video.split('Tank')[1][0])
        dst = path('norm', video, frame)
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(SOURCES['orig'] / video / dst.name))
        mask = animal_pixels(img)
        out = np.zeros_like(img)
        out[mask] = np.clip((img[mask] - mu[tank]) / sd[tank] * sd_g + mu_g, 0, 255)
        cv2.imwrite(str(dst), out)

    for name in (*SOURCES, 'norm'):
        root = OUT / 'variants' / name
        (root / 'manifests').mkdir(exist_ok=True)
        (root / 'beast_models').mkdir(exist_ok=True)
        table = extract.assign(
            image_path=[str(path(name, v, f)) for v, f in zip(extract.video, extract.frame_idx)],
        ).drop(columns='video')
        table.to_parquet(root / 'manifests' / 'extract.parquet')
        link = root / 'classifications'
        if not link.is_symlink():
            link.symlink_to(RES / '.iter-1.0' / 'classifications')
        print(f'{name}: {len(table)} frames')


def keep_pairs(frames: pd.DataFrame) -> list[tuple[str, int]]:
    """(video, frame) tuples of the common frame set."""
    return list(zip(frames.video, frames.frame_idx.astype(int)))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
