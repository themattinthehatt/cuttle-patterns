"""DINOv3 CLS-token features for every frame of the aligned videos (`cuttle extract-dino-cls`).

Decodes whole rectangle videos (`results_dir/rectangles/*_Crop.mp4`) and runs a frozen DINOv3 ViT
through `cuttle_patterns.embedders.dinov3`: frames are stretched to a square (no center crop) at
`resolution` and ImageNet-normalized, exactly as `cuttle embed` does, and the post-norm CLS token
is kept. Inference runs under fp16 autocast; stored features are float16, one row per frame, so
row i of a video's array is frame i. This is the per-frame, whole-video counterpart of `cuttle
embed`'s `cls` readout over exported stills.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from cuttle_patterns.embedders.dinov3 import DINOv3Backbone, resize_and_normalize
from cuttle_patterns.preprocessing.extract.common import (
    prefetch_batches,
    progress_path,
    save_npy_atomic,
    shard_items,
    video_frame_count,
    write_progress,
)

_logger = logging.getLogger(__name__)

DEFAULT_ARCH = 'vits16'
DEFAULT_RESOLUTION = 224
DEFAULT_BATCH_SIZE = 256
CONFIG_FILENAME = 'config.yaml'


def embedding_dir_name(arch: str) -> str:
    """Name of the per-architecture output directory.

    Args:
        arch: DINOv3 architecture key, e.g. `vits16`.

    Returns:
        directory name, e.g. `dinov3_vits16`.
    """
    return f'dinov3_{arch}'


@torch.no_grad()
def embed_video(
    video_path: Path,
    backbone: DINOv3Backbone,
    batch_size: int = DEFAULT_BATCH_SIZE,
    position: int = 0,
) -> np.ndarray:
    """Compute the CLS token of every frame of one video.

    Args:
        video_path: aligned rectangle video.
        backbone: loaded DINOv3 backbone.
        batch_size: frames per forward pass.
        position: tqdm bar row, so sharded processes sharing a terminal don't overwrite each
            other.

    Returns:
        float16 array, shape (n_frames, hidden_size).
    """

    def to_input(batch_bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(batch_bgr[..., ::-1])  # bgr -> rgb
        return resize_and_normalize(rgb, backbone.resolution)

    cls_tokens = []
    with tqdm(total=video_frame_count(video_path), desc=video_path.stem, unit='frame',
              position=position, leave=False) as pbar:
        for batch in prefetch_batches(video_path, batch_size, transform=to_input):
            x = torch.from_numpy(batch).to(backbone.device)
            with torch.autocast(backbone.device.type, dtype=torch.float16):
                cls = backbone.forward(x).cls
            cls_tokens.append(cls.float().cpu().numpy().astype(np.float16))
            pbar.update(len(batch))
    return np.concatenate(cls_tokens)


def run_dino_cls(
    video_paths: list[Path],
    out_dir: Path,
    arch: str = DEFAULT_ARCH,
    resolution: int = DEFAULT_RESOLUTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
    shard: int = 0,
    n_shards: int = 1,
    device: torch.device | None = None,
) -> list[Path]:
    """Embed every frame of every (sharded) video and save one `.npy` per video.

    Videos whose `.npy` already exists are skipped unless `overwrite` is set. Files are written
    under a temporary name and renamed, so an interrupted run never leaves a partial file that
    looks complete. Progress is written to `progress_shard{i}of{n}.json` in `out_dir`.

    Args:
        video_paths: aligned rectangle videos.
        out_dir: output directory (`{video}.npy`, `config.yaml`).
        arch: DINOv3 architecture key (see `cuttle_patterns.embedders.dinov3.ARCH_TO_HF_ID`).
        resolution: square input side length, a multiple of 16.
        batch_size: frames per forward pass.
        overwrite: re-embed videos that already have outputs.
        shard: this process's shard index.
        n_shards: total number of shards.
        device: torch device; defaults to CUDA if available.

    Returns:
        paths of the arrays written by this call.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    backbone = DINOv3Backbone(arch, resolution, device)
    video_paths = shard_items(video_paths, shard, n_shards)
    _logger.info(f'{backbone.key}: {len(video_paths)} videos; writing to {out_dir}')

    config_path = out_dir / CONFIG_FILENAME
    if not config_path.exists():
        config_path.write_text(yaml.safe_dump({
            'backbone': backbone.key,
            'readout': 'cls (post-norm CLS token)',
            'input': 'aligned rectangle videos (results_dir/rectangles), every frame',
            'preprocessing': 'stretch to square (no center crop), imagenet norm',
            'precision': 'fp16 autocast; stored as float16',
            **backbone.metadata(),
        }, sort_keys=False))

    progress_file = progress_path(out_dir, shard, n_shards)
    written = []
    for idx, video_path in enumerate(video_paths):
        write_progress(progress_file, idx, len(video_paths), video_path.stem)
        out_path = out_dir / f'{video_path.stem}.npy'
        if out_path.exists() and not overwrite:
            _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}: exists, skipping')
            continue
        _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}')
        features = embed_video(video_path, backbone, batch_size, position=shard)
        n_expected = video_frame_count(video_path)
        if len(features) != n_expected:
            _logger.warning(
                f'{video_path.stem}: decoded {len(features)} frames but the container reports '
                f'{n_expected}'
            )
        save_npy_atomic(out_path, features)
        written.append(out_path)
        _logger.info(f'{video_path.stem}: saved {features.shape} float16')
    write_progress(progress_file, len(video_paths), len(video_paths), None)
    return written
