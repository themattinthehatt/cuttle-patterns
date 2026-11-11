"""Compute DINOv3 CLS-token features for every frame of the aligned rectangle videos.

Status: working script, not yet promoted into the `cuttle` CLI (`cuttle embed` does the same
backbone/readout but over exported still frames, with a manifest).

Decodes whole videos (`{results_dir}/rectangles/*_Crop.mp4`, the aligned rectangle videos;
`*_overlay.mp4` QC clips are ignored) and runs a frozen DINOv3 ViT through the existing
`cuttle_patterns.embedders.dinov3` backbone: frames are stretched to a square (no center
crop) at `--resolution` and ImageNet-normalized, exactly as `cuttle embed` does, and the
post-norm CLS token is kept. Inference runs under fp16 autocast; the stored features are
float16.

Output, one file per video (resumable, shardable across processes):

    {out_dir}/{video_stem}.npy      float16, shape (n_frames, hidden_size); row i is frame i
    {out_dir}/config.yaml           provenance (backbone, resolution, input, precision)
    {out_dir}/progress_shard{i}of{n}.json   done/total and current video, for a running job

`out_dir` defaults to `{results_dir}/embeddings/dinov3_{arch}`. Videos whose .npy already
exists are skipped unless --overwrite is given. Files are written under a temporary name and
renamed, so an interrupted run never leaves a partial file that looks complete.

Run directly, e.g.:

    python scripts/dinov3_video_cls.py
    # or sharded across processes:
    python scripts/dinov3_video_cls.py --shard 0 --n-shards 2
"""

import argparse
import json
import logging
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from cuttle_patterns import paths
from cuttle_patterns.config import load_config
from cuttle_patterns.embedders.dinov3 import ARCH_TO_HF_ID, DINOv3Backbone, resize_and_normalize

_logger = logging.getLogger(__name__)

DEFAULT_ARCH = 'vits16'
DEFAULT_RESOLUTION = 224
DEFAULT_BATCH_SIZE = 256
VIDEO_GLOB = '*_Crop.mp4'
QUEUE_DEPTH = 3


def read_batches(
    video_path: Path,
    batch_size: int,
    resolution: int,
    out: queue.Queue,
) -> None:
    """Decode and preprocess a video into batches on a queue (runs in a producer thread).

    Puts float32 arrays of shape (<=batch_size, 3, resolution, resolution), then `None` as an end
    marker. If decoding fails, puts the exception instead so the consumer can re-raise it.

    Args:
        video_path: video to decode.
        batch_size: frames per batch.
        resolution: square side length frames are resized to.
        out: queue to put batches on.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f'could not open video file: {video_path}')
        buf = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            buf.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(buf) == batch_size:
                out.put(resize_and_normalize(np.stack(buf), resolution))
                buf = []
        if buf:
            out.put(resize_and_normalize(np.stack(buf), resolution))
        cap.release()
        out.put(None)
    except Exception as e:  # noqa: BLE001 - forwarded to the consumer thread
        out.put(e)


@torch.no_grad()
def embed_video(
    video_path: Path,
    backbone: DINOv3Backbone,
    batch_size: int,
    n_frames: int,
    position: int = 0,
) -> np.ndarray:
    """Compute the CLS token of every frame of one video.

    Args:
        video_path: aligned rectangle video.
        backbone: loaded DINOv3 backbone.
        batch_size: frames per forward pass.
        n_frames: frame count reported by the container, used for the progress bar total.
        position: tqdm bar row, so sharded processes sharing a terminal don't overwrite each
            other.

    Returns:
        float16 array, shape (n_frames, hidden_size).
    """
    batches: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    thread = threading.Thread(
        target=read_batches,
        args=(video_path, batch_size, backbone.resolution, batches),
        daemon=True,
    )
    thread.start()
    cls_tokens = []
    with tqdm(total=n_frames, desc=video_path.stem, unit='frame', position=position,
              leave=False) as pbar:
        while True:
            item = batches.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            x = torch.from_numpy(item).to(backbone.device)
            with torch.autocast('cuda', dtype=torch.float16):
                cls = backbone.forward(x).cls
            cls_tokens.append(cls.float().cpu().numpy().astype(np.float16))
            pbar.update(len(item))
    thread.join()
    return np.concatenate(cls_tokens)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Save an array to exactly `path` via a temporary name, then rename.

    Args:
        path: destination file.
        array: array to save.
    """
    tmp = path.with_name(f'{path.name}.tmp')
    with open(tmp, 'wb') as f:
        np.save(f, array)
    tmp.replace(path)


def main() -> None:
    """Embed every frame of every (sharded) rectangle video."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arch', default=DEFAULT_ARCH, choices=list(ARCH_TO_HF_ID))
    parser.add_argument('--resolution', type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument('--video-dir', type=Path, default=None,
                        help='directory of rectangle videos (default: results_dir/rectangles)')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='output directory (default: results_dir/embeddings/dinov3_arch)')
    parser.add_argument('--videos', nargs='*', default=None,
                        help='video stems to process (default: every *_Crop.mp4)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    config = load_config()
    video_dir = (args.video_dir if args.video_dir is not None
                 else config.results_dir / paths.RECTANGLES_RELPATH)
    out_dir = (args.out_dir if args.out_dir is not None
               else config.results_dir / paths.EMBEDDINGS_RELPATH / f'dinov3_{args.arch}')
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(video_dir.glob(VIDEO_GLOB))
    if args.videos:
        video_paths = [p for p in video_paths if p.stem in set(args.videos)]
    video_paths = video_paths[args.shard::args.n_shards]
    if not video_paths:
        raise ValueError(f'no videos matched in {video_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('fp16 autocast inference requires a CUDA device')
    backbone = DINOv3Backbone(args.arch, args.resolution, device)
    _logger.info(f'{backbone.key}: {len(video_paths)} videos; writing to {out_dir}')

    config_path = out_dir / 'config.yaml'
    if not config_path.exists():
        provenance = {
            'backbone': backbone.key,
            'readout': 'cls (post-norm CLS token)',
            'input': 'aligned rectangle videos (results_dir/rectangles), every frame',
            'preprocessing': 'stretch to square (no center crop), imagenet norm',
            'precision': 'fp16 autocast; stored as float16',
            **backbone.metadata(),
        }
        config_path.write_text(yaml.safe_dump(provenance, sort_keys=False))

    progress_path = out_dir / f'progress_shard{args.shard}of{args.n_shards}.json'
    for idx, video_path in enumerate(video_paths):
        progress_path.write_text(json.dumps({
            'done': idx, 'total': len(video_paths), 'current': video_path.stem,
        }))
        out_path = out_dir / f'{video_path.stem}.npy'
        if out_path.exists() and not args.overwrite:
            _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}: exists, skipping')
            continue
        _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}')
        cap = cv2.VideoCapture(str(video_path))
        n_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        features = embed_video(video_path, backbone, args.batch_size, n_expected, args.shard)
        if len(features) != n_expected:
            _logger.warning(
                f'{video_path.stem}: decoded {len(features)} frames but the container reports '
                f'{n_expected}'
            )
        save_npy_atomic(out_path, features)
        _logger.info(f'{video_path.stem}: saved {features.shape} float16')
    progress_path.write_text(json.dumps({
        'done': len(video_paths), 'total': len(video_paths), 'current': None,
    }))


if __name__ == '__main__':
    main()
