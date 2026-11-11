"""Run the two-head skin-pattern classifier over every frame of full videos.

Status: working script, not yet promoted into the `cuttle` CLI.

Unlike `scripts/classify_skin_pattern.py` (exported still frames only), this decodes whole videos
and classifies every frame, so any frame can be revisited later without re-running anything.
Input is the 512x512 crop videos (`{data_dir}/*_Crop.mp4`), the same space the classifier was
trained in. The classifier is the collaborator-trained resnet18 with two heads on the shared
512-d pooled features: a 5-way pattern head (used here) and a binary black-border head
(ignored). Preprocessing matches training's eval transform: resize to 224x224 (bilinear,
antialiased) and ImageNet normalization.

Output, one file per video (resumable, shardable across processes):

    {out_dir}/{video_stem}.parquet        one row per frame: `frame_idx`, `predicted_pattern`,
                                          `confidence`, and one `prob_{class}` per class
    {out_dir}/{video_stem}_features.npy   (only with --save-features) float16 (n_frames, 512)
    {out_dir}/classifier.yaml             provenance (weights, classes, preprocessing)
    {out_dir}/progress_shard{i}of{n}.json done/total and current video, for a running job

`out_dir` defaults to `{results_dir}/classifications/{model_name}`, where `model_name` defaults to the model
directory's name (e.g. `2026-11-07_classifier_model`). Videos whose parquet
already exists are skipped unless --overwrite is given. Files are written to a temporary name
and renamed, so an interrupted run never leaves a partial file that looks complete.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI; run directly, e.g.:

    python scripts/classify_video_skin_pattern.py
    # or sharded across processes:
    python scripts/classify_video_skin_pattern.py --shard 0 --n-shards 3
"""

import argparse
import json
import logging
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.models import resnet18
from tqdm import tqdm

from cuttle_patterns import paths
from cuttle_patterns.config import load_config

_logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path('/media/mattw/CUTTLE/classification/2026-11-07_classifier_model')
WEIGHTS_FILENAME = 'skin_pattern_border_resnet18_twohead_best.pt'
BEHAVIORS_FILENAME = 'behaviors.json'
DEFAULT_BATCH_SIZE = 256
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_GLOB = '*_Crop.mp4'
QUEUE_DEPTH = 4


class TwoHead(nn.Module):
    """Pattern (n-way) and black-border (binary) linear heads on shared pooled features."""

    def __init__(self, in_features: int, num_pattern_classes: int):
        """Create the two heads.

        Args:
            in_features: dimension of the pooled backbone features.
            num_pattern_classes: number of skin-pattern classes.
        """
        super().__init__()
        self.pattern = nn.Linear(in_features, num_pattern_classes)
        self.border = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pattern logits, border logit)."""
        return self.pattern(x), self.border(x)


def load_classifier(
    model_dir: Path,
    device: torch.device,
) -> tuple[nn.Module, list[str], dict[str, torch.Tensor]]:
    """Load the two-head classifier and hook its 512-d penultimate features.

    Args:
        model_dir: directory holding the weights `state_dict` and `behaviors.json`.
        device: device to load the model onto.

    Returns:
        `(model, classes, captured)`: the model in eval mode, the pattern class names in output
        order, and a dict whose `'value'` entry holds the flattened `(batch, 512)` features
        after each forward call.
    """
    classes = json.loads((model_dir / BEHAVIORS_FILENAME).read_text())
    model = resnet18(weights=None)
    model.fc = TwoHead(model.fc.in_features, len(classes))
    state_dict = torch.load(model_dir / WEIGHTS_FILENAME, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        captured['value'] = torch.flatten(output, 1)

    model.avgpool.register_forward_hook(hook)
    return model, classes, captured


def preprocess(frames_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a batch of bgr uint8 frames to normalized classifier input on the device.

    Args:
        frames_bgr: uint8 array, shape (B, H, W, 3), as decoded by OpenCV.
        device: device to run the resize on.

    Returns:
        float32 tensor, shape (B, 3, 224, 224).
    """
    x = torch.from_numpy(frames_bgr).to(device).flip(-1)  # bgr -> rgb
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', antialias=True,
                      align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def read_batches(video_path: Path, batch_size: int, out: queue.Queue) -> None:
    """Decode a video sequentially into batches on a queue (runs in a producer thread).

    Puts uint8 arrays of shape (<=batch_size, H, W, 3), then `None` as an end marker. If decoding
    fails, puts the exception instead so the consumer can re-raise it.

    Args:
        video_path: video to decode.
        batch_size: frames per batch.
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
            buf.append(frame)
            if len(buf) == batch_size:
                out.put(np.stack(buf))
                buf = []
        if buf:
            out.put(np.stack(buf))
        cap.release()
        out.put(None)
    except Exception as e:  # noqa: BLE001 - forwarded to the consumer thread
        out.put(e)


@torch.no_grad()
def classify_video(
    video_path: Path,
    model: nn.Module,
    captured: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
    save_features: bool,
    n_frames: int,
    position: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Classify every frame of one video.

    Args:
        video_path: crop video.
        model: loaded two-head classifier.
        captured: feature-hook dict from `load_classifier`.
        device: torch device.
        batch_size: frames per forward pass.
        save_features: also return the 512-d penultimate features.
        n_frames: frame count reported by the container, used for the progress bar total.
        position: tqdm bar row, so sharded processes sharing a terminal don't overwrite each
            other.

    Returns:
        `(probs, features)`: float32 pattern softmax of shape (n_frames, n_classes), and float16
        features of shape (n_frames, 512) or None.
    """
    batches: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
    thread = threading.Thread(target=read_batches, args=(video_path, batch_size, batches),
                              daemon=True)
    thread.start()
    probs, feats = [], []
    with tqdm(total=n_frames, desc=video_path.stem, unit='frame', position=position,
              leave=False) as pbar:
        while True:
            item = batches.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            pattern_logits, _ = model(preprocess(item, device))  # border head ignored
            probs.append(F.softmax(pattern_logits, dim=1).cpu().numpy())
            if save_features:
                feats.append(captured['value'].cpu().numpy().astype(np.float16))
            pbar.update(len(item))
    thread.join()
    return np.concatenate(probs), (np.concatenate(feats) if save_features else None)


def build_predictions(probs: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Build the per-frame predictions table.

    Args:
        probs: (n_frames, n_classes) softmax probabilities.
        classes: class names in output order.

    Returns:
        frame with `frame_idx`, `predicted_pattern`, `confidence`, and `prob_{class}` columns
        (spaces in class names replaced with underscores).
    """
    top = probs.argmax(axis=1)
    df = pd.DataFrame({
        'frame_idx': np.arange(len(probs), dtype=np.int32),
        'predicted_pattern': np.array(classes)[top],
        'confidence': probs[np.arange(len(probs)), top].astype(np.float32),
    })
    for idx, name in enumerate(classes):
        df[f'prob_{name.replace(" ", "_")}'] = probs[:, idx].astype(np.float32)
    return df


def write_atomic(path: Path, write_fn) -> None:
    """Write a file via a temporary name, then rename, so partial files never look complete.

    Args:
        path: final path.
        write_fn: callable taking the temporary path and writing the file there.
    """
    tmp = path.with_name(f'{path.name}.tmp')
    write_fn(tmp)
    tmp.replace(path)


def save_npy(path: Path, array: np.ndarray) -> None:
    """Save an array to exactly `path` (np.save would otherwise append '.npy').

    Args:
        path: destination file.
        array: array to save.
    """
    with open(path, 'wb') as f:
        np.save(f, array)


def main() -> None:
    """Classify every frame of every (sharded) crop video."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR,
                        help='directory with the weights and behaviors.json')
    parser.add_argument('--model-name', default=None,
                        help='output subdirectory name (default: the model directory name)')
    parser.add_argument('--video-dir', type=Path, default=None,
                        help='directory of crop videos (default: config data_dir)')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='output directory (default: results_dir/classifications/model_name)')
    parser.add_argument('--videos', nargs='*', default=None,
                        help='video stems to process (default: every *_Crop.mp4)')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--save-features', action='store_true',
                        help='also save the 512-d penultimate features (float16 .npy)')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    config = load_config()
    video_dir = args.video_dir if args.video_dir is not None else config.data_dir
    model_name = args.model_name or args.model_dir.name
    out_dir = (args.out_dir if args.out_dir is not None
               else config.results_dir / paths.CLASSIFICATIONS_RELPATH / model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(video_dir.glob(VIDEO_GLOB))
    if args.videos:
        video_paths = [p for p in video_paths if p.stem in set(args.videos)]
    video_paths = video_paths[args.shard::args.n_shards]
    if not video_paths:
        raise ValueError(f'no videos matched in {video_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes, captured = load_classifier(args.model_dir, device)
    _logger.info(f'classes: {classes}; {len(video_paths)} videos; writing to {out_dir}')

    provenance = {
        'weights': str(args.model_dir / WEIGHTS_FILENAME),
        'classes': classes,
        'input': 'full-frame 512x512 crop videos, every frame',
        'preprocessing': f'resize to {IMG_SIZE}x{IMG_SIZE} (bilinear, antialias), imagenet norm',
        'ignored_head': 'border',
    }
    provenance_path = out_dir / 'classifier.yaml'
    if not provenance_path.exists():
        provenance_path.write_text(yaml.safe_dump(provenance, sort_keys=False))

    progress_path = out_dir / f'progress_shard{args.shard}of{args.n_shards}.json'
    for idx, video_path in enumerate(video_paths):
        progress_path.write_text(json.dumps({
            'done': idx, 'total': len(video_paths), 'current': video_path.stem,
        }))
        out_path = out_dir / f'{video_path.stem}.parquet'
        if out_path.exists() and not args.overwrite:
            _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}: exists, skipping')
            continue
        _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}')
        cap = cv2.VideoCapture(str(video_path))
        n_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        probs, feats = classify_video(
            video_path, model, captured, device, args.batch_size, args.save_features,
            n_expected, args.shard,
        )
        if len(probs) != n_expected:
            _logger.warning(
                f'{video_path.stem}: decoded {len(probs)} frames but the container reports '
                f'{n_expected}'
            )
        if feats is not None:
            feat_path = out_dir / f'{video_path.stem}_features.npy'
            write_atomic(feat_path, lambda tmp: save_npy(tmp, feats))
        # parquet last: its existence marks the video as complete
        write_atomic(out_path, lambda p: build_predictions(probs, classes).to_parquet(p))
        share = pd.Series(np.array(classes)[probs.argmax(1)]).value_counts(normalize=True)
        _logger.info(f'{video_path.stem}: {len(probs)} frames; '
                     + ', '.join(f'{k} {v:.2f}' for k, v in share.items()))
    progress_path.write_text(json.dumps({
        'done': len(video_paths), 'total': len(video_paths), 'current': None,
    }))


if __name__ == '__main__':
    main()
