"""Per-frame skin-pattern classification of full videos (`cuttle extract-classify`).

Runs a collaborator-trained resnet18 classifier over every frame of the 512x512 crop videos
(`{data_dir}/*_Crop.mp4`, the space the classifier was trained in). The classifier has two heads
on the shared 512-d pooled features: a pattern head (n-way, used here) and a binary
black-border head (ignored). Preprocessing matches the classifier's eval transform: resize to
224x224 (bilinear, antialiased) and ImageNet normalization.

Unlike `scripts/classify_skin_pattern.py`, which classifies exported still frames, this decodes
whole videos so any frame can be revisited later without re-running anything.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.models import resnet18
from tqdm import tqdm

from cuttle_patterns.preprocessing.extract.common import (
    prefetch_batches,
    progress_path,
    save_npy_atomic,
    save_parquet_atomic,
    shard_items,
    video_frame_count,
    write_progress,
)

_logger = logging.getLogger(__name__)

WEIGHTS_FILENAME = 'skin_pattern_border_resnet18_twohead_best.pt'
BEHAVIORS_FILENAME = 'behaviors.json'
PROVENANCE_FILENAME = 'classifier.yaml'
DEFAULT_BATCH_SIZE = 256
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        """Return (pattern logits, border logit).

        Args:
            x: pooled features, shape (B, in_features).

        Returns:
            pattern logits (B, num_pattern_classes) and border logit (B, 1).
        """
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
    x = F.interpolate(
        x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', antialias=True, align_corners=False,
    )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def classify_video(
    video_path: Path,
    model: nn.Module,
    captured: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_features: bool = False,
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
        position: tqdm bar row, so sharded processes sharing a terminal don't overwrite each
            other.

    Returns:
        `(probs, features)`: float32 pattern softmax of shape (n_frames, n_classes), and float16
        features of shape (n_frames, 512) or None.
    """
    probs, feats = [], []
    n_frames = video_frame_count(video_path)
    with tqdm(total=n_frames, desc=video_path.stem, unit='frame', position=position,
              leave=False) as pbar:
        for batch in prefetch_batches(video_path, batch_size):
            pattern_logits, _ = model(preprocess(batch, device))  # border head ignored
            probs.append(F.softmax(pattern_logits, dim=1).cpu().numpy())
            if save_features:
                feats.append(captured['value'].cpu().numpy().astype(np.float16))
            pbar.update(len(batch))
    features = np.concatenate(feats) if save_features else None
    return np.concatenate(probs), features


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


def run_classify(
    video_paths: list[Path],
    model_dir: Path,
    out_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    save_features: bool = False,
    overwrite: bool = False,
    shard: int = 0,
    n_shards: int = 1,
    device: torch.device | None = None,
) -> list[Path]:
    """Classify every frame of every (sharded) video and save per-video prediction tables.

    Videos whose parquet already exists are skipped unless `overwrite` is set. The parquet is
    written last, so its existence marks the video as complete. Progress is written to
    `progress_shard{i}of{n}.json` in `out_dir`; a `classifier.yaml` records provenance.

    Args:
        video_paths: crop videos to classify.
        model_dir: directory holding the classifier weights and `behaviors.json`.
        out_dir: output directory (`{video}.parquet`, optionally `{video}_features.npy`).
        batch_size: frames per forward pass.
        save_features: also save the 512-d penultimate features (float16 .npy).
        overwrite: reclassify videos that already have outputs.
        shard: this process's shard index.
        n_shards: total number of shards.
        device: torch device; defaults to CUDA if available.

    Returns:
        paths of the prediction tables written by this call.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device if device is not None else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    model, classes, captured = load_classifier(model_dir, device)
    video_paths = shard_items(video_paths, shard, n_shards)
    _logger.info(f'classes: {classes}; {len(video_paths)} videos; writing to {out_dir}')

    provenance_path = out_dir / PROVENANCE_FILENAME
    if not provenance_path.exists():
        provenance_path.write_text(yaml.safe_dump({
            'weights': str(model_dir / WEIGHTS_FILENAME),
            'classes': classes,
            'input': 'full-frame 512x512 crop videos, every frame',
            'preprocessing': f'resize to {IMG_SIZE}x{IMG_SIZE} (bilinear, antialias), '
                             f'imagenet norm',
            'ignored_head': 'border',
        }, sort_keys=False))

    progress_file = progress_path(out_dir, shard, n_shards)
    written = []
    for idx, video_path in enumerate(video_paths):
        write_progress(progress_file, idx, len(video_paths), video_path.stem)
        out_path = out_dir / f'{video_path.stem}.parquet'
        if out_path.exists() and not overwrite:
            _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}: exists, skipping')
            continue
        _logger.info(f'[{idx + 1}/{len(video_paths)}] {video_path.stem}')
        probs, feats = classify_video(
            video_path, model, captured, device, batch_size, save_features, position=shard,
        )
        n_expected = video_frame_count(video_path)
        if len(probs) != n_expected:
            _logger.warning(
                f'{video_path.stem}: decoded {len(probs)} frames but the container reports '
                f'{n_expected}'
            )
        if feats is not None:
            save_npy_atomic(out_dir / f'{video_path.stem}_features.npy', feats)
        # parquet last: its existence marks the video as complete
        save_parquet_atomic(out_path, build_predictions(probs, classes))
        written.append(out_path)
        share = pd.Series(np.array(classes)[probs.argmax(1)]).value_counts(normalize=True)
        _logger.info(
            f'{video_path.stem}: {len(probs)} frames; '
            + ', '.join(f'{k} {v:.2f}' for k, v in share.items())
        )
    write_progress(progress_file, len(video_paths), len(video_paths), None)
    return written
