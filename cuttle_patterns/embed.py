"""Run a frozen pretrained embedder over exported frames (`cuttle embed`).

Unlike `cuttle predict --save-latents` (one `.npy` per frame), this writes a single
combined `(N, D)` array plus a row-alignment manifest -- writing millions of tiny files
is extremely slow on the external drive `results_dir` currently lives on. Output is
duck-typed as a BEAST model directory (same precedent as the classifier's own embedding,
see `docs/DECISIONS.md`'s "Classifier embeddings" entry), so `cuttle reduce`/`cuttle
cluster`/`cuttle serve` need no embedder-specific code -- `cuttle_patterns.latents.load_latents`
reads this layout via its combined-format branch. See
`docs/implementation_notes/embedder.md` for the full embedder design.
"""

import math
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cuttle_patterns.embedders.base import Embedder
from cuttle_patterns.embedders.dinov3 import ARCH_TO_HF_ID, DINOv3Backbone
from cuttle_patterns.embedders.readouts import (
    CLS_READOUT_NAME,
    GRAM_READOUT_NAME,
    READOUTS_BY_NAME,
    FusedGramReadout,
)
from cuttle_patterns.embedders.vgg import BACKBONE_NAME as VGG_BACKBONE_NAME
from cuttle_patterns.embedders.vgg import MultiLayerVGGBackbone, VGGBackbone
from cuttle_patterns.latents import FRAME_FILENAME_PATTERN, parse_video_name

DEFAULT_BATCH_SIZE = 128
DEFAULT_GRAM_K = 64
DEFAULT_GRAM_WEIGHTS = 'taper'
DEFAULT_VGG_LAYER = 'relu3_1'
# VGG is fully convolutional and cheap per-pixel relative to a ViT, so it can afford a
# higher default resolution than DINOv3's 224 -- a less noisy Gram covariance estimate,
# per "Gram readout"'s resolution caveat in docs/implementation_notes/embedder.md
DEFAULT_VGG_RESOLUTION = 448
# a stateful readout's fit set is a random, per-video-capped sample drawn directly from
# the frames being embedded -- see "Fitting stateful readouts" in
# docs/implementation_notes/embedder.md; fixed size/seed keep it deterministic, so
# refitting every `cuttle embed` run reproduces the same projection every time
DEFAULT_FIT_SET_SIZE = 4000
DEFAULT_FIT_SEED = 42
MODEL_CLASS = 'embedder'


def find_frame_paths(input_dir: Path) -> list[Path]:
    """Find every exported frame image under a `cuttle extract` frame-set directory.

    Args:
        input_dir: a directory of per-video subdirectories of `img{frame_number}.png`
            files, e.g. `results_dir/beast_frames` (`cuttle extract`'s output layout).

    Returns:
        sorted list of frame image paths, across every video subdirectory.

    Raises:
        FileNotFoundError: if input_dir does not exist.
        ValueError: if input_dir exists but contains no matching frame images.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f'input directory does not exist: {input_dir}')

    frame_paths = sorted(input_dir.glob('*/img*.png'))
    if not frame_paths:
        raise ValueError(f'no frame images found under {input_dir}')
    return frame_paths


def build_embedder(
    backbone_arch: str,
    resolution: int,
    readout_name: str,
    device: torch.device,
    vgg_layer: list[str] | None = None,
    readout_kwargs: dict | None = None,
) -> Embedder:
    """Build the one embedder a `cuttle embed` run uses.

    Validates `readout_name` (and the VGG/CLS and VGG-fusion incompatibilities below)
    before loading any weights, so an invalid combination fails fast rather than after a
    slow (network-dependent) backbone load.

    Args:
        backbone_arch: one of `cuttle_patterns.embedders.dinov3.ARCH_TO_HF_ID`'s keys,
            or `cuttle_patterns.embedders.vgg.BACKBONE_NAME` (`'vgg19'`).
        resolution: square input side length, in pixels; must be a multiple of 16 for a
            DINOv3 backbone, or of `vgg_layer`'s downsampling stride for `vgg19` (the
            deepest requested layer's, if more than one).
        readout_name: one of `cuttle_patterns.embedders.readouts.READOUTS_BY_NAME`'s
            keys.
        device: device to load the backbone onto.
        vgg_layer: one or more of `cuttle_patterns.embedders.vgg.VGG_LAYERS`' keys;
            ignored unless `backbone_arch == 'vgg19'`, where it defaults to
            `[DEFAULT_VGG_LAYER]`. More than one value (Gram fusion) requires
            `readout_name == 'gram'`.

    Returns:
        the composed embedder.

    Raises:
        ValueError: if `backbone_arch`, `readout_name`, or a `vgg_layer` entry is
            unrecognized; if `resolution` doesn't divide evenly for the chosen backbone;
            if `readout_name == 'cls'` with `backbone_arch == 'vgg19'`, which has no
            CLS-token equivalent; or if `vgg_layer` has more than one entry with
            `readout_name != 'gram'`.
    """
    if readout_name not in READOUTS_BY_NAME:
        raise ValueError(
            f'unknown readout: {readout_name!r}; choices: {list(READOUTS_BY_NAME)}'
        )

    if backbone_arch == VGG_BACKBONE_NAME:
        layers = vgg_layer if vgg_layer is not None else [DEFAULT_VGG_LAYER]
        if len(layers) > 1:
            if readout_name != GRAM_READOUT_NAME:
                raise ValueError(
                    f'--vgg-layer fusion (more than one layer) requires --readout '
                    f'{GRAM_READOUT_NAME}, got --readout {readout_name!r}'
                )
            backbone = MultiLayerVGGBackbone(layers, resolution, device)
            readout = FusedGramReadout(
                layer_dims=backbone.channels_by_layer, **(readout_kwargs or {}),
            )
            return Embedder(backbone, readout)
        if readout_name == CLS_READOUT_NAME:
            raise ValueError(
                f'{VGG_BACKBONE_NAME} has no CLS token; --readout {CLS_READOUT_NAME} '
                'is unsupported'
            )
        backbone = VGGBackbone(layers[0], resolution, device)
    elif backbone_arch in ARCH_TO_HF_ID:
        backbone = DINOv3Backbone(backbone_arch, resolution, device)
    else:
        raise ValueError(
            f'unknown backbone: {backbone_arch!r}; '
            f'choices: {[*ARCH_TO_HF_ID, VGG_BACKBONE_NAME]}'
        )

    readout = READOUTS_BY_NAME[readout_name](dim=backbone.embed_dim, **(readout_kwargs or {}))
    return Embedder(backbone, readout)


def _parse_frame_path(frame_path: Path) -> dict[str, str | int]:
    """Parse a frame image path into its video/frame identity.

    Args:
        frame_path: a path matching `{video_name}/img{frame_number}.png`.

    Returns:
        dict with keys `video_name`, `day`, `tank`, `role`, `frame_number`.

    Raises:
        ValueError: if the filename or parent directory name don't match the expected
            pattern.
    """
    frame_match = FRAME_FILENAME_PATTERN.match(frame_path.stem)
    if frame_match is None:
        raise ValueError(f'unexpected frame filename: {frame_path}')

    video_name = frame_path.parent.name
    row = parse_video_name(video_name)
    row['video_name'] = video_name
    row['frame_number'] = int(frame_match['frame_number'])
    return row


def _load_frames(frame_paths: list[Path]) -> np.ndarray:
    """Load a batch of frame images as canonical uint8 RGB crops.

    Args:
        frame_paths: frame image paths to load.

    Returns:
        uint8 array, shape (len(frame_paths), H, W, 3), channel order RGB.
    """
    return np.stack([
        cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frame_paths
    ])


def run_embed(
    embedder: Embedder,
    frame_paths: list[Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run one embedder over every frame, batch by batch.

    Args:
        embedder: the backbone+readout pair to run; if it `requires_fit`, it must
            already be fit (see `fit_readout`) before calling this.
        frame_paths: frame image paths to embed.
        batch_size: number of frames per forward pass.

    Returns:
        `(embeddings, meta)`: `embeddings` is a float32 array, shape
        `(len(frame_paths), embedder.dim)`; `meta` has columns `video_name`, `day`,
        `tank`, `role`, `frame_number`, row-aligned with `embeddings`, sorted by
        `(video_name, frame_number)` -- matching `cuttle_patterns.latents.load_latents`'s
        existing row order.
    """
    rows = [_parse_frame_path(frame_path) for frame_path in frame_paths]
    vectors = []

    for start in tqdm(range(0, len(frame_paths), batch_size), desc='embedding frames'):
        batch_paths = frame_paths[start:start + batch_size]
        vectors.append(embedder.embed(_load_frames(batch_paths)))

    embeddings = np.concatenate(vectors, axis=0)
    meta = pd.DataFrame(rows, columns=['video_name', 'day', 'tank', 'role', 'frame_number'])

    order = meta.sort_values(['video_name', 'frame_number']).index.to_numpy()
    return embeddings[order], meta.iloc[order].reset_index(drop=True)


def sample_fit_frame_paths(
    frame_paths: list[Path],
    fit_set_size: int = DEFAULT_FIT_SET_SIZE,
    seed: int = DEFAULT_FIT_SEED,
) -> list[Path]:
    """Sample a random, per-video-capped fit set for a stateful readout's fit passes.

    Args:
        frame_paths: candidate frame paths to sample from -- typically the same frames
            a `cuttle embed` run is about to embed.
        fit_set_size: approximate total number of frames to sample, spread evenly
            across videos.
        seed: random seed, so the fit set (and therefore the fitted readout) is
            reproducible for fixed inputs.

    Returns:
        sorted list of sampled frame paths, capped at
        `ceil(fit_set_size / n_videos)` per video (fewer if a video has fewer frames
        than the cap).
    """
    paths_by_video = defaultdict(list)
    for frame_path in frame_paths:
        paths_by_video[frame_path.parent.name].append(frame_path)

    rng = np.random.default_rng(seed)
    per_video_cap = math.ceil(fit_set_size / len(paths_by_video))
    sampled = []
    for video_paths in paths_by_video.values():
        n = min(per_video_cap, len(video_paths))
        idx = rng.choice(len(video_paths), size=n, replace=False)
        sampled.extend(video_paths[i] for i in idx)
    return sorted(sampled)


def fit_readout(
    embedder: Embedder,
    fit_frame_paths: list[Path],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Fit a stateful readout's parameters over a sampled fit set, in place.

    Runs `embedder.readout.fit_passes` passes over `fit_frame_paths`, calling
    `partial_fit` on every batch's backbone output and `finalize_pass` at the end of
    each pass -- see "Fitting stateful readouts" in
    `docs/implementation_notes/embedder.md`. A no-op if the readout is stateless
    (`embedder.requires_fit` is False).

    Args:
        embedder: the embedder whose readout to fit.
        fit_frame_paths: frame image paths making up the fit set (see
            `sample_fit_frame_paths`).
        batch_size: number of frames per forward pass.
    """
    readout = embedder.readout
    for pass_idx in range(readout.fit_passes):
        desc = f'fitting {readout.name} (pass {pass_idx + 1}/{readout.fit_passes})'
        for start in tqdm(range(0, len(fit_frame_paths), batch_size), desc=desc):
            batch_paths = fit_frame_paths[start:start + batch_size]
            frames = _load_frames(batch_paths)
            with torch.no_grad():
                tokens = embedder.backbone.forward(embedder.backbone.preprocess(frames))
            readout.partial_fit(tokens, pass_idx)
        readout.finalize_pass(pass_idx)


def _get_git_commit(repo_dir: Path) -> str | None:
    """Best-effort current commit hash, for provenance only.

    Args:
        repo_dir: directory inside the git repository to run `git rev-parse` from.

    Returns:
        the commit hash, or `None` if it can't be determined (e.g. `git` isn't
        installed, or `repo_dir` isn't inside a git repository).
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_dir, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_embedder_output(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    embedder: Embedder,
    model_dir: Path,
    predictions_name: str,
) -> Path:
    """Write a `cuttle embed` run's output, duck-typed as a BEAST model directory.

    Args:
        embeddings: float32 array, shape (n_frames, embedder.dim).
        meta: row-aligned metadata from `run_embed`.
        embedder: the embedder that produced `embeddings`, for provenance metadata.
        model_dir: `results_dir/beast_models/{model_name}`; created if needed.
        predictions_name: names the `image_predictions/{predictions_name}` subdirectory,
            matching `cuttle predict`'s own convention.

    Returns:
        the `latents/` directory written into.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model_params = {
        'backbone': embedder.backbone.key,
        'readout': embedder.readout.name,
        'embed_dim': embedder.dim,
        'code_git_commit': _get_git_commit(Path(__file__).resolve().parent),
        'extraction_timestamp': datetime.now(UTC).isoformat(),
        **embedder.backbone.metadata(),
        **embedder.readout.metadata(),
    }
    with (model_dir / 'config.yaml').open('w') as f:
        yaml.safe_dump({'model': {'model_class': MODEL_CLASS, 'model_params': model_params}}, f)

    latents_dir = model_dir / 'image_predictions' / predictions_name / 'latents'
    latents_dir.mkdir(parents=True, exist_ok=True)
    np.save(latents_dir / 'embeddings.npy', embeddings.astype(np.float32))
    meta.to_parquet(latents_dir / 'manifest.parquet', index=False)

    # a stateful readout (e.g. gram) is refit from scratch every run (see
    # DEFAULT_FIT_SET_SIZE/DEFAULT_FIT_SEED above), so this is a provenance record, not
    # a cache -- nothing in cuttle_patterns reads it back
    readout_state = embedder.readout.state_dict()
    if readout_state:
        torch.save(readout_state, model_dir / 'readout_state.pt')

    return latents_dir
