"""Run a collaborator-trained skin-pattern classifier over every exported extraction frame.

Loads a ResNet18 classifier (a plain `state_dict`, plus a `behaviors.json` giving its output
classes in order) shared by a collaborator working in parallel on supervised pattern
classification, and runs it once over every frame under `results_dir/beast_frames/{video_name}/`
(the full set `cuttle extract` exported and `cuttle predict`/`cuttle reduce`/`cuttle cluster`
already operate on -- anchors and their +/-1 context neighbors, not just the anchors listed in
each video's `selected_frames.csv`), producing two outputs from that single pass:

1. Predicted labels: one row per frame to `results_dir/classifications/{model_name}.parquet`
   with the hard predicted label plus the full per-class probability vector, not just argmax --
   so the visualizer (`cuttle serve`) can color points by e.g. P(Leopard) as a continuous
   gradient rather than only a categorical label, useful for spotting frames that blend two
   patterns rather than falling cleanly into one. Model-independent, like `cuttle_patterns/
   paths.py`'s `CLASSIFICATIONS_RELPATH` docstring explains -- attachable to any model's plot.

2. Embeddings: the classifier's own 512-d penultimate (post-avgpool, pre-fc) activation per
   frame, written to `results_dir/beast_models/{model_name}/image_predictions/{predictions_name}/
   latents/` -- exactly the layout `beast predict --save-latents` writes, plus a minimal
   `config.yaml` -- so `{model_name}` shows up as an ordinary model everywhere downstream
   (`cuttle reduce --model-name {model_name}`, `cuttle cluster --model-name {model_name}
   --n-clusters K`, and the `cuttle serve` Model dropdown) with no code changes anywhere else.
   Pick a `--model-name` that reads as clearly non-BEAST at a glance, e.g.
   `iter-1.1_classifier_d512` (BEAST models here are named `iter-1.1_resnet-18_d16` /
   `iter-1.1_msps-vae_d16`) -- see the "Classifier embeddings" entry in docs/DECISIONS.md.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI (see the "scripts/ directory" entry
in docs/DECISIONS.md); run directly, e.g.:

    python scripts/classify_skin_pattern.py --model-name iter-1.1_classifier_d512 \
        --weights /path/to/2026-09-09_classifier_model/skin_pattern_resnet18_best.pt \
        --behaviors /path/to/2026-09-09_classifier_model/behaviors.json
    cuttle reduce --model-name iter-1.1_classifier_d512
    cuttle cluster --model-name iter-1.1_classifier_d512 --n-clusters 10  # optional

Requires: torch, torchvision (both already pulled in transitively by the required
beast-backbones install -- see README.md's Setup section).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm import tqdm

from cuttle_patterns import paths
from cuttle_patterns.config import load_config
from cuttle_patterns.latents import FRAME_FILENAME_PATTERN

DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 2

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# recorded in the pseudo-model's config.yaml; anything other than embeddings.MSPS_VAE_MODEL_CLASS
# makes split_latent_spaces treat this as an ordinary single-latent-space model
MODEL_CLASS = 'classifier'


class FrameDataset(Dataset):
    """Minimal dataset over a flat list of frame image paths, no labels."""

    def __init__(self, frame_paths: list[Path]):
        """Store the frame paths to classify.

        Args:
            frame_paths: paths to classify, in the order they'll be iterated.
        """
        self.frame_paths = frame_paths

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image = Image.open(self.frame_paths[idx]).convert('RGB')
        return TRANSFORM(image), str(self.frame_paths[idx])


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


def load_classifier(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load the skin-pattern ResNet18 classifier from its saved state_dict.

    Args:
        weights_path: path to a plain `state_dict` (not a full pickled model) -- the
            same resnet18-plus-fresh-final-layer architecture used at training time must
            be reconstructed here first, then the weights loaded into it.
        num_classes: number of output classes, i.e. `len(classes)`.
        device: device to move the model to.

    Returns:
        the loaded model, in eval mode.
    """
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def register_feature_hook(model: nn.Module) -> dict[str, torch.Tensor]:
    """Register a forward hook capturing a resnet18's 512-d penultimate activation.

    Hooks `model.avgpool` (the layer immediately before `model.fc`) rather than
    reimplementing `forward`, so the exact same `model(images)` call already used for
    the classifier's own predictions also yields its penultimate features for free --
    no second forward pass needed to get both.

    Args:
        model: the loaded classifier (a torchvision resnet18).

    Returns:
        a dict that gets a fresh `'value'` entry (the flattened `(batch, 512)` tensor)
        every time `model(...)` is called; read it right after each call.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        captured['value'] = torch.flatten(output, 1)

    model.avgpool.register_forward_hook(hook)
    return captured


def run_classifier(
    model: nn.Module,
    frame_paths: list[Path],
    classes: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Run the classifier once over every frame, returning predictions and features.

    Captures each frame's 512-d penultimate activation (via `register_feature_hook`)
    alongside its softmax predictions, so one forward pass per frame produces both.

    Args:
        model: the loaded classifier, in eval mode.
        frame_paths: frame image paths to classify.
        classes: class names, in the model's output-index order.
        device: device to run inference on.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        `(predictions_df, embeddings, embedding_meta)`:
        - `predictions_df`: one row per frame, columns `video_name`, `frame_number`,
          `predicted_pattern`, `confidence`, and one `prob_{class_name}` column per
          class (spaces replaced with underscores).
        - `embeddings`: float32 array, shape `(n_frames, 512)`.
        - `embedding_meta`: columns `video_name`, `frame_number`, row-aligned with
          `embeddings`.
        All three are sorted by `(video_name, frame_number)`, matching
        `cuttle_patterns.latents.load_latents`'s row order.
    """
    loader = DataLoader(
        FrameDataset(frame_paths), batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    captured_features = register_feature_hook(model)

    prediction_rows = []
    embedding_rows = []
    embedding_vectors = []
    with torch.no_grad():
        for images, image_paths in tqdm(loader, desc='classifying frames'):
            logits = model(images.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            features = captured_features['value'].cpu().numpy()

            for image_path, frame_probs, frame_features in zip(
                image_paths, probs, features, strict=True,
            ):
                frame_path = Path(image_path)
                frame_match = FRAME_FILENAME_PATTERN.match(frame_path.stem)
                video_name = frame_path.parent.name
                frame_number = int(frame_match['frame_number'])

                top1_idx = int(frame_probs.argmax())
                row = {
                    'video_name': video_name,
                    'frame_number': frame_number,
                    'predicted_pattern': classes[top1_idx],
                    'confidence': float(frame_probs[top1_idx]),
                }
                for class_name, prob in zip(classes, frame_probs, strict=True):
                    safe_class_name = class_name.replace(' ', '_')
                    row[f'prob_{safe_class_name}'] = float(prob)
                prediction_rows.append(row)

                embedding_rows.append({'video_name': video_name, 'frame_number': frame_number})
                embedding_vectors.append(frame_features)

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df = predictions_df.sort_values(
        ['video_name', 'frame_number'],
    ).reset_index(drop=True)

    embedding_meta = pd.DataFrame(embedding_rows)
    order = embedding_meta.sort_values(['video_name', 'frame_number']).index.to_numpy()
    embedding_meta = embedding_meta.iloc[order].reset_index(drop=True)
    embeddings = np.stack(embedding_vectors)[order]

    return predictions_df, embeddings, embedding_meta


def write_latents(embeddings: np.ndarray, embedding_meta: pd.DataFrame, latents_dir: Path) -> None:
    """Write one `.npy` file per frame, matching `beast predict --save-latents`'s layout.

    This is what lets `cuttle reduce`/`cuttle cluster` run against this classifier's
    embeddings unmodified, via `cuttle_patterns.latents.load_latents` -- the same
    function that reads a real BEAST model's saved latents.

    Args:
        embeddings: float32 array, shape `(n_frames, feature_dim)`.
        embedding_meta: row-aligned metadata with columns `video_name`, `frame_number`.
        latents_dir: `.../image_predictions/{predictions_name}/latents` directory to
            write into (one subdirectory per video, created as needed).
    """
    for video_name, frame_number, vector in zip(
        embedding_meta['video_name'], embedding_meta['frame_number'], embeddings, strict=True,
    ):
        video_dir = latents_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)
        np.save(video_dir / f'img{frame_number:08d}.npy', vector)


def write_model_config(model_dir: Path) -> None:
    """Write a minimal config.yaml so this directory duck-types as a BEAST model dir.

    `cuttle_patterns.latents.split_latent_spaces` (used by `cuttle reduce`/`cuttle
    cluster`) reads `model.model_class` from a real BEAST model's config.yaml to decide
    whether to split the latent space -- only for `msps_vae`. Recording `MODEL_CLASS`
    here takes that same single-latent-space path, so this directory needs no other
    special-casing to work with either downstream command.

    Args:
        model_dir: `results_dir/beast_models/{model_name}` to write into; created if it
            doesn't already exist.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / 'config.yaml'
    with config_path.open('w') as f:
        yaml.safe_dump({'model': {'model_class': MODEL_CLASS}}, f)


def main() -> None:
    """Parse arguments, run the classifier once, and write predictions plus embeddings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help='name for this classifier, used both as a pseudo BEAST-model directory '
        '(results_dir/beast_models/{model_name}/, holding embeddings for cuttle '
        'reduce/cuttle cluster) and as the predictions filename '
        '(results_dir/classifications/{model_name}.parquet); pick something that reads '
        'as non-BEAST at a glance, e.g. iter-1.1_classifier_d512',
    )
    parser.add_argument(
        '--weights',
        type=Path,
        required=True,
        help='path to the classifier weights, a plain state_dict (e.g. '
        'skin_pattern_resnet18_best.pt)',
    )
    parser.add_argument(
        '--behaviors',
        type=Path,
        required=True,
        help='path to the behaviors.json class-order file matching --weights',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=None,
        help='directory of per-video frame subdirectories to classify (defaults to '
        'results_dir/beast_frames, i.e. every frame cuttle extract exported)',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='defaults to results_dir from the cuttle config',
    )
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--device', default=None, help='defaults to cuda if available, else cpu')
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir is not None else load_config().results_dir
    input_dir = (
        args.input_dir if args.input_dir is not None
        else results_dir / paths.BEAST_FRAMES_RELPATH
    )
    device = torch.device(
        args.device if args.device is not None
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    with args.behaviors.open() as f:
        classes = json.load(f)
    print(f'device: {device}')
    print(f'classes ({len(classes)}): {classes}')

    frame_paths = find_frame_paths(input_dir)
    print(f'found {len(frame_paths)} frames under {input_dir}')

    model = load_classifier(args.weights, len(classes), device)
    predictions_df, embeddings, embedding_meta = run_classifier(
        model, frame_paths, classes, device, args.batch_size, args.num_workers,
    )

    classifications_dir = results_dir / paths.CLASSIFICATIONS_RELPATH
    classifications_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = classifications_dir / f'{args.model_name}.parquet'
    if predictions_path.exists():
        print(f'warning: overwriting existing {predictions_path}')
    predictions_df.to_parquet(predictions_path, index=False)
    print(f'wrote {len(predictions_df)} rows to {predictions_path}')

    model_dir = results_dir / paths.BEAST_MODELS_RELPATH / args.model_name
    predictions_name = input_dir.name
    latents_dir = model_dir / 'image_predictions' / predictions_name / 'latents'
    write_model_config(model_dir)
    write_latents(embeddings, embedding_meta, latents_dir)
    print(
        f'wrote {len(embedding_meta)} embeddings ({embeddings.shape[1]}-d) to {latents_dir}\n'
        f'next: cuttle reduce --model-name {args.model_name} (and optionally cuttle cluster '
        f'--model-name {args.model_name} --n-clusters K)'
    )

    print('\npredicted-pattern counts:')
    print(predictions_df['predicted_pattern'].value_counts().to_string())


if __name__ == '__main__':
    main()
