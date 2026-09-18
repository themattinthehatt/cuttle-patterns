"""Run a collaborator-trained skin-pattern classifier over every exported extraction frame.

Loads a ResNet18 classifier (a plain `state_dict`, plus a `behaviors.json` giving its output
classes in order) shared by a collaborator working in parallel on supervised pattern
classification, and runs it over every frame under `results_dir/beast_frames/{video_name}/`
(the full set `cuttle extract` exported and `cuttle predict`/`cuttle reduce`/`cuttle cluster`
already operate on — anchors and their +/-1 context neighbors, not just the anchors listed in
each video's `selected_frames.csv`), so the output lines up frame-for-frame with any existing
`reduce`/`cluster` parquet.

Writes one row per frame to `results_dir/classifications/{name}.parquet` (`{name}` defaults to
the weights file's stem) with the hard predicted label plus the full per-class probability
vector, not just argmax -- so the visualizer (`cuttle serve`) can color points by e.g.
P(Leopard) as a continuous gradient rather than only a categorical label, useful for spotting
frames that blend two patterns rather than falling cleanly into one.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI (see the "scripts/ directory" entry
in docs/DECISIONS.md); run directly, e.g.:

    python scripts/classify_skin_pattern.py \
        --weights /path/to/2026-09-09_classifier_model/skin_pattern_resnet18_best.pt \
        --behaviors /path/to/2026-09-09_classifier_model/behaviors.json

Requires: torch, torchvision (both already pulled in transitively by the required
beast-backbones install -- see README.md's Setup section).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm import tqdm

from cuttle_patterns import paths
from cuttle_patterns.config import load_config
from cuttle_patterns.embeddings import FRAME_FILENAME_PATTERN

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


def predict_frame_classes(
    model: nn.Module,
    frame_paths: list[Path],
    classes: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    """Run the classifier over every frame and build a per-frame predictions table.

    Args:
        model: the loaded classifier, in eval mode.
        frame_paths: frame image paths to classify.
        classes: class names, in the model's output-index order.
        device: device to run inference on.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        one row per frame, with columns `video_name`, `frame_number`,
        `predicted_pattern`, `confidence`, `margin_top1_minus_top2`, and one
        `prob_{class_name}` column per class (spaces replaced with underscores), sorted
        by `(video_name, frame_number)` to match
        `cuttle_patterns.embeddings.load_latents`'s row order.
    """
    loader = DataLoader(
        FrameDataset(frame_paths), batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    rows = []
    with torch.no_grad():
        for images, image_paths in tqdm(loader, desc='classifying frames'):
            logits = model(images.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            for image_path, frame_probs in zip(image_paths, probs, strict=True):
                frame_path = Path(image_path)
                frame_match = FRAME_FILENAME_PATTERN.match(frame_path.stem)
                top1_idx = int(frame_probs.argmax())
                sorted_probs = sorted(frame_probs, reverse=True)
                row = {
                    'video_name': frame_path.parent.name,
                    'frame_number': int(frame_match['frame_number']),
                    'predicted_pattern': classes[top1_idx],
                    'confidence': float(sorted_probs[0]),
                    'margin_top1_minus_top2': float(sorted_probs[0] - sorted_probs[1]),
                }
                for class_name, prob in zip(classes, frame_probs, strict=True):
                    safe_class_name = class_name.replace(' ', '_')
                    row[f'prob_{safe_class_name}'] = float(prob)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(['video_name', 'frame_number']).reset_index(drop=True)


def main() -> None:
    """Parse arguments, run the classifier over every exported frame, and write predictions."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        '--name',
        default=None,
        help='name for the output parquet file; written to '
        'results_dir/classifications/{name}.parquet (defaults to --weights\' stem)',
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
    name = args.name if args.name is not None else args.weights.stem

    with args.behaviors.open() as f:
        classes = json.load(f)
    print(f'device: {device}')
    print(f'classes ({len(classes)}): {classes}')

    frame_paths = find_frame_paths(input_dir)
    print(f'found {len(frame_paths)} frames under {input_dir}')

    model = load_classifier(args.weights, len(classes), device)
    df = predict_frame_classes(
        model, frame_paths, classes, device, args.batch_size, args.num_workers,
    )

    output_dir = results_dir / paths.CLASSIFICATIONS_RELPATH
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{name}.parquet'
    if output_path.exists():
        print(f'warning: overwriting existing {output_path}')
    df.to_parquet(output_path, index=False)
    print(f'wrote {len(df)} rows to {output_path}')

    print('\npredicted-pattern counts:')
    print(df['predicted_pattern'].value_counts().to_string())


if __name__ == '__main__':
    main()
