"""Cut a side-by-side original/reconstruction QC clip (and matching gif) for one BEAST session.

Randomly samples `--n-frames` anchor frames from `{session}/selected_frames.csv` under
`results_dir/beast_frames` (the anchor frames BEAST was trained/predicted on, excluding the
+/-1 context frames also present in that directory), looks each one up in the
`prediction_metadata.yaml` written by `beast predict` (`{model_dir}/image_predictions/{session}/`)
to find its reconstruction, and stacks the two side by side with a label above each: original on
the left, reconstruction (resized to the original's shape, undoing the model's fixed input size)
on the right. Frames are played back in chronological order regardless of sampling order. Writes
the result plus a matching gif to
`results_dir/beast_frames_qc/reconstructions/{model_dir.name}/`.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI; run directly, e.g.:

    python scripts/make_reconstruction_clip.py --session-name Day1_Tank2_Cuttle1_Resident_Crop \
        --model-dir /media/mattw/CUTTLE/results/beast_models/iter-1.1_resnet-18_d16
"""

import argparse
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from cuttle_patterns import paths
from cuttle_patterns.config import load_config
from cuttle_patterns.visualization.video_utils import build_gif_command, open_ffmpeg_raw_writer

DEFAULT_N_FRAMES = 200
DEFAULT_FPS = 4
DEFAULT_TARGET_HEIGHT = 300

LABEL_LEFT = 'original'
LABEL_RIGHT = 'reconstruction'
LABEL_BAND_HEIGHT = 40
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.8
LABEL_COLOR_BGR = (255, 255, 255)
LABEL_THICKNESS = 2


def load_frame_lookup(metadata_path: Path) -> dict[str, tuple[Path, Path]]:
    """Map frame filename to (original_path, reconstruction_path) from `beast predict` metadata.

    Args:
        metadata_path: path to a `prediction_metadata.yaml` written by `beast predict`.

    Returns:
        dict from frame filename (e.g. 'img00000519.png') to (original_path, reconstruction_path).
    """
    with metadata_path.open() as f:
        entries = yaml.safe_load(f)

    return {
        Path(entry['original_path']).name: (
            Path(entry['original_path']), Path(entry['reconstruction_path']),
        )
        for entry in entries
    }


def select_frame_filenames(
    selected_frames_path: Path,
    n_frames: int,
    seed: int,
) -> list[str]:
    """Randomly sample anchor frame filenames, returned in chronological order.

    Args:
        selected_frames_path: path to a `selected_frames.csv` written by `cuttle extract`
            (one anchor frame filename per line).
        n_frames: number of frames to sample; if fewer are available, all of them are used.
        seed: random seed for reproducible sampling.

    Returns:
        sorted list of sampled frame filenames.
    """
    filenames = [
        line.strip() for line in selected_frames_path.read_text().splitlines() if line.strip()
    ]
    n_select = min(n_frames, len(filenames))
    return sorted(random.Random(seed).sample(filenames, n_select))


def load_bgr(path: Path) -> np.ndarray:
    """Read an image as a BGR array.

    Args:
        path: path to the image file.

    Returns:
        the image as a BGR uint8 array.

    Raises:
        OSError: if the image cannot be read.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f'could not read image: {path}')
    return image


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize an image to a target height, preserving aspect ratio.

    Args:
        image: a BGR image array.
        target_height: desired output height, in pixels.

    Returns:
        the resized image.
    """
    height, width = image.shape[:2]
    target_width = max(1, round(width * target_height / height))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def build_label_band(width: int, text: str) -> np.ndarray:
    """Build a black band with centered white text, to place above an image panel.

    Args:
        width: band width, in pixels; should match the panel it sits above.
        text: label text.

    Returns:
        a BGR band array of shape (LABEL_BAND_HEIGHT, width, 3).
    """
    band = np.zeros((LABEL_BAND_HEIGHT, width, 3), dtype=np.uint8)
    (text_width, text_height), _ = cv2.getTextSize(
        text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS,
    )
    x = max(0, (width - text_width) // 2)
    y = (LABEL_BAND_HEIGHT + text_height) // 2
    cv2.putText(
        band, text, (x, y), LABEL_FONT, LABEL_FONT_SCALE, LABEL_COLOR_BGR, LABEL_THICKNESS,
        cv2.LINE_AA,
    )
    return band


def build_comparison_frame(
    original_bgr: np.ndarray,
    reconstruction_bgr: np.ndarray,
    target_height: int,
) -> np.ndarray:
    """Stack an original frame and its reconstruction side by side, with labels above each.

    Args:
        original_bgr: the training frame, as a BGR array.
        reconstruction_bgr: the model's reconstruction of that frame, as a BGR array.
        target_height: height (px) each panel is resized to before stacking.

    Returns:
        a single BGR frame: labels on top, original on the left, reconstruction on the right.
    """
    original_height, original_width = original_bgr.shape[:2]
    reconstruction_bgr = cv2.resize(
        reconstruction_bgr, (original_width, original_height), interpolation=cv2.INTER_CUBIC,
    )

    left = resize_to_height(original_bgr, target_height)
    right = resize_to_height(reconstruction_bgr, target_height)
    labels = np.hstack([
        build_label_band(left.shape[1], LABEL_LEFT),
        build_label_band(right.shape[1], LABEL_RIGHT),
    ])
    panels = np.hstack([left, right])
    return np.vstack([labels, panels])


def main() -> None:
    """Parse arguments and write the sampled original/reconstruction clip and matching gif."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--session-name',
        required=True,
        help='beast_frames session stem, e.g. Day1_Tank2_Cuttle1_Resident_Crop',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='model directory, as passed to cuttle train/predict, e.g. '
        'results_dir/beast_models/{model_name}',
    )
    parser.add_argument(
        '--n-frames', '-n',
        type=int,
        default=DEFAULT_N_FRAMES,
        help='number of anchor frames to randomly sample for the clip',
    )
    parser.add_argument('--seed', type=int, default=0, help='random seed for frame sampling')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help='output clip frame rate')
    parser.add_argument(
        '--target-height',
        type=int,
        default=DEFAULT_TARGET_HEIGHT,
        help='height (px) each panel is resized to before stacking',
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=None,
        help='defaults to results_dir from the cuttle config',
    )
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir is not None else load_config().results_dir

    selected_frames_path = (
        results_dir / paths.BEAST_FRAMES_RELPATH / args.session_name / 'selected_frames.csv'
    )
    if not selected_frames_path.exists():
        raise FileNotFoundError(f'no selected_frames.csv found at {selected_frames_path}')

    metadata_path = (
        args.model_dir / 'image_predictions' / args.session_name / 'prediction_metadata.yaml'
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f'no prediction_metadata.yaml found at {metadata_path}; run cuttle predict '
            f'--save-reconstructions first'
        )

    frame_lookup = load_frame_lookup(metadata_path)
    filenames = select_frame_filenames(selected_frames_path, args.n_frames, args.seed)

    missing = [filename for filename in filenames if filename not in frame_lookup]
    if missing:
        raise ValueError(
            f'{len(missing)} of {len(filenames)} sampled frames have no reconstruction in '
            f'{metadata_path}, e.g. {missing[:5]}'
        )

    output_dir = (
        results_dir / paths.BEAST_FRAMES_QC_RECONSTRUCTIONS_RELPATH / args.model_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f'{args.session_name}.mp4'
    gif_path = mp4_path.with_suffix('.gif')

    writer = None
    try:
        for filename in tqdm(filenames, desc=f'{args.session_name} ({args.model_dir.name})'):
            original_path, reconstruction_path = frame_lookup[filename]
            frame = build_comparison_frame(
                load_bgr(original_path), load_bgr(reconstruction_path), args.target_height,
            )
            if writer is None:
                height, width = frame.shape[:2]
                writer = open_ffmpeg_raw_writer(mp4_path, width, height, args.fps)
            writer.stdin.write(frame.tobytes())
    finally:
        if writer is not None:
            writer.stdin.close()
            writer.wait()

    if writer is None or writer.returncode != 0:
        raise RuntimeError(f'ffmpeg failed writing {mp4_path}')
    print(f'wrote {mp4_path}')

    gif_cmd = build_gif_command(mp4_path, gif_path)
    subprocess.run(gif_cmd, check=True)
    print(f'wrote {gif_path}')


if __name__ == '__main__':
    main()
