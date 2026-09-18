"""Sample random frames from a raw video and visualize Phase 2a inscribed rectangles.

Not part of the installed package — rerun anytime to sanity-check
`cuttle_patterns.preprocessing.inscribe` against real data. Results are written under
`results_dir`, outside git.

Usage:
    python scratch/run_inscribe_v1.py [--video-path PATH] [--n-frames N] [--seed N]
        [--version NAME]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from cuttle_patterns.config import load_config
from cuttle_patterns.ingest import find_raw_videos, read_blank_frame_indices
from cuttle_patterns.preprocessing.inscribe import InscribedRect, inscribe_rectangle

DEFAULT_SEED = 0
DEFAULT_N_FRAMES = 20
DEFAULT_VERSION = 'v1'
RECT_COLOR_BGR = (0, 255, 0)


def sample_frame_indices(
    n_total: int,
    blank_indices: set[int],
    n_samples: int,
    seed: int,
) -> list[int]:
    """Randomly sample frame indices, excluding known-blank frames.

    Args:
        n_total: total number of frames in the video.
        blank_indices: frame indices flagged blank (no fish visible).
        n_samples: number of indices to sample.
        seed: random seed, for reproducibility.

    Returns:
        sorted list of sampled frame indices.
    """
    candidates = [idx for idx in range(n_total) if idx not in blank_indices]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=min(n_samples, len(candidates)), replace=False)
    return sorted(int(idx) for idx in chosen)


def draw_inscribed_rectangle(frame: np.ndarray, result: InscribedRect | None) -> np.ndarray:
    """Draw an inscribed rectangle's corners on a copy of the frame, in green.

    Args:
        frame: decoded video frame (BGR).
        result: inscribed rectangle to draw, or None if no body was detected.

    Returns:
        a copy of the frame with the rectangle drawn (unchanged if result is None).
    """
    canvas = frame.copy()
    if result is not None:
        poly = result.corners.astype(np.int32)
        cv2.polylines(canvas, [poly], isClosed=True, color=RECT_COLOR_BGR, thickness=2)
    return canvas


def main() -> None:
    """Sample frames from a raw video and write inscribed-rectangle visualizations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--video-path',
        type=Path,
        default=None,
        help='video to sample from; defaults to the first raw video found in data_dir',
    )
    parser.add_argument('--n-frames', type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--version',
        default=DEFAULT_VERSION,
        help='output subfolder under {results_dir}/inscribe/, for comparing pipeline versions',
    )
    args = parser.parse_args()

    config = load_config()

    video_path = args.video_path
    if video_path is None:
        videos = find_raw_videos(config.data_dir)
        if not videos:
            raise FileNotFoundError(f'no raw videos found in {config.data_dir}')
        video_path = videos[0]

    blank_frames_path = video_path.with_suffix('.txt')
    blank_indices = (
        set(read_blank_frame_indices(blank_frames_path)) if blank_frames_path.exists() else set()
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(n_total, blank_indices, args.n_frames, args.seed)
    print(f'sampling {len(frame_indices)} frames from {video_path} ({n_total} total frames)')

    output_dir = config.results_dir / 'inscribe' / args.version
    output_dir.mkdir(parents=True, exist_ok=True)

    n_no_body = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f'could not read frame {idx}, skipping')
            continue

        result = inscribe_rectangle(frame)
        if result is None:
            n_no_body += 1

        canvas = draw_inscribed_rectangle(frame, result)
        image_name = f'{video_path.stem}_frame-{idx:06d}'
        out_path = output_dir / f'{image_name}.png'
        cv2.imwrite(str(out_path), canvas)

    cap.release()
    print(f'wrote {len(frame_indices)} images to {output_dir}')
    print(f'{n_no_body} frame(s) had no detected body')


if __name__ == '__main__':
    main()
