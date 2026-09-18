"""Find and visualize frames where the inscribed rectangle looks too small.

QC check for `cuttle extract`'s output: flags frames where both keypoints are
trustworthy (likelihood >= 0.9) but the rectangle's long edge is less than 50% of the
neck-tail distance — a body that's clearly longer than the box drawn around it. Reports
a per-video violation count and writes up to `--max-frames-per-video` example frames
(sampled from each video's `_overlay.mp4`, so both the box and keypoints are visible) to
`results_dir/beast_frames_qc/small_rectangles/`.

The underlying check now also runs as a `cuttle extract` filter — see
`cuttle_patterns.preprocessing.extract.compute_small_rectangle_mask` — so this script
just calls that directly rather than keeping its own copy of the logic; it's still useful
standalone for reporting per-video violation counts and dumping visual examples, neither
of which `cuttle extract` itself needs to do.

Usage:
    python scratch/qc_small_rectangles.py --pose-dir /path/to/pose/predictions
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.config import load_config
from cuttle_patterns.preprocessing.extract import (
    DEFAULT_SMALL_RECT_RATIO_THRESH,
    compute_small_rectangle_mask,
)
from cuttle_patterns.preprocessing.pose import DEFAULT_LIKELIHOOD_THRESH, load_pose_predictions

DEFAULT_RATIO_THRESH = DEFAULT_SMALL_RECT_RATIO_THRESH
DEFAULT_MAX_FRAMES_PER_VIDEO = 100
DEFAULT_SEED = 0
OUTPUT_RELPATH = Path('beast_frames_qc') / 'small_rectangles'


def find_small_rectangle_frames(
    rect_csv_path: Path,
    pose_csv_path: Path,
    ratio_thresh: float = DEFAULT_RATIO_THRESH,
    likelihood_thresh: float = DEFAULT_LIKELIHOOD_THRESH,
) -> np.ndarray:
    """Flag frames where the rectangle's long edge is too short relative to the body.

    Args:
        rect_csv_path: path to `cuttle inscribe`'s per-frame corner geometry CSV.
        pose_csv_path: path to the matching tail/neck pose predictions CSV.
        ratio_thresh: a frame is flagged if the rectangle's long edge is less than this
            fraction of the neck-tail distance.
        likelihood_thresh: only frames where both keypoints meet this likelihood are
            considered (a low-confidence prediction can't be trusted as ground truth).

    Returns:
        indices of flagged frames.

    Raises:
        ValueError: if the rectangle and pose CSVs don't have the same number of frames.
    """
    pose_df = load_pose_predictions(pose_csv_path)
    is_small = compute_small_rectangle_mask(
        rect_csv_path, pose_df, ratio_thresh=ratio_thresh, likelihood_thresh=likelihood_thresh,
    )
    return np.where(is_small)[0]


def export_example_frames(
    overlay_video_path: Path,
    frame_idxs: np.ndarray,
    output_dir: Path,
    video_name: str,
) -> None:
    """Export specific frames from an overlay video as PNGs.

    Args:
        overlay_video_path: path to `cuttle overlay`'s `{video_name}_overlay.mp4`.
        frame_idxs: indices of frames to export.
        output_dir: directory to write PNGs into.
        video_name: source video's stem, used as a filename prefix.

    Raises:
        OSError: if the overlay video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(overlay_video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {overlay_video_path}')

    try:
        for idx in sorted(frame_idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                print(f'  could not read frame {idx} from {overlay_video_path}, skipping')
                continue
            out_path = output_dir / f'{video_name}_frame-{idx:06d}.png'
            cv2.imwrite(str(out_path), frame)
    finally:
        cap.release()


def main() -> None:
    """Find and export example small-rectangle frames across all rectangle videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--pose-dir',
        type=Path,
        required=True,
        help='directory of {video_name}.csv tail/neck pose predictions',
    )
    parser.add_argument(
        '--rectangles-dir',
        type=Path,
        default=None,
        help='directory of cuttle inscribe/overlay output; defaults to results_dir/rectangles',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help=f'defaults to results_dir/{OUTPUT_RELPATH}',
    )
    parser.add_argument('--ratio-thresh', type=float, default=DEFAULT_RATIO_THRESH)
    parser.add_argument('--likelihood-thresh', type=float, default=DEFAULT_LIKELIHOOD_THRESH)
    parser.add_argument('--max-frames-per-video', type=int, default=DEFAULT_MAX_FRAMES_PER_VIDEO)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    config = load_config()
    rectangles_dir = args.rectangles_dir or config.results_dir / 'rectangles'
    output_dir = args.output_dir or config.results_dir / OUTPUT_RELPATH
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rect_csv_paths = sorted(rectangles_dir.glob('*.csv'))

    total_frames = 0
    total_violations = 0
    for rect_csv_path in rect_csv_paths:
        video_name = rect_csv_path.stem
        pose_csv_path = args.pose_dir / f'{video_name}.csv'
        if not pose_csv_path.exists():
            print(f'{video_name}: no pose predictions at {pose_csv_path}, skipping')
            continue

        n_frames = len(pd.read_csv(rect_csv_path))
        violation_idxs = find_small_rectangle_frames(
            rect_csv_path, pose_csv_path,
            ratio_thresh=args.ratio_thresh, likelihood_thresh=args.likelihood_thresh,
        )
        total_frames += n_frames
        total_violations += len(violation_idxs)
        print(
            f'{video_name}: {len(violation_idxs)}/{n_frames} frames '
            f'({100 * len(violation_idxs) / n_frames:.1f}%) have a rectangle long edge '
            f'< {args.ratio_thresh:.0%} of the neck-tail distance'
        )
        if len(violation_idxs) == 0:
            continue

        sample_idxs = violation_idxs
        if len(sample_idxs) > args.max_frames_per_video:
            sample_idxs = rng.choice(sample_idxs, size=args.max_frames_per_video, replace=False)

        overlay_video_path = rectangles_dir / f'{video_name}_overlay.mp4'
        if not overlay_video_path.exists():
            print(f'  no overlay video at {overlay_video_path}, skipping example export')
            continue
        export_example_frames(overlay_video_path, sample_idxs, output_dir, video_name)
        print(f'  wrote {len(sample_idxs)} example frames to {output_dir}')

    print(
        f'\ntotal: {total_violations}/{total_frames} frames '
        f'({100 * total_violations / total_frames:.1f}%) flagged across '
        f'{len(rect_csv_paths)} videos'
    )


if __name__ == '__main__':
    main()
