"""Select candidate frames for further pose-estimation labeling, ranked by cross-model
prediction disagreement.

For every raw video, loads per-frame tail/neck predictions from each of several trained
pose models (see docs/pose_estimation.md), scores each frame by how much the models
disagree with each other (summed x/y variance per keypoint, maxed over keypoints),
excludes frames that are already blank-flagged or already labeled, and writes QC images
for the highest-disagreement frames -- one folder per video under --output-dir -- so they
can be paged through and manually added to the labeling queue.

Not yet promoted into cuttle_patterns/ + the `cuttle` CLI; run directly, e.g.:

    python scripts/pose_plot_outliers.py \
        --models vits-dino_seed-0 vits-dino_seed-1 vits-dino_seed-2
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.config import load_config
from cuttle_patterns.ingest import build_manifest, read_blank_frame_indices
from cuttle_patterns.preprocessing.pose import KEYPOINTS, load_pose_predictions

DEFAULT_PROJECT_DIR = Path('/media/mattw/CUTTLE/pose-estimation/cuttle-test')
DEFAULT_TOP_K = 100

# cycled by position in --models, so a given model keeps the same color across every
# video's QC images regardless of which subset of models actually has predictions there
MODEL_COLORS_BGR = [
    (60, 60, 255),   # red
    (255, 120, 60),  # blue
    (60, 200, 60),   # green
    (60, 220, 255),  # yellow
    (255, 60, 255),  # magenta
]
KEYPOINT_SHAPES = {'tail': 'circle', 'neck': 'square'}
KEYPOINT_MARKER_SIZE = 7
KEYPOINT_MARKER_THICKNESS = 2


def load_labeled_frame_indices(collected_data_path: Path) -> dict[str, set[int]]:
    """Parse the Lightning Pose project's CollectedData.csv into labeled frame indices.

    Args:
        collected_data_path: path to the project's top-level CollectedData.csv.

    Returns:
        video_name -> set of already-labeled frame indices. Empty dict if
        collected_data_path does not exist yet.
    """
    if not collected_data_path.exists():
        return {}

    frame_paths = pd.read_csv(collected_data_path, header=None, skiprows=3, usecols=[0])[0]

    labeled_indices: dict[str, set[int]] = {}
    for frame_path in frame_paths:
        video_name = Path(frame_path).parent.name
        frame_idx = int(Path(frame_path).stem.removeprefix('img'))
        labeled_indices.setdefault(video_name, set()).add(frame_idx)

    return labeled_indices


def load_model_predictions(
    project_dir: Path,
    model_names: list[str],
    video_name: str,
) -> dict[str, pd.DataFrame]:
    """Load per-model tail/neck predictions for one video, skipping missing models.

    Args:
        project_dir: Lightning Pose project root (contains models/{model_name}/...).
        model_names: model directory names under project_dir/models.
        video_name: video stem to look up under each model's video_preds/.

    Returns:
        model_name -> tidy prediction frame (see pose.load_pose_predictions), for
        whichever models have a video_preds CSV for this video.
    """
    predictions = {}
    for model_name in model_names:
        csv_path = project_dir / 'models' / model_name / 'video_preds' / f'{video_name}.csv'
        if not csv_path.exists():
            continue
        predictions[model_name] = load_pose_predictions(csv_path)

    return predictions


def compute_variance_scores(predictions: dict[str, pd.DataFrame]) -> np.ndarray:
    """Score each frame by cross-model keypoint-prediction disagreement.

    Per keypoint, sums the variance (across models) of the x and y coordinates; a
    frame's score is the max of that sum over keypoints.

    Args:
        predictions: model_name -> tidy prediction frame, all the same length (one row
            per video frame).

    Returns:
        one score per frame.

    Raises:
        ValueError: if the loaded predictions don't all have the same number of frames.
    """
    lengths = {model_name: len(df) for model_name, df in predictions.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f'prediction length mismatch across models: {lengths}')

    keypoint_scores = []
    for keypoint in KEYPOINTS:
        x_stack = np.stack([df[f'{keypoint}_x'].to_numpy() for df in predictions.values()])
        y_stack = np.stack([df[f'{keypoint}_y'].to_numpy() for df in predictions.values()])
        keypoint_scores.append(x_stack.var(axis=0) + y_stack.var(axis=0))

    return np.max(np.stack(keypoint_scores), axis=0)


def select_top_frames(
    scores: np.ndarray,
    excluded_indices: set[int],
    top_k: int,
) -> list[tuple[int, int, float]]:
    """Rank frames by descending disagreement score, dropping excluded ones.

    Args:
        scores: one score per frame, indexed by frame index.
        excluded_indices: frame indices to drop (blank and/or already-labeled frames).
        top_k: number of highest-scoring frames to keep.

    Returns:
        (rank, frame_idx, score) tuples, rank starting at 1, highest score first.
    """
    candidates = [
        (frame_idx, score) for frame_idx, score in enumerate(scores)
        if frame_idx not in excluded_indices
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    return [
        (rank, frame_idx, score)
        for rank, (frame_idx, score) in enumerate(candidates[:top_k], start=1)
    ]


def draw_model_predictions(
    frame: np.ndarray,
    predictions: dict[str, pd.DataFrame],
    model_names: list[str],
    frame_idx: int,
) -> np.ndarray:
    """Draw every available model's tail/neck prediction on a frame.

    One color per model (see MODEL_COLORS_BGR), tail as a circle and neck as a square.
    Drawn regardless of prediction likelihood -- disagreement between low-confidence
    predictions is exactly what this is meant to surface.

    Args:
        frame: BGR frame to draw on, modified in place.
        predictions: model_name -> tidy prediction frame (see load_model_predictions).
        model_names: the full requested --models list, in order, so marker colors stay
            tied to a model's position in that list rather than to which subset of
            models happens to have predictions for this particular video.
        frame_idx: row to draw from each model's predictions.

    Returns:
        frame, for convenience.
    """
    for model_idx, model_name in enumerate(model_names):
        df = predictions.get(model_name)
        if df is None:
            continue

        color = MODEL_COLORS_BGR[model_idx % len(MODEL_COLORS_BGR)]
        row = df.iloc[frame_idx]
        for keypoint in KEYPOINTS:
            center = (int(round(row[f'{keypoint}_x'])), int(round(row[f'{keypoint}_y'])))
            if KEYPOINT_SHAPES[keypoint] == 'circle':
                cv2.circle(
                    frame, center, KEYPOINT_MARKER_SIZE, color,
                    thickness=KEYPOINT_MARKER_THICKNESS,
                )
            else:
                half = KEYPOINT_MARKER_SIZE
                cv2.rectangle(
                    frame,
                    (center[0] - half, center[1] - half),
                    (center[0] + half, center[1] + half),
                    color,
                    thickness=KEYPOINT_MARKER_THICKNESS,
                )

    return frame


def draw_frame_label(frame: np.ndarray, frame_idx: int, score: float) -> np.ndarray:
    """Burn the frame index and disagreement score into the top-left corner.

    Args:
        frame: BGR frame to draw on, modified in place.
        frame_idx: frame index to display.
        score: disagreement score to display.

    Returns:
        frame, for convenience.
    """
    text = f'frame {frame_idx}  var {score:.1f}'
    # black outline first so the white text stays legible over any background
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return frame


def draw_legend(frame: np.ndarray, model_names: list[str]) -> np.ndarray:
    """Draw a model-name -> marker-color legend in the bottom-left corner.

    Args:
        frame: BGR frame to draw on, modified in place.
        model_names: model names in the same order used for marker colors.

    Returns:
        frame, for convenience.
    """
    height = frame.shape[0]
    for line_idx, model_name in enumerate(model_names):
        color = MODEL_COLORS_BGR[line_idx % len(MODEL_COLORS_BGR)]
        y = height - 10 - (len(model_names) - 1 - line_idx) * 20
        cv2.putText(
            frame, model_name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, model_name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return frame


def process_video(
    video_path: Path,
    blank_frames_path: Path | None,
    project_dir: Path,
    model_names: list[str],
    labeled_indices: dict[str, set[int]],
    output_dir: Path,
    top_k: int,
) -> None:
    """Select and save QC images for one video's most-disagreeing candidate frames.

    Args:
        video_path: path to the raw video.
        blank_frames_path: path to the video's blank-frame indices file, if any.
        project_dir: Lightning Pose project root.
        model_names: model directory names to look up predictions from.
        labeled_indices: video_name -> already-labeled frame indices (see
            load_labeled_frame_indices).
        output_dir: directory to write {video_name}/rank*.png into.
        top_k: number of frames to save per video.
    """
    video_name = video_path.stem

    predictions = load_model_predictions(project_dir, model_names, video_name)
    if len(predictions) < 2:
        print(
            f'{video_name}: only {len(predictions)}/{len(model_names)} models have '
            f'predictions, skipping (need >= 2 to compute variance)'
        )
        return

    scores = compute_variance_scores(predictions)

    excluded_indices = set(labeled_indices.get(video_name, set()))
    if blank_frames_path is not None and blank_frames_path.exists():
        excluded_indices.update(read_blank_frame_indices(blank_frames_path))

    selected = select_top_frames(scores, excluded_indices, top_k)
    if not selected:
        print(f'{video_name}: no candidate frames left after filtering')
        return

    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    try:
        for rank, frame_idx, score in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                print(f'{video_name}: could not read frame {frame_idx}, skipping')
                continue

            draw_model_predictions(frame, predictions, model_names, frame_idx)
            draw_frame_label(frame, frame_idx, score)
            draw_legend(frame, model_names)

            frame_path = (
                video_output_dir / f'rank{rank:03d}_frame{frame_idx:08d}_var{score:.1f}.png'
            )
            cv2.imwrite(str(frame_path), frame)
    finally:
        cap.release()

    print(
        f'{video_name}: wrote {len(selected)} QC frames to {video_output_dir} '
        f'(using {len(predictions)}/{len(model_names)} models)'
    )


def main() -> None:
    """Parse arguments and run outlier selection over every video in data_dir."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--project-dir',
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help='Lightning Pose project root (contains models/, CollectedData.csv)',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        metavar='NAME',
        help='model directory names under {project_dir}/models to load predictions from',
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='raw video directory; defaults to data_dir from the cuttle config',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='defaults to {project_dir}/qc',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=DEFAULT_TOP_K,
        help='number of highest-disagreement frames to save per video',
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else load_config().data_dir
    output_dir = args.output_dir if args.output_dir is not None else args.project_dir / 'qc'

    labeled_indices = load_labeled_frame_indices(args.project_dir / 'CollectedData.csv')

    manifest = build_manifest(data_dir)
    for _, row in manifest.iterrows():
        process_video(
            video_path=Path(row['video_path']),
            blank_frames_path=(
                Path(row['blank_frames_path']) if row['blank_frames_path'] is not None else None
            ),
            project_dir=args.project_dir,
            model_names=args.models,
            labeled_indices=labeled_indices,
            output_dir=output_dir,
            top_k=args.top_k,
        )


if __name__ == '__main__':
    main()
