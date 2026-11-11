"""Contact sheets of frames binned by tail/neck keypoint jump size.

For each of N random videos, writes one 5x10 grid: rows are jump-size bins (>50, 40-50,
30-40, 20-30, 10-20 px), columns are 10 random frames from that bin. A frame's jump is the max
over both keypoints and both steps (t-1 -> t, t -> t+1) of the euclidean displacement, counting
only steps whose two endpoints are both high-likelihood. Frames that are blank, or have a
low-likelihood keypoint themselves, are excluded.

Usage:
    python jump_contact_sheets.py --n-videos 5 --seed 0 --out-dir <dir>
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from cuttle_patterns.config import load_config
from cuttle_patterns.preprocessing.pose import KEYPOINTS, load_pose_predictions

_logger = logging.getLogger(__name__)

POSE_DIR = Path(
    '/media/mattw/CUTTLE/pose-estimation/cuttle-test/models/iter-1.2_eks/video_preds'
)
LIKELIHOOD_THRESH = 0.9
# (low, high) px per row, top to bottom; high=inf for the first row
BINS = [(50, np.inf), (40, 50), (30, 40), (20, 30), (10, 20)]
N_COLS = 10
THUMB = 256
COLORS = {'neck': (0, 0, 255), 'tail': (255, 128, 0)}  # bgr


def compute_jumps(pose_df) -> tuple[np.ndarray, np.ndarray]:
    """Compute the per-frame max keypoint jump and the keypoint/side responsible.

    Args:
        pose_df: tidy pose frame from `load_pose_predictions`.

    Returns:
        (jump, ok): jump has shape (n_frames,), the max displacement over keypoints and
        steps (nan where no valid step exists); ok is a bool mask of frames whose own
        keypoints are all high-likelihood.
    """
    n = len(pose_df)
    jump = np.full(n, np.nan)
    ok = np.ones(n, dtype=bool)
    for kp in KEYPOINTS:
        xy = pose_df[[f'{kp}_x', f'{kp}_y']].to_numpy()
        good = pose_df[f'{kp}_likelihood'].to_numpy() >= LIKELIHOOD_THRESH
        ok &= good
        step = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # step[i]: i -> i+1
        valid = good[:-1] & good[1:]
        step = np.where(valid, step, np.nan)
        # frame t sees step[t-1] (t-1 -> t) and step[t] (t -> t+1)
        before = np.concatenate([[np.nan], step])
        after = np.concatenate([step, [np.nan]])
        jump = np.fmax(jump, np.fmax(before, after))
    return jump, ok


def read_frames(video_path: Path, idxs: list[int]) -> dict[int, np.ndarray]:
    """Read specific frames from a video.

    Args:
        video_path: path to the video.
        idxs: frame indices to read.

    Returns:
        mapping from frame index to bgr image.
    """
    cap = cv2.VideoCapture(str(video_path))
    out = {}
    for idx in sorted(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out[idx] = frame
    cap.release()
    return out


def annotate(frame, pose_df, idx: int, jump: float) -> np.ndarray:
    """Draw keypoints and their neighbor-frame positions on a thumbnail.

    Args:
        frame: 512x512 bgr frame.
        pose_df: tidy pose frame.
        idx: frame index.
        jump: jump size to print.

    Returns:
        annotated thumbnail of side `THUMB`.
    """
    img = frame.copy()
    for kp in KEYPOINTS:
        cur = pose_df[[f'{kp}_x', f'{kp}_y']].to_numpy()[idx]
        for nb in (idx - 1, idx + 1):
            if 0 <= nb < len(pose_df):
                pt = pose_df[[f'{kp}_x', f'{kp}_y']].to_numpy()[nb]
                cv2.line(img, tuple(int(v) for v in cur), tuple(int(v) for v in pt),
                         (255, 255, 255), 2)
                cv2.circle(img, tuple(int(v) for v in pt), 5, (255, 255, 255), 1)
        cv2.circle(img, tuple(int(v) for v in cur), 7, COLORS[kp], -1)
    img = cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    cv2.putText(img, f'{idx} d={jump:.0f}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)
    return img


def make_sheet(video_path: Path, pose_path: Path, blank_path: Path, rng) -> tuple:
    """Build the 5x10 contact sheet for one video.

    Args:
        video_path: crop video.
        pose_path: pose csv.
        blank_path: blank-frame list.
        rng: numpy random generator.

    Returns:
        (sheet image, list of per-bin frame counts).
    """
    pose_df = load_pose_predictions(pose_path)
    jump, ok = compute_jumps(pose_df)
    blank = set(int(v) for v in blank_path.read_text().split()) if blank_path.exists() else set()
    is_blank = np.zeros(len(pose_df), dtype=bool)
    is_blank[[b for b in blank if b < len(pose_df)]] = True
    use = ok & ~is_blank & ~np.isnan(jump)

    rows, counts, picks = [], [], []
    for lo, hi in BINS:
        cand = np.where(use & (jump >= lo) & (jump < hi))[0]
        counts.append(len(cand))
        picks.append(np.sort(rng.choice(cand, min(N_COLS, len(cand)), replace=False)))
    frames = read_frames(video_path, [int(i) for p in picks for i in p])
    for pick in picks:
        tiles = [annotate(frames[int(i)], pose_df, int(i), jump[i]) for i in pick if int(i) in frames]
        tiles += [np.zeros((THUMB, THUMB, 3), np.uint8)] * (N_COLS - len(tiles))
        rows.append(np.hstack(tiles))
    return np.vstack(rows), counts


def main() -> None:
    """Make contact sheets for random videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-videos', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = load_config().data_dir
    rng = np.random.default_rng(args.seed)
    pose_files = sorted(POSE_DIR.glob('*.csv'))
    chosen = rng.choice(len(pose_files), args.n_videos, replace=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in chosen:
        pose_path = pose_files[i]
        stem = pose_path.stem  # e.g. Day10_Tank1_Cuttle1_Intruder_Crop
        blank = data_dir / f'{stem.removesuffix("_Crop")}_black_frames.txt'
        sheet, counts = make_sheet(data_dir / f'{stem}.mp4', pose_path, blank, rng)
        out = args.out_dir / f'{stem}_jumps.jpg'
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _logger.info(f'{stem}: rows (>50,40-50,30-40,20-30,10-20) have {counts} frames -> {out}')


if __name__ == '__main__':
    main()
