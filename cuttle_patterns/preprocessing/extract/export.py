"""Export the selected frames as individual PNGs (`cuttle extract-export`).

Writes one `img{frame_idx:08d}.png` per selected frame under `{output_dir}/{video}/`, read from
the aligned rectangle videos, plus a per-video `selected_frames.csv` (sorted file names) and the
combined `manifests/extract.parquet` that downstream stages (`cuttle train`, `cuttle embed`,
`cuttle predict`) read. Only the selected frames themselves are exported; there are no context
(neighbor) frames.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cuttle_patterns.ingest import FILENAME_PATTERN

_logger = logging.getLogger(__name__)

N_DIGITS = 8
EXTENSION = 'png'
MANIFEST_COLUMNS = ['session_id', 'fish_id', 'frame_idx', 'image_path']


def frame_filename(frame_idx: int) -> str:
    """File name of an exported frame.

    Args:
        frame_idx: frame index within its video.

    Returns:
        e.g. `img00001234.png`.
    """
    return f'img{str(frame_idx).zfill(N_DIGITS)}.{EXTENSION}'


def read_frames_sequential(video_path: Path, frame_idxs: np.ndarray) -> dict[int, np.ndarray]:
    """Read specific frames of a video by decoding it once, front to back.

    Cheaper than seeking for ~1k frames of a long video: frames that aren't wanted are only
    grabbed, not decoded to an image.

    Args:
        video_path: video to read.
        frame_idxs: frame indices to read.

    Returns:
        mapping from frame index to bgr image, for the frames that could be read.

    Raises:
        OSError: if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')
    wanted = set(int(i) for i in frame_idxs)
    last = max(wanted) if wanted else -1
    frames = {}
    for idx in range(last + 1):
        if idx in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            frames[idx] = frame
        elif not cap.grab():
            break
    cap.release()
    return frames


def export_video_frames(
    video_path: Path,
    output_dir: Path,
    frame_idxs: np.ndarray,
) -> Path:
    """Export the given frames of one video as PNGs, plus its `selected_frames.csv`.

    Args:
        video_path: aligned rectangle video.
        output_dir: parent directory; frames go in `output_dir/{video stem}/`.
        frame_idxs: frame indices to export.

    Returns:
        the per-video directory.

    Raises:
        ValueError: if some requested frames could not be read from the video.
    """
    save_dir = output_dir / video_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)
    frames = read_frames_sequential(video_path, frame_idxs)
    missing = sorted(set(int(i) for i in frame_idxs) - set(frames))
    if missing:
        raise ValueError(f'{video_path}: could not read frames {missing[:5]}...')
    for idx, frame in frames.items():
        cv2.imwrite(str(save_dir / frame_filename(idx)), frame)
    names = np.array(sorted(frame_filename(int(i)) for i in frame_idxs))
    np.savetxt(save_dir / 'selected_frames.csv', names, delimiter=',', fmt='%s')
    return save_dir


def build_extraction_manifest(rows: list[dict]) -> pd.DataFrame:
    """Build the combined frame manifest from per-frame rows.

    Args:
        rows: dicts with keys session_id, fish_id, frame_idx, image_path, one per exported frame.

    Returns:
        one row per exported frame, columns session_id, fish_id, frame_idx, image_path.
    """
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def run_export(
    manifest: pd.DataFrame,
    video_dir: Path,
    output_dir: Path,
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Export every selected frame in a selection manifest and build the extract manifest.

    Args:
        manifest: selection manifest (`video`, `frame_idx` columns, from
            `cuttle extract-kmeans-pooled`).
        video_dir: directory of aligned rectangle videos.
        output_dir: parent directory for per-video frame directories.
        skip_existing: skip a video whose `selected_frames.csv` already exists (its frames are
            still listed in the returned manifest).

    Returns:
        the combined extract manifest (`session_id`, `fish_id`, `frame_idx`, `image_path`).
    """
    rows = []
    for video, group in manifest.groupby('video'):
        match = FILENAME_PATTERN.match(video)
        if match is None:
            _logger.warning(f'skipping video with unexpected name: {video}')
            continue
        frame_idxs = np.sort(group['frame_idx'].to_numpy())
        save_dir = output_dir / video
        if skip_existing and (save_dir / 'selected_frames.csv').exists():
            _logger.info(f'{video}: already exported, skipping')
        else:
            save_dir = export_video_frames(video_dir / f'{video}.mp4', output_dir, frame_idxs)
            _logger.info(f'{video}: wrote {len(frame_idxs)} frames to {save_dir}')
        rows.extend(
            {
                'session_id': match['session_id'],
                'fish_id': match['fish_id'],
                'frame_idx': int(idx),
                'image_path': str(save_dir / frame_filename(int(idx))),
            }
            for idx in frame_idxs
        )
    return build_extraction_manifest(rows)
