"""Build a manifest of raw session/fish videos and their blank-frame annotations.

Raw videos live directly under `data_dir` as
`Day{day}_Tank{tank}_Cuttle{n}_{role}_crop.mp4` (`crop`/`Crop` both seen), each with an
accompanying `Day{day}_Tank{tank}_Cuttle{n}_{role}_black_frames.txt` listing the frame
indices (one per line) that the collaborators flagged as blank. `session_id` is the
`Day{day}_Tank{tank}` prefix (e.g. `Day1_Tank2`) and `fish_id` is the `Cuttle{n}_{role}`
suffix (e.g. `Cuttle1_Resident`) — both strings, not the small integers used previously.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd

logger = logging.getLogger(__name__)

FILENAME_PATTERN = re.compile(
    r'^(?P<session_id>Day\d+_Tank\d+)_(?P<fish_id>Cuttle\d+_[A-Za-z]+)_[Cc]rop$'
)


@dataclass
class VideoInfo:
    """Metadata read directly from a video file.

    Attributes:
        n_frames: frame count reported by the video container.
        fps: frames per second.
        width: frame width in pixels.
        height: frame height in pixels.
    """

    n_frames: int
    fps: float
    width: int
    height: int


def find_raw_videos(raw_dir: Path) -> list[Path]:
    """Find raw session/fish video files in a directory.

    Args:
        raw_dir: directory containing `Day{day}_Tank{tank}_Cuttle{n}_{role}_crop.mp4`
            files.

    Returns:
        sorted list of video paths.

    Raises:
        FileNotFoundError: if raw_dir does not exist.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f'raw video directory does not exist: {raw_dir}')
    return sorted(raw_dir.glob('Day*_Tank*_Cuttle*_*.mp4'))


def read_video_info(video_path: Path) -> VideoInfo:
    """Read frame count, fps, and resolution from a video file.

    Args:
        video_path: path to the video file.

    Returns:
        metadata read from the video.

    Raises:
        OSError: if the video file cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')

    try:
        info = VideoInfo(
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()

    return info


def read_blank_frame_indices(blank_frames_path: Path) -> list[int]:
    """Read blank-frame indices from a text file (one integer index per line).

    Args:
        blank_frames_path: path to the text file.

    Returns:
        sorted list of blank-frame indices.
    """
    lines = blank_frames_path.read_text().splitlines()
    return sorted(int(line) for line in lines if line.strip())


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    """Build a manifest of raw videos, their metadata, and blank-frame counts.

    Args:
        raw_dir: directory containing raw
            `Day{day}_Tank{tank}_Cuttle{n}_{role}_crop.mp4`/`_black_frames.txt` file
            pairs.

    Returns:
        one row per video, with columns: session_id, fish_id, video_path,
        blank_frames_path, n_frames, n_blank_frames, fps, width, height. If a video's
        blank-frames file is missing, blank_frames_path and n_blank_frames are None.
    """
    rows = []
    for video_path in find_raw_videos(raw_dir):
        match = FILENAME_PATTERN.match(video_path.stem)
        if match is None:
            logger.warning(f'skipping file with unexpected name: {video_path}')
            continue

        session_id = match['session_id']
        fish_id = match['fish_id']

        # the black-frames txt file uses a different suffix than the video, so its name
        # has to be rebuilt from the parsed session/fish ids rather than swapping the
        # video path's extension
        blank_frames_path = video_path.parent / f'{session_id}_{fish_id}_black_frames.txt'
        blank_frame_indices = None
        if blank_frames_path.exists():
            blank_frame_indices = read_blank_frame_indices(blank_frames_path)
        else:
            logger.warning(f'no blank-frames file found for {video_path}')

        video_info = read_video_info(video_path)

        if blank_frame_indices and blank_frame_indices[-1] >= video_info.n_frames:
            logger.warning(
                f'{blank_frames_path} has an index ({blank_frame_indices[-1]}) outside '
                f'the video\'s frame range ({video_info.n_frames} frames)'
            )

        has_blank_frames = blank_frame_indices is not None
        rows.append(
            {
                'session_id': session_id,
                'fish_id': fish_id,
                'video_path': str(video_path),
                'blank_frames_path': str(blank_frames_path) if has_blank_frames else None,
                'n_frames': video_info.n_frames,
                'n_blank_frames': len(blank_frame_indices) if has_blank_frames else None,
                'fps': video_info.fps,
                'width': video_info.width,
                'height': video_info.height,
            }
        )

    return pd.DataFrame(rows)
