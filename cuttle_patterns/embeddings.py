"""Load per-frame BEAST embeddings written by `cuttle predict --save-latents`.

Shared by Phase 5 (`cuttle reduce`) and Phase 6 (`cuttle cluster`), both of which read
the same per-frame `.npy` latent vectors under a model's
`image_predictions/{predictions_name}/latents/{video_name}/img{frame_number}.npy` tree
(BEAST's own output layout — see the docstring of `cuttle_patterns/paths.py`).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# mirrors ingest.FILENAME_PATTERN's video-name shape, but captures day/tank/role
# directly instead of the combined session_id/fish_id groups, since downstream metadata
# needs those split out
VIDEO_NAME_PATTERN = re.compile(
    r'^Day(?P<day>\d+)_Tank(?P<tank>\d+)_Cuttle\d+_(?P<role>[A-Za-z]+)_[Cc]rop$'
)
FRAME_FILENAME_PATTERN = re.compile(r'^img(?P<frame_number>\d+)$')


def parse_video_name(video_name: str) -> dict[str, int | str]:
    """Parse day/tank/role out of a `Day{d}_Tank{t}_Cuttle{n}_{role}_Crop` video name.

    Args:
        video_name: a video-name stem, e.g. `Day1_Tank2_Cuttle1_Resident_Crop` — the
            directory name BEAST groups per-frame latents under (see `load_latents`).

    Returns:
        dict with keys `day` (int), `tank` (int), `role` (str).

    Raises:
        ValueError: if video_name doesn't match the expected naming convention.
    """
    match = VIDEO_NAME_PATTERN.match(video_name)
    if match is None:
        raise ValueError(f'video name does not match expected pattern: {video_name}')
    return {
        'day': int(match['day']),
        'tank': int(match['tank']),
        'role': match['role'],
    }


def load_latents(latents_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load every per-frame latent vector under a `cuttle predict --save-latents` tree.

    Args:
        latents_dir: `.../image_predictions/{predictions_name}/latents` directory,
            containing one subdirectory per video, each holding `img{frame_number}.npy`
            files (BEAST's own output layout).

    Returns:
        (X, meta): `X` is a float array of shape (n_frames, latent_dim); `meta` is a
        DataFrame row-aligned with `X`, with columns `video_name`, `day`, `tank`,
        `role`, `frame_number`, sorted by `(video_name, frame_number)`.

    Raises:
        FileNotFoundError: if latents_dir does not exist.
        ValueError: if latents_dir exists but contains no `.npy` files, or a filename
            doesn't match the expected `img{frame_number}.npy` pattern.
    """
    if not latents_dir.is_dir():
        raise FileNotFoundError(
            f'latents directory does not exist: {latents_dir}; run `cuttle predict '
            f'--save-latents` first'
        )

    latent_paths = sorted(latents_dir.glob('*/*.npy'))
    if not latent_paths:
        raise ValueError(f'no .npy latent files found under {latents_dir}')

    rows = []
    vectors = []
    for latent_path in latent_paths:
        frame_match = FRAME_FILENAME_PATTERN.match(latent_path.stem)
        if frame_match is None:
            raise ValueError(f'unexpected latent filename: {latent_path}')

        video_name = latent_path.parent.name
        row = parse_video_name(video_name)
        row['video_name'] = video_name
        row['frame_number'] = int(frame_match['frame_number'])
        rows.append(row)
        vectors.append(np.load(latent_path))

    meta = pd.DataFrame(rows, columns=['video_name', 'day', 'tank', 'role', 'frame_number'])
    X = np.stack(vectors)

    order = meta.sort_values(['video_name', 'frame_number']).index.to_numpy()
    return X[order], meta.iloc[order].reset_index(drop=True)
