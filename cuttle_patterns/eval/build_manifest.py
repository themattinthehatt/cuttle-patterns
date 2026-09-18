"""Build the eval manifest: join classifier predictions onto the existing frame set.

The eval frame set is not a new sample — it's the same one `cuttle extract` already
produced (`results_dir/manifests/extract.parquet`). This module only attaches the
classifier's predicted pattern label to each of those frames, since the eval metrics
need a pattern-type proxy alongside each frame's video identity.
"""

from pathlib import Path

import pandas as pd

from cuttle_patterns.paths import CLASSIFICATIONS_RELPATH, EXTRACT_MANIFEST_RELPATH

# matches cuttle_patterns.embeddings.VIDEO_NAME_PATTERN's naming convention
VIDEO_NAME_SUFFIX = '_Crop'


def build_eval_manifest(results_dir: Path, classifier_name: str) -> pd.DataFrame:
    """Join classifier predictions onto the existing eval frame set.

    Args:
        results_dir: root results directory (see `cuttle_patterns.config.Config`).
        classifier_name: stem of the classifications file to join, i.e. the file at
            `results_dir/classifications/{classifier_name}.parquet`
            (`scripts/classify_skin_pattern.py`'s output).

    Returns:
        one row per eval frame, with columns `video_name`, `frame_number`,
        `session_id`, `fish_id`, `image_path`, `predicted_pattern`, `confidence`, and
        one `prob_{class}` column per classifier class.

    Raises:
        FileNotFoundError: if either input file doesn't exist.
        ValueError: if any eval frame has no matching classifier prediction.
    """
    extract_path = results_dir / EXTRACT_MANIFEST_RELPATH
    if not extract_path.is_file():
        raise FileNotFoundError(f'no extract manifest found at {extract_path}')
    extract = pd.read_parquet(extract_path)
    extract = extract.assign(
        video_name=(
            extract['session_id'].astype(str)
            + '_'
            + extract['fish_id'].astype(str)
            + VIDEO_NAME_SUFFIX
        ),
        frame_number=extract['frame_idx'].astype('int64'),
    )

    classifications_path = results_dir / CLASSIFICATIONS_RELPATH / f'{classifier_name}.parquet'
    if not classifications_path.is_file():
        raise FileNotFoundError(f'no classifications file found at {classifications_path}')
    classifications = pd.read_parquet(classifications_path)
    classifications = classifications.assign(
        video_name=classifications['video_name'].astype(str),
        frame_number=classifications['frame_number'].astype('int64'),
    )

    manifest = extract.merge(
        classifications,
        on=['video_name', 'frame_number'],
        how='left',
        validate='one_to_one',
    )
    missing = manifest['predicted_pattern'].isna()
    if missing.any():
        raise ValueError(
            f'{missing.sum()} eval frames have no matching classifier prediction in '
            f'{classifications_path}'
        )

    return manifest
