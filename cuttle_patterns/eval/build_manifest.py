"""Build the eval manifest: join classifier predictions onto the existing frame set.

The eval frame set is not a new sample — it's the frame set `cuttle extract-export`
already exported (`results_dir/manifests/extract.parquet`). This module only attaches the
classifier's predicted pattern label to each of those frames, since the eval metrics
need a pattern-type proxy alongside each frame's video identity.
"""

from pathlib import Path

import pandas as pd

from cuttle_patterns.paths import CLASSIFICATIONS_RELPATH, EXTRACT_MANIFEST_RELPATH

# matches cuttle_patterns.latents.VIDEO_NAME_PATTERN's naming convention
VIDEO_NAME_SUFFIX = '_Crop'


def _load_classifications(path: Path, video_names: set[str]) -> pd.DataFrame:
    """Load classifier predictions from either on-disk layout.

    Args:
        path: `classifications/{classifier_name}`; a directory of per-video parquets
            (`cuttle extract-classify`: `{video_name}.parquet` with a `frame_idx`
            column), or, with `.parquet` appended, a single legacy file with
            `video_name` and `frame_number` columns.
        video_names: videos to read in the directory layout; others are skipped.

    Returns:
        predictions with `video_name` and `frame_number` columns.

    Raises:
        FileNotFoundError: if neither layout exists, or a needed video has no file.
    """
    if path.is_dir():
        frames = []
        for video_name in sorted(video_names):
            video_path = path / f'{video_name}.parquet'
            if not video_path.is_file():
                raise FileNotFoundError(f'no classifications file found at {video_path}')
            frames.append(
                pd.read_parquet(video_path).assign(video_name=video_name).rename(
                    columns={'frame_idx': 'frame_number'},
                )
            )
        return pd.concat(frames, ignore_index=True)
    legacy_path = path.with_name(f'{path.name}.parquet')
    if not legacy_path.is_file():
        raise FileNotFoundError(f'no classifications file found at {legacy_path}')
    return pd.read_parquet(legacy_path)


def build_eval_manifest(results_dir: Path, classifier_name: str) -> pd.DataFrame:
    """Join classifier predictions onto the existing eval frame set.

    Args:
        results_dir: root results directory (see `cuttle_patterns.config.Config`).
        classifier_name: classifications to join: the directory
            `results_dir/classifications/{classifier_name}` of per-video parquets
            (`cuttle extract-classify`), or the single file
            `results_dir/classifications/{classifier_name}.parquet`.

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

    classifications_path = results_dir / CLASSIFICATIONS_RELPATH / classifier_name
    classifications = _load_classifications(classifications_path, set(extract['video_name']))
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
