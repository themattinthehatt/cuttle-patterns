"""Load and validate the per-frame data the visualizer plots (Phase 7).

Reads the one-row-per-frame parquet files `cuttle reduce`/`cuttle cluster` write under a
model's `reduce/`/`clusters/` directories (see `cuttle_patterns/paths.py`) and joins them
on `(video_name, frame_number)`, since neither file carries the image itself — frames live
under `results_dir/beast_frames/{video_name}/img{frame_number:08d}.png` (Phase 3 output).
"""

from pathlib import Path

import pandas as pd

from cuttle_patterns import paths

INDEX_COLS = ('video_name', 'frame_number')
NON_COLORABLE_COLUMNS = {'umap_x', 'umap_y', 'image_relpath'}
DEFAULT_MAX_CATEGORIES = 20


def list_model_names(results_dir: Path) -> list[str]:
    """List trained model names under `results_dir/beast_models/`.

    Args:
        results_dir: the resolved results directory.

    Returns:
        sorted list of model directory names; empty if `beast_models/` doesn't exist.
    """
    models_dir = results_dir / paths.BEAST_MODELS_RELPATH
    if not models_dir.is_dir():
        return []
    return sorted(p.name for p in models_dir.iterdir() if p.is_dir())


def list_reduce_paths(model_dir: Path) -> list[Path]:
    """List available `cuttle reduce` output files for a model.

    Args:
        model_dir: `results_dir/beast_models/{model_name}`.

    Returns:
        sorted list of paths under `model_dir/reduce/`; empty if the directory is missing.
    """
    reduce_dir = model_dir / paths.REDUCE_RELPATH
    if not reduce_dir.is_dir():
        return []
    return sorted(reduce_dir.glob('*.parquet'))


def list_cluster_paths(model_dir: Path) -> list[Path]:
    """List available `cuttle cluster` output files for a model, without loading them.

    Args:
        model_dir: `results_dir/beast_models/{model_name}`.

    Returns:
        sorted list of paths under `model_dir/clusters/`; empty if the directory is
        missing. Callers load a file's contents (via `attach_cluster_column`) only once
        the user actually selects it.
    """
    clusters_dir = model_dir / paths.CLUSTERS_RELPATH
    if not clusters_dir.is_dir():
        return []
    return sorted(clusters_dir.glob('*.parquet'))


def build_image_relpath(video_name: str, frame_number: int) -> str:
    """Build a frame's image path, relative to `results_dir/beast_frames/`.

    Args:
        video_name: the video directory name, e.g. `Day1_Tank2_Cuttle1_Resident_Crop`.
        frame_number: the frame's number, as stored in a reduce/cluster parquet (not
            zero-padded — the padding is reapplied here to match the on-disk filename
            `cuttle extract` writes).

    Returns:
        a URL-style relative path, e.g. `Day1_Tank2_Cuttle1_Resident_Crop/img00000042.png`.
    """
    return f'{video_name}/img{frame_number:08d}.png'


def load_reduce_dataframe(reduce_path: Path) -> pd.DataFrame:
    """Load a `cuttle reduce` output file and attach each frame's image path.

    Args:
        reduce_path: path to a `reduce/umap_{hparams}.parquet` file.

    Returns:
        the parquet's DataFrame with an added `image_relpath` column.
    """
    df = pd.read_parquet(reduce_path)
    df['image_relpath'] = [
        build_image_relpath(video_name, frame_number)
        for video_name, frame_number in zip(df['video_name'], df['frame_number'], strict=True)
    ]
    return df


def attach_cluster_column(df: pd.DataFrame, cluster_path: Path) -> pd.DataFrame:
    """Attach a `cuttle cluster` output file's labels as a new column.

    The new column is named after `cluster_path`'s stem (e.g. `kmeans_k10`) rather than
    the literal `cluster` column the file stores it under, so multiple cluster files can
    be attached at once without colliding.

    Args:
        df: a DataFrame from `load_reduce_dataframe` (or one already carrying other
            attached cluster columns), with `INDEX_COLS` present.
        cluster_path: path to a `clusters/{method}_{hparams}.parquet` file.

    Returns:
        a new DataFrame (`df` is not mutated) with the cluster labels attached.

    Raises:
        ValueError: if `cluster_path`'s `(video_name, frame_number)` keys don't exactly
            match `df`'s.
    """
    cluster_df = pd.read_parquet(cluster_path)

    df_keys = set(zip(*(df[col] for col in INDEX_COLS), strict=True))
    cluster_keys = set(zip(*(cluster_df[col] for col in INDEX_COLS), strict=True))
    if df_keys != cluster_keys:
        missing_from_cluster = df_keys - cluster_keys
        missing_from_df = cluster_keys - df_keys
        raise ValueError(
            f'{cluster_path.name} does not match the loaded reduction on '
            f'{INDEX_COLS}: {len(missing_from_cluster)} key(s) present in the reduction '
            f'but missing from {cluster_path.name}, {len(missing_from_df)} key(s) present '
            f'in {cluster_path.name} but missing from the reduction'
        )

    column_name = cluster_path.stem
    merged = pd.merge(
        df,
        cluster_df[[*INDEX_COLS, 'cluster']].rename(columns={'cluster': column_name}),
        on=list(INDEX_COLS),
        how='left',
        validate='one_to_one',
    )
    return merged


def colorable_columns(df: pd.DataFrame) -> list[str]:
    """List the columns a user can color the scatter plot by.

    Args:
        df: a loaded (and possibly cluster-augmented) frame DataFrame.

    Returns:
        `df`'s columns, excluding the UMAP coordinates and the image path.
    """
    return [c for c in df.columns if c not in NON_COLORABLE_COLUMNS]


def is_categorical_column(
    df: pd.DataFrame,
    column: str,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> bool:
    """Decide whether a column should be colored with a discrete or continuous palette.

    Args:
        df: a loaded (and possibly cluster-augmented) frame DataFrame.
        column: the column to check.
        max_categories: numeric columns with at most this many distinct values are still
            treated as categorical (e.g. `cluster`, `day`, `tank`).

    Returns:
        True if `column` should use a discrete/categorical color palette.
    """
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return series.nunique() <= max_categories
