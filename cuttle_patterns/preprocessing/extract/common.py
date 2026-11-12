"""Helpers shared by the `cuttle extract-*` pipeline stages.

Video discovery, sharding, atomic file writes, progress files for long-running jobs, and a
background-thread frame reader that overlaps video decoding with GPU work.
"""

import json
import queue
import re
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_GLOB = '*_Crop.mp4'
QUEUE_DEPTH = 3
_DAY_PATTERN = re.compile(r'^Day(\d+)_')


def list_videos(video_dir: Path, stems: list[str] | None = None) -> list[Path]:
    """List the per-fish crop or rectangle videos in a directory.

    `*_overlay.mp4` QC clips are not matched.

    Args:
        video_dir: directory to search.
        stems: if given, only videos whose stem is in this list.

    Returns:
        sorted list of video paths.
    """
    video_paths = sorted(video_dir.glob(VIDEO_GLOB))
    if stems:
        video_paths = [p for p in video_paths if p.stem in set(stems)]
    return video_paths


def get_day(stem: str) -> int:
    """Parse the recording day from a video stem.

    Args:
        stem: video stem, e.g. `Day5_Tank3_Cuttle2_Resident_Crop`.

    Returns:
        the day number.

    Raises:
        ValueError: if the stem doesn't start with `Day{n}_`.
    """
    match = _DAY_PATTERN.match(stem)
    if match is None:
        raise ValueError(f'cannot parse a day from video name: {stem}')
    return int(match[1])


def shard_items(items: list, shard: int, n_shards: int) -> list:
    """Select every `n_shards`-th item, starting at `shard`.

    Args:
        items: items to split.
        shard: this process's shard index, in [0, n_shards).
        n_shards: total number of shards.

    Returns:
        the items belonging to this shard.

    Raises:
        ValueError: if `shard` is not in [0, n_shards).
    """
    if not 0 <= shard < n_shards:
        raise ValueError(f'shard must be in [0, {n_shards}), got {shard}')
    return items[shard::n_shards]


def video_frame_count(video_path: Path) -> int:
    """Read the frame count reported by a video container.

    Args:
        video_path: path to the video.

    Returns:
        frame count reported by the container.

    Raises:
        OSError: if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f'could not open video file: {video_path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames


def write_progress(path: Path, done: int, total: int, current: str | None) -> None:
    """Write a small json progress file that can be read without attaching to the job.

    Args:
        path: file to write.
        done: number of finished items.
        total: total number of items.
        current: the item being worked on, or None when finished.
    """
    path.write_text(json.dumps({'done': done, 'total': total, 'current': current}))


def progress_path(out_dir: Path, shard: int, n_shards: int) -> Path:
    """Name of a shard's progress file inside an output directory.

    Args:
        out_dir: output directory.
        shard: shard index.
        n_shards: total number of shards.

    Returns:
        path of the form `{out_dir}/progress_shard{shard}of{n_shards}.json`.
    """
    return out_dir / f'progress_shard{shard}of{n_shards}.json'


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write a file via a temporary name, then rename, so a partial file never looks complete.

    Args:
        path: final path.
        write_fn: callable taking the temporary path and writing the file there.
    """
    tmp = path.with_name(f'{path.name}.tmp')
    write_fn(tmp)
    tmp.replace(path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    """Save an array to exactly `path` (no appended '.npy') via a temporary name.

    Args:
        path: destination file.
        array: array to save.
    """

    def _write(tmp: Path) -> None:
        with open(tmp, 'wb') as f:
            np.save(f, array)

    atomic_write(path, _write)


def save_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    """Save a dataframe as parquet via a temporary name.

    Args:
        path: destination file.
        df: dataframe to save.
    """
    atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))


def prefetch_batches(
    video_path: Path,
    batch_size: int,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Iterator[np.ndarray]:
    """Decode a video sequentially in a background thread and yield batches of frames.

    Decoding (and `transform`, if given) run in the producer thread so they overlap with the
    consumer's GPU work. Frames are yielded in video order.

    Args:
        video_path: video to decode.
        batch_size: frames per batch (the last batch may be smaller).
        transform: optional function applied to each batch, a uint8 bgr array of shape
            (B, H, W, 3), before it is queued.

    Yields:
        batches of frames (bgr uint8 (B, H, W, 3) arrays, or whatever `transform` returns).

    Raises:
        OSError: if the video cannot be opened.
    """
    batches: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)

    def _produce() -> None:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise OSError(f'could not open video file: {video_path}')
            buf = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                buf.append(frame)
                if len(buf) == batch_size:
                    batch = np.stack(buf)
                    batches.put(transform(batch) if transform else batch)
                    buf = []
            if buf:
                batch = np.stack(buf)
                batches.put(transform(batch) if transform else batch)
            cap.release()
            batches.put(None)
        except Exception as e:  # noqa: BLE001 - forwarded to the consuming thread
            batches.put(e)

    thread = threading.Thread(target=_produce, daemon=True)
    thread.start()
    try:
        while True:
            item = batches.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        thread.join(timeout=1.0)
