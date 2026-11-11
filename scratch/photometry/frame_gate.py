"""Step 0: gate every iter-1.0 frame on whether its bbox row is usable for crop->original mapping.

For each (session, fish, frame_idx) in `.iter-1.0/manifests/extract.parquet` (plus the +-1
context frames `export_frames` also writes), measures the visible animal's extent in the 512
crop video and compares it with each `obj_id`'s bbox size scaled by the session's crop scale
(`512 / canvas_side` from `canvas_sides.csv`). A frame passes if exactly one `obj_id` matches
in width and height and the animal's extent is centered in the crop (the crop is centered on
the bbox). Frames where the animal touches the crop edge are marked `clipped` (size can't be
verified). Read-only on raw data; writes one parquet.

Example:
    python scratch/photometry/frame_gate.py
"""

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW_DIR = Path('/media/mattw/CUTTLE/raw_data')
RESULTS_DIR = Path('/media/mattw/CUTTLE/results')
ITER = '.iter-1.0'
OUT_ITER = '.iter-1.0-photometry'
CROP_SIDE = 512
BLACK_THRESH = 25  # max channel value above which a pixel counts as animal
MIN_PIXELS = 2000
SIZE_TOL_PX = 12  # allowed |measured - expected| extent in crop pixels (edge blur, arms)
CENTER_TOL_PX = 6


def measure_extent(img: np.ndarray) -> tuple[int, int, float, float] | None:
    """Measure the visible animal's extent in a crop frame.

    Args:
        img: BGR crop frame

    Returns:
        (width, height, center_x, center_y) of the non-black bounding box, or None if the frame
        is (nearly) blank
    """
    ys, xs = np.where(img.max(2) > BLACK_THRESH)
    if len(xs) < MIN_PIXELS:
        return None
    return (
        int(xs.max() - xs.min()),
        int(ys.max() - ys.min()),
        float(xs.min() + xs.max()) / 2,
        float(ys.min() + ys.max()) / 2,
    )


def gate_video(args: tuple) -> list[dict]:
    """Gate all requested frames of one video.

    Args:
        args: (session_id, fish_id, video_path, frame_idxs, scale, box_prefix)

    Returns:
        one row per frame with status, matched obj_id and match errors
    """
    session_id, fish_id, video_path, frame_idxs, scale, prefix = args
    bboxes = pd.read_csv(RAW_DIR / f'{session_id}_bboxes.csv').set_index('frame')
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    for anchor in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(anchor - 1, 0))
        for idx in range(max(anchor - 1, 0), anchor + 2):
            ok, img = cap.read()
            row = {
                'session_id': session_id, 'fish_id': fish_id, 'frame_idx': idx,
                'is_anchor': idx == anchor, 'status': 'unreadable', 'obj_id': 0,
                'err_px': np.nan, 'err_other_px': np.nan, 'center_off_px': np.nan,
            }
            rows.append(row)
            if not ok:
                continue
            ext = measure_extent(img)
            if ext is None:
                row['status'] = 'blank'
                continue
            width, height, cx, cy = ext
            row['center_off_px'] = max(abs(cx - 255.5), abs(cy - 255.5))
            row['extent_w'], row['extent_h'] = width, height
            clipped = max(width, height) >= CROP_SIDE - 3  # animal touches the crop edge
            errs = {}
            if idx in bboxes.index:
                for _, r in bboxes.loc[[idx]].iterrows():
                    bbox_w = r[f'orig_{prefix}_x1'] - r[f'orig_{prefix}_x0']
                    bbox_h = r[f'orig_{prefix}_y1'] - r[f'orig_{prefix}_y0']
                    errs[int(r.obj_id)] = max(
                        abs(width - scale * bbox_w), abs(height - scale * bbox_h),
                    )
            if not errs:
                row['status'] = 'no_bbox'
                continue
            best = min(errs, key=errs.get)
            others = [e for k, e in errs.items() if k != best]
            row['obj_id'] = best
            row['err_px'] = errs[best]
            row['err_other_px'] = min(others) if others else np.nan
            if clipped:
                # extent is cut off by the crop edge, so size/center can't be checked; keep the
                # best match by size as a hint only
                row['status'] = 'clipped'
            elif errs[best] > SIZE_TOL_PX or row['center_off_px'] > CENTER_TOL_PX:
                row['status'] = 'size_mismatch'
            elif others and min(others) <= SIZE_TOL_PX:
                row['status'] = 'ambiguous'
            else:
                row['status'] = 'pass'
    cap.release()
    return rows


def main() -> None:
    """Build the frame gate manifest for all iter-1.0 frames."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results_dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--n_procs', type=int, default=8)
    parser.add_argument('--sessions', nargs='*', default=None, help='restrict to these sessions')
    args = parser.parse_args()

    extract = pd.read_parquet(args.results_dir / ITER / 'manifests' / 'extract.parquet')
    ingest = pd.read_parquet(args.results_dir / ITER / 'manifests' / 'ingest.parquet')
    canvas = pd.read_csv(RAW_DIR / 'canvas_sides.csv')
    if args.sessions:
        extract = extract[extract.session_id.isin(args.sessions)]

    jobs = []
    for (session_id, fish_id), grp in extract.groupby(['session_id', 'fish_id']):
        video_path = ingest[
            (ingest.session_id == session_id) & (ingest.fish_id == fish_id)
        ].video_path.iloc[0]
        day, tank = session_id.split('_')
        cuttle = int(fish_id.split('_')[0].removeprefix('Cuttle'))
        row = canvas[
            (canvas.day == f'DAY_{day.removeprefix("Day")}')
            & (canvas.tank == f'Tank_{tank.removeprefix("Tank")}')
            & (canvas.cuttle == cuttle)
        ].iloc[0]
        prefix = 'raw' if row.box == 'raw' else 'main'
        scale = CROP_SIDE / row.canvas_side
        jobs.append((
            session_id, fish_id, video_path, sorted(grp.frame_idx.tolist()), scale, prefix,
        ))

    with Pool(args.n_procs) as pool:
        results = pool.map(gate_video, jobs, chunksize=1)
    out = pd.DataFrame([r for rows in results for r in rows])
    out_path = args.results_dir / OUT_ITER / 'manifests' / 'frame_gate.parquet'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path)
    print(f'wrote {out_path} ({len(out)} frames)')
    print(out.groupby('is_anchor').status.value_counts().unstack(fill_value=0))
    print(out[out.is_anchor].groupby('session_id').status.value_counts().unstack(fill_value=0))


if __name__ == '__main__':
    main()
