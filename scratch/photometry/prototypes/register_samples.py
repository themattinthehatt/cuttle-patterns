"""Register crop videos against full-frame sample images for one session.

Warps each crop frame into sample-image coordinates using the bbox mapping (crop = square window
of `canvas_side` original px centered on the bbox, resized to 512), and compares it with the
sample image over the animal's pixels. Answers: (1) is the geometry right, (2) which video
frame index does a sample's `frame` label correspond to (tries several offsets), (3) do crop
and sample colors agree (per-channel ratio)?

Example:
    python scratch/photometry/prototypes/register_samples.py Day2_Tank2 out_dir
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path('/media/mattw/CUTTLE/raw_data')
SAMPLES = Path('/media/mattw/CUTTLE/raw_data_samples')
RESULTS = Path('/media/mattw/CUTTLE/results/.iter-1.0')
CROP_SIDE = 512
OFFSETS = (-3, -2, -1, 0, 1, 2)  # candidate (video frame index) - (sample frame label)


def crop_to_sample_affine(
    bbox_row: pd.Series,
    prefix: str,
    canvas_side: float,
    sample_scale: float,
) -> np.ndarray:
    """Affine mapping crop pixel indices to sample-image pixel indices.

    Args:
        bbox_row: bbox csv row
        prefix: 'raw' or 'main' column prefix
        canvas_side: crop window side in original px
        sample_scale: sample px per original px

    Returns:
        2x3 affine for cv2.warpAffine (pixel-center convention)
    """
    cx = (bbox_row[f'orig_{prefix}_x0'] + bbox_row[f'orig_{prefix}_x1']) / 2
    cy = (bbox_row[f'orig_{prefix}_y0'] + bbox_row[f'orig_{prefix}_y1']) / 2
    orig_per_crop = canvas_side / CROP_SIDE
    k = orig_per_crop * sample_scale
    tx = sample_scale * (cx + (0.5 - CROP_SIDE / 2) * orig_per_crop) - 0.5
    ty = sample_scale * (cy + (0.5 - CROP_SIDE / 2) * orig_per_crop) - 0.5
    return np.array([[k, 0, tx], [0, k, ty]], dtype=np.float64)


def match_bbox(
    img: np.ndarray,
    bboxes_frame: pd.DataFrame,
    prefix: str,
    scale: float,
) -> pd.Series | None:
    """Pick the bbox row whose scaled size matches the crop's visible extent."""
    ys, xs = np.where(img.max(2) > 25)
    if len(xs) < 2000:
        return None
    width, height = xs.max() - xs.min(), ys.max() - ys.min()
    best = None
    for _, row in bboxes_frame.iterrows():
        bbox_w = row[f'orig_{prefix}_x1'] - row[f'orig_{prefix}_x0']
        bbox_h = row[f'orig_{prefix}_y1'] - row[f'orig_{prefix}_y0']
        err = max(abs(width - scale * bbox_w), abs(height - scale * bbox_h))
        if best is None or err < best[0]:
            best = (err, row)
    return best[1] if best[0] < 12 else None


def main(session: str, out_dir: Path) -> None:
    """Run the registration test for one session and print/plot results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    day, tank = session.split('_')
    bboxes = pd.read_csv(RAW / f'{session}_bboxes.csv').set_index('frame')
    canvas = pd.read_csv(RAW / 'canvas_sides.csv')
    ingest = pd.read_parquet(RESULTS / 'manifests' / 'ingest.parquet')
    manifest = pd.read_csv(SAMPLES / session / '_manifest.csv')
    sample_scale = manifest.out_w.iloc[0] / bboxes.orig_w_full.iloc[0]
    print(f'sample scale {sample_scale:.5f} (sample w {manifest.out_w.iloc[0]})')

    results = []
    caps = {}
    for fish_idx in (1, 2):
        crow = canvas[
            (canvas.day == f'DAY_{day[3:]}') & (canvas.tank == f'Tank_{tank[4:]}')
            & (canvas.cuttle == fish_idx)
        ].iloc[0]
        vid = ingest[
            (ingest.session_id == session) & (ingest.fish_id.str.startswith(f'Cuttle{fish_idx}'))
        ].video_path.iloc[0]
        caps[fish_idx] = (
            cv2.VideoCapture(vid), crow.canvas_side, 'raw' if crow.box == 'raw' else 'main',
        )

    for _, srow in manifest.iloc[20:980:12].iterrows():
        sample = cv2.imread(str(SAMPLES / session / Path(srow.out_file).name)).astype(np.float32)
        for fish_idx, (cap, canvas_side, prefix) in caps.items():
            for off in OFFSETS:
                frame = int(srow.frame) + off
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, crop = cap.read()
                if not ok or frame not in bboxes.index:
                    continue
                brow = match_bbox(
                    crop, bboxes.loc[[frame]], prefix, CROP_SIDE / canvas_side,
                )
                if brow is None:
                    continue
                affine = crop_to_sample_affine(brow, prefix, canvas_side, sample_scale)
                k = affine[0, 0]
                crop_f = crop.astype(np.float32)
                blurred = cv2.GaussianBlur(crop_f, (0, 0), sigmaX=0.5 / k)
                shape = (sample.shape[1], sample.shape[0])
                warped = cv2.warpAffine(blurred, affine, shape, flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(
                    (crop.max(2) > 25).astype(np.float32), affine, shape, flags=cv2.INTER_LINEAR,
                    # note: blurred-binary mask, then eroded below
                )
                inner = cv2.erode((mask > 0.98).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                if inner.sum() < 100:
                    continue
                diff = np.abs(warped[inner] - sample[inner]).mean()
                ratio = warped[inner].mean(0) / sample[inner].mean(0)
                results.append({
                    'label': int(srow.frame), 'fish': fish_idx, 'offset': off,
                    'mad': diff, 'n_px': int(inner.sum()),
                    'ratio_b': ratio[0], 'ratio_g': ratio[1], 'ratio_r': ratio[2],
                })
    df = pd.DataFrame(results)
    df.to_csv(out_dir / f'register_{session}.csv', index=False)
    print(df.groupby('offset')[['mad', 'ratio_b', 'ratio_g', 'ratio_r']].median().round(3))
    print(df.groupby('offset').size())


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]))
