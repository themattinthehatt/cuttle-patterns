"""Propose the tank-floor box per session from its own fish-free median frame (to be verified by eye).

For each session in `.iter-1.0`, builds a temporal median of the 771x434 samples (skipping the
divider phase), finds the inner floor rectangle from the strongest luminance edges along the
central row/column band, and writes a verification montage with the proposed boxes drawn.
Boxes are saved to `boxes_proposed.json` (tank box only; drain/spout are diagnostics).

Example:
    python scratch/photometry/propose_boxes.py out_dir
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SAMPLES = Path('/media/mattw/CUTTLE/raw_data_samples')
EXTRACT = Path('/media/mattw/CUTTLE/results/.iter-1.0/manifests/extract.parquet')


def median_frame(session: str) -> np.ndarray:
    """Temporal median (uint8 BGR) of a session's samples, skipping the first 10%."""
    files = sorted((SAMPLES / session).glob('*.png'))
    files = files[len(files) // 10::5]
    return np.median(np.stack([cv2.imread(str(f)) for f in files]), 0).astype(np.uint8)


def edge_positions(profile: np.ndarray, lo: int, hi: int, sign: int) -> int:
    """Index in [lo, hi) of the strongest smoothed gradient of the given sign."""
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1).astype(np.float32), (1, 0), 2).ravel()
    grad = np.gradient(smooth)
    seg = grad[lo:hi] * sign
    return lo + int(np.argmax(seg))


def propose(med: np.ndarray) -> list[int]:
    """Propose (x0, y0, x1, y1) of the floor interior from edge profiles."""
    h, w = med.shape[:2]
    lum = med.mean(2)
    cols = np.median(lum[int(h * 0.35):int(h * 0.65)], 0)
    rows = np.median(lum[:, int(w * 0.35):int(w * 0.65)], 1)
    # floor is darker than the bright wall glow: falling edge on the left/top, rising on the
    # right/bottom
    x0 = edge_positions(cols, int(w * 0.10), int(w * 0.40), -1)
    x1 = edge_positions(cols, int(w * 0.60), int(w * 0.92), +1)
    y0 = edge_positions(rows, int(h * 0.05), int(h * 0.35), -1)
    y1 = edge_positions(rows, int(h * 0.65), int(h * 0.97), +1)
    return [x0, y0, x1, y1]


def main(out_dir: Path) -> None:
    """Propose boxes for every iter-1.0 session and write montages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = sorted(pd.read_parquet(EXTRACT).session_id.unique())
    boxes, tiles = {}, []
    for session in sessions:
        med = median_frame(session)
        box = propose(med)
        boxes[session] = {'frame_size': [med.shape[1], med.shape[0]], 'tank': box}
        vis = med.copy()
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.putText(vis, f'{session} {box}', (5, 15), 0, 0.5, (0, 255, 255), 1)
        tiles.append(vis)
        print(session, med.shape[:2], box)
    json.dump(boxes, open(Path(__file__).parent / 'boxes_proposed.json', 'w'), indent=1)
    for i in range(0, len(tiles), 4):
        group = tiles[i:i + 4]
        if len(group) < 4:
            group += [np.zeros_like(tiles[0])] * (4 - len(group))
        cv2.imwrite(
            str(out_dir / f'proposed_{i // 4}.png'),
            np.vstack([np.hstack(group[:2]), np.hstack(group[2:])]),
        )


if __name__ == '__main__':
    main(Path(sys.argv[1]))
