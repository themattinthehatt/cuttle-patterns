"""Fit per-session photometric parameters (fog, fish-free flat-field map, per-frame gain).

Model, in linear light, on the full-frame samples:
    corrected = (pixel - fog) / (map * gain_t) * target
- fog: per-channel offset = darkest animal pixels (p0.5, full resolution) inside the floor window
- map: per-pixel fish-free picture of the empty tank, brightest-60% median of (frame - fog),
  normalized to 1 at its tank-box median
- gain_t: per-frame per-channel scale = median of (frame - fog) / map over fish-free floor pixels,
  median-smoothed over time
- target: floor value after correction (per channel); defaults to a neutral gray with the
  session's own floor luminance (a global target is set once all sessions are fitted)

Also reports held-out floor flatness (map fit on even samples, tested on odd ones).

Example:
    python scratch/photometry/prototypes/fit_session.py Day2_Tank2 qc_dir  # floor = central window
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SAMPLES = Path('/media/mattw/CUTTLE/raw_data_samples')
OUT_ROOT = Path('/media/mattw/CUTTLE/results/.iter-1.0-photometry/params')
CENTER = (0.25, 0.75)  # central window (fraction of frame) used as floor; checked by eye per session
POOL = 2
BRIGHT_PCT = 40  # per-pixel: keep frames at or above this brightness percentile (fish are dark)
FLOOR_TOL = 0.1  # floor pixels: within 10% of the map


def to_linear(img_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 (BGR order kept) to linear float32."""
    x = img_u8.astype(np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def pool(img: np.ndarray) -> np.ndarray:
    """Average-pool an HxWx3 array by POOL."""
    h, w = (img.shape[0] // POOL) * POOL, (img.shape[1] // POOL) * POOL
    return img[:h, :w].reshape(h // POOL, POOL, w // POOL, POOL, 3).mean((1, 3))


def box_slice(box: list[int]) -> tuple[slice, slice]:
    """Pooled-coordinate slices for an (x0, y0, x1, y1) box in sample pixels."""
    x0, y0, x1, y1 = (v // POOL for v in box)
    return slice(y0, y1), slice(x0, x1)


def fit_map(frames: np.ndarray, fog: np.ndarray, tank: tuple[slice, slice]) -> np.ndarray:
    """Fish-free flat-field map from frames (N, H, W, 3), normalized to 1 over the tank box."""
    adj = np.maximum(frames - fog, 1e-4)
    lum = adj.mean(3)
    keep = lum >= np.percentile(lum, BRIGHT_PCT, axis=0)
    raw_map = np.nanmedian(np.where(keep[..., None], adj, np.nan), 0)
    return raw_map / np.median(raw_map[tank].reshape(-1, 3), 0)


def frame_gain(frame: np.ndarray, fog: np.ndarray, flat: np.ndarray, tank: tuple) -> np.ndarray:
    """Per-channel gain of one frame relative to the map, from fish-free floor pixels."""
    ratio = np.maximum(frame - fog, 1e-4)[tank] / flat[tank]
    lum = ratio.mean(2)
    med = np.median(lum)
    floor = np.abs(lum / med - 1) < FLOOR_TOL
    return np.median(ratio[floor], 0)


def spread(img: np.ndarray, tank: tuple, floor: np.ndarray) -> tuple[float, float]:
    """Relative luminance spread (p5-p95 / median) and B/R spread over floor pixels."""
    sub = img[tank][floor]
    lum = sub.mean(1)
    b_over_r = sub[:, 0] / np.maximum(sub[:, 2], 1e-4)  # BGR order
    return (
        float(np.ptp(np.percentile(lum, [5, 95])) / np.median(lum)),
        float(np.ptp(np.percentile(b_over_r, [5, 95]))),
    )


def main(session: str, qc_dir: Path) -> None:
    """Fit and save parameters for one session, plus QC figures and held-out flatness."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(SAMPLES / session / '_manifest.csv')
    files = [SAMPLES / session / Path(f).name for f in manifest.out_file]
    frame_idx = manifest.frame.to_numpy() - 1  # sample label N is video frame N-1
    frames = np.stack([pool(to_linear(cv2.imread(str(f)))) for f in files])
    height, width = frames.shape[1] * POOL, frames.shape[2] * POOL
    tank_full = [
        int(CENTER[0] * width), int(CENTER[0] * height), int(CENTER[1] * width),
        int(CENTER[1] * height),
    ]
    tank = box_slice(tank_full)
    print(f'{len(frames)} samples, pooled shape {frames.shape[1:3]}; floor window {tank_full}')

    start = int(0.03 * len(frames))  # skip the divider phase for the map

    # pass 1: no fog -> map, gains, fish masks; then fog from the darkest fish pixels (full res)
    flat0 = fit_map(frames[start:], np.zeros(3, np.float32), tank)
    gains0 = np.stack([frame_gain(f, np.zeros(3, np.float32), flat0, tank) for f in frames])
    has_tank = gains0.mean(1) > 0.5 * np.median(gains0.mean(1))  # drop blank/non-tank frames
    flat0_full = cv2.resize(
        flat0.astype(np.float32), (frames.shape[2] * POOL, frames.shape[1] * POOL),
        interpolation=cv2.INTER_LINEAR,
    )
    dark = []
    x0, y0, x1, y1 = tank_full
    for k in range(start, len(frames), 6):
        if not has_tank[k]:
            continue
        full = to_linear(cv2.imread(str(files[k])))[y0:y1, x0:x1]
        expect = flat0_full[y0:y1, x0:x1] * gains0[k]
        fish = full.mean(2) < 0.6 * expect.mean(2)
        if fish.sum() >= 200:
            dark.append(np.percentile(full[fish], 0.5, axis=0))
    fog = np.median(dark, 0).astype(np.float32)
    print(f'fog from darkest animal pixels (p0.5, BGR, linear): {fog.round(4)} '
          f'(from {len(dark)} frames)')

    flat = fit_map(frames[start:], fog, tank)
    gains = np.stack([frame_gain(f, fog, flat, tank) for f in frames])
    valid = has_tank
    for c in range(3):  # interpolate over frames without a tank, then smooth
        gains[:, c] = np.interp(frame_idx, frame_idx[valid], gains[valid, c])
    gains_smooth = median_filter(gains, size=(9, 1), mode='nearest')
    floor_ref = np.median(gains_smooth[valid].mean(1))
    target = np.full(3, floor_ref, dtype=np.float32)  # neutral gray at the session's floor level

    # held-out flatness: map fit on even samples, evaluated on odd samples
    even, odd = np.arange(start, len(frames), 2), np.arange(start + 1, len(frames), 2)
    flat_even = fit_map(frames[even], fog, tank)
    rows = []
    for k in odd:
        g = frame_gain(frames[k], fog, flat_even, tank)
        raw_lum = frames[k][tank].mean(2)
        floor = np.abs(raw_lum / np.median(raw_lum) - 1) < 0.15
        if floor.sum() < 100 or not has_tank[k]:
            continue  # blank or non-tank frame
        variants = {
            'raw': frames[k],
            'fog+gain': np.maximum(frames[k] - fog, 0) / np.median(g),
            'fog+map': np.maximum(frames[k] - fog, 0) / flat_even,
            'fog+map+gain': np.maximum(frames[k] - fog, 0) / (flat_even * g),
        }
        for name, img in variants.items():
            lum_spread, br_spread = spread(img, tank, floor)
            rows.append({'variant': name, 'lum_spread': lum_spread, 'br_spread': br_spread})
    res = pd.DataFrame(rows).groupby('variant').mean().loc[
        ['raw', 'fog+gain', 'fog+map', 'fog+map+gain']
    ]
    print('\nheld-out floor spread within the tank box (lower = flatter)')
    print(res.round(3).to_string())
    res.to_csv(qc_dir / f'flatness_{session}.csv')

    # save parameters (full-resolution map via upsampling)
    full_h, full_w = frames.shape[1] * POOL, frames.shape[2] * POOL
    flat_full = cv2.resize(flat.astype(np.float32), (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_ROOT / f'{session}.npz', fog=fog, flat=flat_full, target=target,
        sample_frame_idx=frame_idx, gains=gains_smooth, gains_raw=gains,
        has_tank=has_tank, tank=np.array(tank_full), floor_ref=floor_ref,
    )

    # qc figure
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    ax[0, 0].plot(frame_idx, gains_smooth.mean(1))
    ax[0, 0].set_title('mean gain over video')
    for c, col in enumerate('bgr'):
        ax[0, 1].plot(frame_idx, gains[:, c], col, alpha=0.3)
        ax[0, 1].plot(frame_idx, gains_smooth[:, c], col)
    ax[0, 1].set_title('per-frame gain B/G/R (raw, smoothed)')
    ax[0, 2].plot(frame_idx, gains_smooth[:, 0] / gains_smooth[:, 2])
    ax[0, 2].set_title('gain B/R over video')
    to_u8 = lambda x: (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)  # noqa: E731
    ax[1, 0].imshow(to_u8(flat / flat.max())[..., ::-1])
    ax[1, 0].set_title('flat-field map (normalized)')
    mid = len(frames) // 2
    raw_mid = frames[mid]
    cor_mid = np.maximum(raw_mid - fog, 0) / (flat * gains_smooth[mid]) * target
    ax[1, 1].imshow(to_u8(raw_mid)[..., ::-1])
    ax[1, 1].set_title(f'raw sample {frame_idx[mid]}')
    ax[1, 2].imshow(to_u8(cor_mid)[..., ::-1])
    ax[1, 2].set_title('corrected (fog, map, gain; floor -> neutral)')
    for a in ax[0]:
        a.set_xlabel('video frame')
    plt.tight_layout()
    plt.savefig(qc_dir / f'fit_{session}.png', dpi=70)


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]))
