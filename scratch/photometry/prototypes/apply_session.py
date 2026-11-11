"""Apply the fitted photometric correction to one session's iter-1.0 frames (crop space).

For each gate-passing iter-1.0 anchor frame: read the 512 crop, map each crop pixel to sample
coordinates (bbox + canvas_side), look up the fitted flat-field map and per-frame gain there,
correct in linear light, then warp to the 200x100 canonical frame with the *existing* inscribed
rectangle corners (`results/rectangles/*.csv`). Also writes a raw re-export (same warp, no
correction) as a control, since the original frames passed through an extra mp4 encode in
`cuttle inscribe`.

Crop and sample colors differ by a fixed per-channel gain (measured here, per session); the
correction is applied in the sample color domain.

Example:
    python scratch/photometry/prototypes/apply_session.py Day2_Tank2 qc_dir
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from register_samples import (  # noqa: E402
    CROP_SIDE, RAW, SAMPLES, crop_to_sample_affine, match_bbox,
)

from cuttle_patterns.preprocessing.align import CORNER_COLUMNS, warp_to_canonical  # noqa: E402

RES = Path('/media/mattw/CUTTLE/results')
OUT = RES / '.iter-1.0-photometry'
CANON = (200, 100)
FLOOR_LUM_MIN = 0.2  # sample-brightness (linear) at/above which crop-vs-sample gain is fit
FLAT_MIN = 0.3  # floor on the flat-field map to avoid blow-up near bright walls / outside tank


def to_linear(img_u8: np.ndarray) -> np.ndarray:
    """sRGB uint8 to linear float32."""
    x = img_u8.astype(np.float32) / 255
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def to_srgb_u8(lin: np.ndarray) -> np.ndarray:
    """Linear float to sRGB uint8."""
    x = np.clip(lin, 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (x * 255 + 0.5).astype(np.uint8)


def crop_maps(affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-crop-pixel sample coordinates (map_x, map_y) for cv2.remap."""
    u, v = np.meshgrid(np.arange(CROP_SIDE), np.arange(CROP_SIDE))
    return (
        (affine[0, 0] * u + affine[0, 2]).astype(np.float32),
        (affine[1, 1] * v + affine[1, 2]).astype(np.float32),
    )


def calibrate_crop_vs_sample(
    session: str,
    fish: dict,
    manifest: pd.DataFrame,
    bboxes: pd.DataFrame,
    sample_scale: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Per-channel linear gain c with crop ~ c * sample, at floor-level brightness.

    The crop/sample ratio depends on brightness, so c is fit only on animal pixels at least as
    bright as `FLOOR_LUM_MIN` (the floor's level), where the floor gain is defined.

    Returns:
        (c in BGR order, DataFrame of ratio by sample-brightness bin)
    """
    num, den = np.zeros(3), np.zeros(3)  # all animal pixels
    num_b, den_b = np.zeros(3), np.zeros(3)  # floor-level brightness only
    pairs = []
    pix = []
    for _, srow in manifest.iloc[20:980:12].iterrows():
        sample = to_linear(cv2.imread(str(SAMPLES / session / Path(srow.out_file).name)))
        frame = int(srow.frame) - 1
        for info in fish.values():
            cap = info['cap']
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, crop = cap.read()
            if not ok or frame not in bboxes.index:
                continue
            brow = match_bbox(crop, bboxes.loc[[frame]], info['prefix'], CROP_SIDE / info['canvas'])
            if brow is None:
                continue
            affine = crop_to_sample_affine(brow, info['prefix'], info['canvas'], sample_scale)
            k = affine[0, 0]
            blurred = cv2.GaussianBlur(to_linear(crop), (0, 0), sigmaX=0.5 / k)
            shape = (sample.shape[1], sample.shape[0])
            warped = cv2.warpAffine(blurred, affine, shape, flags=cv2.INTER_LINEAR)
            mask = cv2.warpAffine((crop.max(2) > 25).astype(np.float32), affine, shape)
            inner = cv2.erode((mask > 0.98).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if inner.sum() < 100:
                continue
            w, s = warped[inner], sample[inner]
            num += (w * s).sum(0)
            den += (s * s).sum(0)
            bright = s.mean(1) > FLOOR_LUM_MIN
            num_b += (w[bright] * s[bright]).sum(0)
            den_b += (s[bright] ** 2).sum(0)
            sel = np.random.default_rng(0).choice(len(w), min(len(w), 400), replace=False)
            pix.append(np.stack([s[sel], w[sel]]))
            pairs.append(np.column_stack([s.mean(1), w.mean(1) / np.maximum(s.mean(1), 1e-4)]))
    c_all, c_bright = num / den, num_b / den_b
    print(f'crop/sample gain, all animal pixels (BGR): {c_all.round(4)}')
    c = c_bright
    pairs = np.concatenate(pairs)
    bins = pd.qcut(pairs[:, 0], 8)
    table = pd.DataFrame({'sample_lum': pairs[:, 0], 'ratio': pairs[:, 1]}).groupby(
        bins, observed=True).median()
    return c, table, np.concatenate(pix, axis=1)


CURVE_BINS = 16


def fit_tone_curve(pix: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-channel monotone map from crop linear value to sample linear value.

    Args:
        pix: (2, N, 3) array of (sample, crop-warped) linear animal pixels

    Returns:
        per channel (crop knots, sample knots) for np.interp, anchored at the origin and
        extended past the top bin with the top bin's ratio
    """
    curves = []
    for ch in range(3):
        samp, crop = pix[0, :, ch], pix[1, :, ch]
        edges = np.quantile(samp, np.linspace(0, 1, CURVE_BINS + 1))
        idx = np.clip(np.digitize(samp, edges[1:-1]), 0, CURVE_BINS - 1)
        xs = np.array([np.median(samp[idx == k]) for k in range(CURVE_BINS)])
        ys = np.array([np.median(crop[idx == k]) for k in range(CURVE_BINS)])
        ys = np.maximum.accumulate(ys)  # enforce monotone
        keep = np.concatenate([[True], np.diff(ys) > 1e-6])
        ys, xs = ys[keep], xs[keep]
        top = xs[-1] / ys[-1]
        crop_knots = np.concatenate([[0.0], ys, [ys[-1] * 20]])
        samp_knots = np.concatenate([[0.0], xs, [xs[-1] * 20 * 1.0]])
        samp_knots[-1] = xs[-1] + (crop_knots[-1] - ys[-1]) * top
        curves.append((crop_knots, samp_knots))
    return curves


def main(session: str, qc_dir: Path, curve: bool = False, out_suffix: str = '') -> None:
    """Correct and export the session's iter-1.0 frames; print control statistics."""
    qc_dir.mkdir(parents=True, exist_ok=True)
    day, tank = session.split('_')
    param_from = os.environ.get('PARAM_FROM', session)  # wrong-parameter control: other session
    params = np.load(OUT / 'params' / f'{param_from}.npz')
    fog, flat, target = params['fog'], params['flat'], params['target']
    sample_idx, gains = params['sample_frame_idx'], params['gains']
    if os.environ.get('TARGET_OWN'):  # brightness: the (borrowed or own) session's floor level
        target = np.full(3, float(params['floor_ref']), dtype=np.float32)
    if os.environ.get('FOG_OFF'):  # diagnostic: skip fog subtraction
        fog = np.zeros_like(fog)
    bboxes = pd.read_csv(RAW / f'{session}_bboxes.csv').set_index('frame')
    canvas = pd.read_csv(RAW / 'canvas_sides.csv')
    manifest = pd.read_csv(SAMPLES / session / '_manifest.csv')
    sample_scale = manifest.out_w.iloc[0] / bboxes.orig_w_full.iloc[0]
    if flat.shape[1] != manifest.out_w.iloc[0]:  # borrowed map from a different-resolution session
        flat = cv2.resize(flat, (int(manifest.out_w.iloc[0]), int(manifest.out_h.iloc[0])))
    ingest = pd.read_parquet(RES / '.iter-1.0' / 'manifests' / 'ingest.parquet')
    extract = pd.read_parquet(RES / '.iter-1.0' / 'manifests' / 'extract.parquet')
    gate = pd.read_parquet(OUT / 'manifests' / 'frame_gate.parquet')

    fish = {}
    for fish_id in sorted(extract[extract.session_id == session].fish_id.unique()):
        cuttle = int(fish_id.split('_')[0].removeprefix('Cuttle'))
        crow = canvas[
            (canvas.day == f'DAY_{day[3:]}') & (canvas.tank == f'Tank_{tank[4:]}')
            & (canvas.cuttle == cuttle)
        ].iloc[0]
        video = ingest[(ingest.session_id == session) & (ingest.fish_id == fish_id)].video_path.iloc[0]
        fish[fish_id] = dict(
            cap=cv2.VideoCapture(video), canvas=crow.canvas_side,
            prefix='raw' if crow.box == 'raw' else 'main', stem=Path(video).stem,
        )

    c, table, pix = calibrate_crop_vs_sample(session, fish, manifest, bboxes, sample_scale)
    print(f'crop/sample linear gain per channel (BGR): {c.round(4)}')
    print('ratio crop/sample by sample brightness bin (linear):')
    print(table.round(3).to_string())
    np.save(OUT / 'params' / f'{session}_crop_vs_sample.npy', c)
    if curve:
        print('tone curve knots (crop->sample, ch0):', [np.round(k, 3).tolist() for k in fit_tone_curve(pix)[0]])

    curves = fit_tone_curve(pix) if curve else None
    rows = []
    for fish_id, info in fish.items():
        corners = pd.read_csv(RES / '.iter-1.0' / 'rectangles' / f'{info["stem"]}.csv')[CORNER_COLUMNS]
        corners = corners.to_numpy().reshape(-1, 4, 2)
        sel = extract[(extract.session_id == session) & (extract.fish_id == fish_id)]
        ok_idx = set(gate[
            (gate.session_id == session) & (gate.fish_id == fish_id) & gate.is_anchor
            & (gate.status == 'pass')
        ].frame_idx)
        out_dirs = {
            name: OUT / (name + out_suffix) / info['stem']
            for name in ('beast_frames', 'beast_frames_rawreexport')
        }
        for d in out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        for frame in sorted(sel.frame_idx):
            if frame not in ok_idx:
                continue
            info['cap'].set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, crop = info['cap'].read()
            if not ok:
                continue
            brow = match_bbox(crop, bboxes.loc[[frame]], info['prefix'], CROP_SIDE / info['canvas'])
            if brow is None:
                continue
            affine = crop_to_sample_affine(brow, info['prefix'], info['canvas'], sample_scale)
            map_x, map_y = crop_maps(affine)
            flat_crop = np.maximum(
                cv2.remap(flat, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
                FLAT_MIN,
            )
            gain = np.array([np.interp(frame, sample_idx, gains[:, ch]) for ch in range(3)])
            lin = to_linear(crop) / c
            if curves is not None:
                crop_lin = to_linear(crop)
                lin = np.stack(
                    [np.interp(crop_lin[..., ch], *curves[ch]) for ch in range(3)], axis=-1,
                ).astype(np.float32)
            corrected = np.maximum(lin - fog, 0) / (flat_crop * gain) * target
            corrected_u8 = to_srgb_u8(corrected)
            name = f'img{frame:08d}.png'
            cv2.imwrite(str(out_dirs['beast_frames'] / name),
                        warp_to_canonical(corrected_u8, corners[frame], CANON))
            raw_re = warp_to_canonical(crop, corners[frame], CANON)
            cv2.imwrite(str(out_dirs['beast_frames_rawreexport'] / name), raw_re)
            orig = cv2.imread(str(RES / '.iter-1.0' / 'beast_frames' / info['stem'] / name))
            rows.append({
                'fish_id': fish_id, 'frame': frame,
                'mad_orig_vs_rawreexport': float(np.abs(orig.astype(float) - raw_re).mean())
                if orig is not None else np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(qc_dir / f'apply_{session}.csv', index=False)
    print(f'wrote {len(df)} frames; original vs raw re-export mean abs diff (0-255):')
    print(df.groupby('fish_id').mad_orig_vs_rawreexport.describe().round(2).to_string())


if __name__ == '__main__':
    suffix = sys.argv[3] if len(sys.argv) > 3 else ''  # '_curve' / '_wrong' use the tone curve
    main(sys.argv[1], Path(sys.argv[2]), curve=bool(suffix), out_suffix=suffix)
