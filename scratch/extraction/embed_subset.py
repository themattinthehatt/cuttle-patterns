"""Embed eligible frames: DINOv3-S CLS on aligned rectangle videos, classifier on 512x512 crops.

For each video, decodes sequentially (skipping ineligible frames without decoding) and writes
under `<out-dir>/<video>/`:
    frame_idx.npy       int32 (n,)   eligible frame indices
    dino_cls.npy        float16 (n, 384)   DINOv3 ViT-S/16 CLS at 224 px
    clf_feat.npy        float16 (n, 512)   classifier penultimate activation
    clf_prob.npy        float32 (n, 6)     classifier softmax

Eligibility comes from the parquet tables written by `count_eligible.py`.

Usage:
    python embed_subset.py --elig-dir <dir> --out-dir <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from cuttle_patterns.config import load_config
from cuttle_patterns.embedders.dinov3 import DINOv3Backbone

_logger = logging.getLogger(__name__)

CLF_DIR = Path('/media/mattw/CUTTLE/classification/2026-09-09_classifier_model')
DEFAULT_VIDEOS = [
    'Day5_Tank3_Cuttle2_Resident_Crop',
    'Day4_Tank1_Cuttle1_Resident_Crop',
    'Day1_Tank2_Cuttle1_Resident_Crop',
    'Day1_Tank5_Cuttle1_Resident_Crop',
    'Day7_Tank6_Cuttle2_Resident_Crop',
    'Day8_Tank3_Cuttle2_Intruder_Crop',
    'Day2_Tank1_Cuttle1_Resident_Crop',
    'Day12_Tank2_Cuttle2_Resident_Crop',
]
BATCH = 256
IMG = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_classifier(device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Load the skin-pattern resnet18 and hook its penultimate activation.

    Args:
        device: target device.

    Returns:
        (model, captured) where captured['value'] holds the (B, 512) features after a call.
    """
    n_classes = len(json.loads((CLF_DIR / 'behaviors.json').read_text()))
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(CLF_DIR / 'skin_pattern_resnet18_best.pt', map_location=device))
    model.to(device).eval()
    captured = {}
    model.avgpool.register_forward_hook(
        lambda m, i, o: captured.__setitem__('value', torch.flatten(o, 1))
    )
    return model, captured


@torch.no_grad()
def process_batch(
    rect_frames: list[np.ndarray],
    crop_frames: list[np.ndarray],
    backbone,
    clf,
    captured,
    device,
) -> tuple:
    """Embed one batch: DINOv3 on aligned rectangle frames, classifier on 512x512 crops.

    Args:
        rect_frames: list of aligned-rectangle bgr uint8 frames (200x100).
        crop_frames: list of 512x512 bgr uint8 crop frames.
        backbone: `DINOv3Backbone`.
        clf: classifier.
        captured: feature-hook dict from `load_classifier`.
        device: torch device.

    Returns:
        (dino_cls, clf_feat, clf_prob) as numpy arrays.
    """
    rgb_rect = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in rect_frames])
    with torch.autocast('cuda', dtype=torch.float16):
        dino = backbone.forward(backbone.preprocess(rgb_rect)).cls.float()
    rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in crop_frames])
    x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(IMG, IMG), mode='bilinear', antialias=True, align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    prob = F.softmax(clf(x), dim=1)
    return dino.cpu().numpy(), captured['value'].cpu().numpy(), prob.cpu().numpy()


def main() -> None:
    """Embed the chosen videos' eligible frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--elig-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--videos', nargs='*', default=DEFAULT_VIDEOS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    device = torch.device('cuda')
    backbone = DINOv3Backbone('vits16', IMG, device)
    clf, captured = load_classifier(device)
    config = load_config()
    data_dir = config.data_dir
    rect_dir = config.results_dir / 'rectangles'

    for stem in args.videos:
        out = args.out_dir / stem
        if (out / 'clf_prob.npy').exists():
            _logger.info(f'{stem}: done already, skipping')
            continue
        out.mkdir(parents=True, exist_ok=True)
        elig = pd.read_parquet(args.elig_dir / f'{stem}.parquet')
        idx_elig = elig.loc[elig['eligible'], 'frame_idx'].to_numpy()
        want = set(idx_elig.tolist())
        cap = cv2.VideoCapture(str(data_dir / f'{stem}.mp4'))  # 512x512 crop (classifier)
        cap_rect = cv2.VideoCapture(str(rect_dir / f'{stem}.mp4'))  # aligned (DINOv3)
        dino, feat, prob, buf, buf_rect = [], [], [], [], []

        def flush() -> None:
            d, f, p = process_batch(buf_rect, buf, backbone, clf, captured, device)
            buf_rect.clear()
            dino.append(d.astype(np.float16))
            feat.append(f.astype(np.float16))
            prob.append(p)
            buf.clear()

        for idx in range(len(elig)):
            if idx not in want:
                cap.grab()
                cap_rect.grab()
                continue
            ok, frame = cap.read()
            ok_rect, frame_rect = cap_rect.read()
            if not (ok and ok_rect):
                raise RuntimeError(f'{stem}: failed to read frame {idx}')
            buf.append(frame)
            buf_rect.append(frame_rect)
            if len(buf) == BATCH:
                flush()
            if idx % 5000 == 0:
                _logger.info(f'{stem}: frame {idx}/{len(elig)}')
        if buf:
            flush()
        cap.release()
        cap_rect.release()
        np.save(out / 'frame_idx.npy', idx_elig.astype(np.int32))
        np.save(out / 'dino_cls.npy', np.concatenate(dino))
        np.save(out / 'clf_feat.npy', np.concatenate(feat))
        np.save(out / 'clf_prob.npy', np.concatenate(prob))
        _logger.info(f'{stem}: embedded {len(idx_elig)} frames')


if __name__ == '__main__':
    main()
