"""Figures and contact sheets for the photometry / Day 1 fog slides.

Reads the saved results (split csv, outlier tables, eval text outputs, sample frames, variant
frames) and writes PNGs. Colors follow the dataviz reference palette (light surface).

Example:
    python scratch/photometry/make_slide_figures.py scratchpad_dir figs_dir
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, GRAY, SLATE = '#2a78d6', '#eb6834', '#1baf7a', '#9a9993', '#52514e'
SAMPLES = Path('/media/mattw/CUTTLE/raw_data_samples')
VARIANTS = Path('/media/mattw/CUTTLE/results/.iter-1.0-photometry/variants')
NAMES = {
    'vgg19_1_448_gram_k64_taper': 'VGG-19 relu1_1 Gram',
    'vgg19_12345_448_gram_k28_taper': 'VGG-19 layers 1-5 Gram',
    'vgg19_5_448_gram_k64_taper': 'VGG-19 relu5_1 Gram',
    'dinov3_vitb16_224_cls': 'DINOv3 CLS',
    'dinov3_vitb16_224_meanpatch_taper': 'DINOv3 mean-patch',
    'dinov3_vitb16_224_gram_k64_taper': 'DINOv3 Gram 224',
    'dinov3_vitb16_448_gram_k64_taper': 'DINOv3 Gram 448',
    'iter-1.1_resnet-18_d16': 'resnet-18 (trained)',
    'iter-1.1_msps-vae-mask_d16_z_u': 'MSPS-VAE-mask z_u',
    'iter-1.1_msps-vae-mask_d16_z_b': 'MSPS-VAE-mask z_b',
}


def style() -> None:
    """Global matplotlib look."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
        'axes.edgecolor': GRID, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 15,
        'axes.axisbelow': True, 'font.family': 'DejaVu Sans',
    })


def fig_scene(scratch: Path, out: Path) -> None:
    """Scene stats by day: dark level and floor blue/red for all 72 sessions."""
    s = pd.read_csv(scratch / 'outl' / 'scene_outliers.csv')
    s['day'] = s.session.str.extract(r'Day(\d+)').astype(int)
    s['tank'] = s.session.str.extract(r'Tank(\d+)').astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, col, title in (
        (axes[0], 'dark_over_floor', 'Dark level relative to floor brightness'),
        (axes[1], 'floor_B_R', 'Floor color: blue / red'),
    ):
        rest = s[s.day > 1]
        ax.scatter(rest.day + np.random.default_rng(0).uniform(-.2, .2, len(rest)), rest[col],
                   s=60, color=GRAY, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        d1 = s[s.day == 1]
        ax.scatter(d1.day, d1[col], s=90, color=ORANGE, edgecolor=SURFACE, linewidth=1.5,
                   zorder=4)
        for r in d1.itertuples():
            dy = {'dark_over_floor': {5: -9, 6: 9}, 'floor_B_R': {3: 8, 5: -8}}[col].get(r.tank, 0)
            ax.annotate(f'T{r.tank}', (r.day, getattr(r, col)), xytext=(10, dy),
                        textcoords='offset points', color=INK, fontsize=14, va='center')
        ax.set_xticks(range(1, 13))
        ax.set_xlabel('day')
        ax.set_title(title, loc='left', fontsize=17, color=INK)
    axes[0].set_ylabel('p0.5 dark level / floor level')
    axes[1].set_ylabel('floor B / R')
    fig.tight_layout()
    fig.savefig(out / 'fig_scene_by_day.png', dpi=130)
    plt.close(fig)


def fig_animal_black(scratch: Path, out: Path) -> None:
    """Animal black level per video for every iter-1.0 session."""
    a = pd.read_csv(scratch / 'outl' / 'animal_outliers.csv')
    sessions = sorted(a.session.unique(), key=lambda s: (int(s[3]), int(s.split('Tank')[1])))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
    for ax, pat in zip(axes, ('Leopard', 'Dark aggression')):
        sub = a[a.pattern == pat]
        for i, ses in enumerate(sessions):
            v = sub[sub.session == ses]
            ax.scatter([i] * len(v), v.blk, s=80, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       color=ORANGE if ses.startswith('Day1') else BLUE)
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels([s.replace('Day', 'D').replace('_Tank', 'T') for s in sessions],
                           rotation=60, fontsize=12)
        ax.set_title(pat, loc='left', fontsize=17, color=INK)
    axes[0].set_ylabel('animal black level (L* 2nd percentile)')
    fig.tight_layout()
    fig.savefig(out / 'fig_animal_black.png', dpi=130)
    plt.close(fig)


def fig_eta(out: Path) -> None:
    """eta^2 of session within pattern, original vs corrected (same frames)."""
    stats = ['black level', 'b*', 'saturation', 'a*', 'L*']
    orig = [0.600, 0.420, 0.285, 0.576, 0.236]
    corr = [0.101, 0.166, 0.194, 0.600, 0.181]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(stats))
    ax.bar(x - 0.2, orig, 0.38, color=GRAY, label='original frames')
    ax.bar(x + 0.2, corr, 0.38, color=BLUE, label='corrected frames')
    for xi, (o, c) in enumerate(zip(orig, corr)):
        ax.text(xi - 0.2, o + 0.012, f'{o:.2f}', ha='center', fontsize=13, color=INK2)
        ax.text(xi + 0.2, c + 0.012, f'{c:.2f}', ha='center', fontsize=13, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.set_ylabel('variance explained by session (η²)\nlower = less lighting structure')
    ax.set_ylim(0, 0.7)
    ax.legend(frameon=False, loc='upper right')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    fig.savefig(out / 'fig_eta.png', dpi=130)
    plt.close(fig)


def fig_delta(scratch: Path, out: Path) -> None:
    """Change in AMI(cluster, video | class) from correction, per subset, fixed and refit."""
    d = pd.read_csv(scratch / 'split.csv')
    subsets = [('all', 'all frames (Days 1-3)', SLATE), ('day1', 'Day 1 only', ORANGE),
               ('day23', 'Days 2-3 only', BLUE), ('nofog', 'without fogged sessions', AQUA)]
    order = ['vgg19_1_448_gram_k64_taper', 'vgg19_12345_448_gram_k28_taper',
             'iter-1.1_msps-vae-mask_d16_z_b', 'dinov3_vitb16_224_gram_k64_taper',
             'iter-1.1_resnet-18_d16', 'iter-1.1_msps-vae-mask_d16_z_u']
    for mode, fname in (('fixed', 'fig_delta_fixed.png'), ('refit', 'fig_delta_refit.png')):
        t = d[d['mode'] == mode].pivot_table(index=['embedder', 'subset'], columns='variant',
                                             values='ami_video')
        delta = (t['final'] - t['raw']).unstack()
        fig, ax = plt.subplots(figsize=(15, 6.6))
        h = 0.19
        for j, (key, label, color) in enumerate(subsets):
            vals = [delta.loc[e, key] for e in order]
            ypos = np.arange(len(order)) + (j - 1.5) * h * 1.05
            ax.barh(ypos, vals, h, color=color, label=label)
        ax.axvline(0, color=INK2, linewidth=1.2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([NAMES[e] for e in order])
        ax.invert_yaxis()
        ax.set_xlabel('change in AMI(cluster, video | class) after correction  (negative = less video identity)')
        ax.set_xlim(-0.16, 0.16)
        ax.grid(axis='y', visible=False)
        ax.legend(frameon=False, loc='lower right', fontsize=13)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)


def parse_scoreboards(scratch: Path) -> pd.DataFrame:
    """Parse the per-embedder eval text outputs into one tidy table."""
    variants = {'orig', 'raw', 'final', 'wrong', 'norm'}
    rows = []
    files = sorted((scratch / 'tier1').glob('eval_*.txt')) + sorted((scratch / 'tier2').glob('eval_*.txt'))
    for f in files:
        emb = None
        for line in f.read_text().splitlines():
            tok = line.split()
            if len(tok) < 7:
                continue
            try:
                vals = [float(v) for v in tok[-6:]]
            except ValueError:
                continue
            if tok[0] in variants:
                var = tok[0]
            elif len(tok) > 1 and tok[1] in variants:
                emb, var = tok[0], tok[1]
            else:
                continue
            rows.append({'embedder': emb, 'variant': var, 'ami_video': vals[0], 'eff_videos': vals[1],
                         'knn': vals[2], 'probe': vals[4]})
    return pd.DataFrame(rows)


def fig_variants(scratch: Path, out: Path) -> None:
    """AMI within class across variants for all scored embedders (all frames)."""
    t = parse_scoreboards(scratch).drop_duplicates(['embedder', 'variant'])
    t.to_csv(scratch / 'all_embedders_variants.csv', index=False)
    p = t.pivot(index='embedder', columns='variant', values='ami_video')
    p = p.loc[[e for e in NAMES if e in p.index]]
    fig, ax = plt.subplots(figsize=(15, 7))
    ys = np.arange(len(p))
    for variant, label, color, marker, size in (
        ('orig', 'original frames', GRAY, 'o', 110), ('raw', 'raw re-export (no correction)', SLATE, 's', 90),
        ('final', 'photometric correction', BLUE, 'o', 150), ('wrong', 'wrong-tank parameters', ORANGE, 'D', 90),
        ('norm', 'per-tank mean/std normalization', AQUA, '^', 100),
    ):
        ax.scatter(p[variant], ys, s=size, color=color, marker=marker, label=label,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4 if variant == 'final' else 3)
    ax.set_yticks(ys)
    ax.set_yticklabels([NAMES[e] for e in p.index])
    ax.invert_yaxis()
    ax.set_xlim(0.1, 0.95)
    ax.set_xlabel('AMI(cluster, video | class)   (lower = less video identity)')
    ax.grid(axis='x')
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'fig_variants_all.png', dpi=130)
    plt.close(fig)


def label(img: np.ndarray, text: str, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Draw a small caption with a dark backing in the top-left corner."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (10 + 13 * len(text), 34), (20, 20, 20), -1)
    cv2.putText(img, text, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return img


def sheet_floor(out: Path) -> None:
    """Contact sheets of per-session median floor and darkest-pixel images, Days 1-3."""
    sessions = [f'Day{d}_Tank{t}' for d in (1, 2, 3) for t in range(1, 7)]
    med, dark = {}, {}
    for s in sessions:
        files = sorted((SAMPLES / s).glob('*.png'))[::6]
        arr = np.stack([cv2.imread(str(f)) for f in files]).astype(np.float32)
        lum = arr.mean(3)
        keep = lum >= np.percentile(lum, 40, axis=0)
        med[s] = np.nanmedian(np.where(keep[..., None], arr, np.nan), 0).astype(np.uint8)
        dark[s] = np.percentile(arr, 1, axis=0).astype(np.uint8)
    for name, imgs in (('sheet_floor_median.png', med),):
        rows = []
        for d in (1, 2, 3):
            cells = [label(cv2.resize(imgs[f'Day{d}_Tank{t}'], (520, 293)), f'D{d} T{t}')
                     for t in range(1, 7)]
            rows.append(np.hstack(cells))
        cv2.imwrite(str(out / name), np.vstack(rows))


def fig_swatches(scratch: Path, out: Path) -> None:
    """Grid of the darkest-pixel level per session (Days 1-3) as gray swatches with values."""
    s = pd.read_csv(scratch / 'outl' / 'scene_outliers.csv').set_index('session')
    fig, axes = plt.subplots(3, 6, figsize=(15, 6.4))
    for d in (1, 2, 3):
        for t in range(1, 7):
            ax = axes[d - 1, t - 1]
            lin = float(s.loc[f'Day{d}_Tank{t}', 'dark_lum'])
            srgb = 12.92 * lin if lin <= 0.0031308 else 1.055 * lin ** (1 / 2.4) - 0.055
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=(srgb,) * 3))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(ORANGE if lin > 0.012 else GRID)
                sp.set_linewidth(4 if lin > 0.012 else 1)
            ax.text(0.5, 0.62, f'D{d} T{t}', ha='center', color='white' if srgb < 0.5 else INK,
                    fontsize=16)
            ax.text(0.5, 0.32, f'{lin:.3f}', ha='center', color='white' if srgb < 0.5 else INK,
                    fontsize=14)
    fig.tight_layout()
    fig.savefig(out / 'fig_black_swatches.png', dpi=110)
    plt.close(fig)


def sheet_effect(out: Path) -> None:
    """Original / corrected animal frames for a fogged and a normal session."""
    picks = [('Day1_Tank2_Cuttle2_Intruder_Crop', 'fogged: Day 1 Tank 2'),
             ('Day2_Tank4_Cuttle1_Resident_Crop', 'not fogged: Day 2 Tank 4')]
    rng = np.random.default_rng(3)
    blocks = []
    for video, title in picks:
        frames = sorted(p.name for p in (VARIANTS / 'final' / 'beast_frames' / video).glob('*.png'))
        chosen = [frames[i] for i in rng.choice(len(frames), 4, replace=False)]
        top = np.hstack([cv2.resize(cv2.imread(str(VARIANTS / 'orig' / 'beast_frames' / video / f)), (400, 200)) for f in chosen])
        bot = np.hstack([cv2.resize(cv2.imread(str(VARIANTS / 'final' / 'beast_frames' / video / f)), (400, 200)) for f in chosen])
        blocks.append(np.vstack([label(top, title + ' - original'), label(bot, 'corrected')]))
        blocks.append(np.full((16, blocks[-1].shape[1], 3), 252, np.uint8))
    cv2.imwrite(str(out / 'sheet_effect.png'), np.vstack(blocks))


def main(scratch: Path, out: Path) -> None:
    """Make all figures."""
    out.mkdir(parents=True, exist_ok=True)
    style()
    fig_scene(scratch, out)
    fig_animal_black(scratch, out)
    fig_eta(out)
    fig_swatches(scratch, out)
    fig_delta(scratch, out)
    fig_variants(scratch, out)
    sheet_floor(out)
    sheet_effect(out)


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
