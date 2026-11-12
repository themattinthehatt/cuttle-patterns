# Frame selection for BEAST training (scratch plan)

**Status: superseded.** The result of this work is now the `cuttle extract-*` pipeline
(guide: `cuttle_patterns/preprocessing/extract/README.md`; `cuttle extract` lists the stages; mechanism in `docs/PHASES.md` Phase 3, rationale in the
"Class-quota frame selection" entry of `docs/DECISIONS.md`). The scripts in this directory
are the exploratory record: threshold tuning (`jump_contact_sheets.py`,
`motion_contact_sheets.py`, `black_contact_sheets.py`), the selection-method prototypes
(`embed_subset.py`, `compare_selection.py`, `cross_video_selection.py`), and the versions of
eligibility/selection/evaluation that were promoted (`count_eligible.py`,
`final_eligibility.py`, `select_by_class.py`, `evaluate_selection.py`, `kmeans_utils.py`).
Intermediate data is under `results/.extraction/`.

Prototype of a revised `cuttle extract`: pick a diverse, less state-biased pool of frames
to use as `beast_frames`, on which the different extractors/backbones are then compared.
Orthogonal to the photometry work in `scratch/photometry/`. Once the settings are chosen,
the logic moved into `cuttle_patterns/preprocessing/extract/` and the CLI (see
"Promotion to the CLI" below).

Design reference: `/media/mattw/poseinterface/mighty-mouse/docs/pretrain_frame_selection.md`
(same pipeline shape: per-video DINOv3 embeddings, then hierarchical k-means with balanced
sampling).

## Problem

The current `cuttle extract` keeps the top half of frames by pixel motion energy, then runs
PCA + flat k-means on raw pixels and takes the frame nearest each centroid. Dense states
(notably black aggression) take most of the cluster budget, which biases the pool and every
downstream k-means. Goals:

- fewer frames from dense regions of embedding space, while rare patterns are retained;
- selection in a semantic embedding space instead of raw-pixel PCA;
- no context frames anywhere in this project (no temporal contrastive loss is used).

## Pipeline

### Step 1: eligibility

A frame is eligible only if **all** of the following hold:

1. not blank (`_black_frames.txt`);
2. every tail/neck keypoint has likelihood >= 0.9;
3. the `cuttle inscribe` rectangle's long edge is >= 50% of the neck-tail distance
   (existing small-rectangle filter);
4. no keypoint jump: for each keypoint, the displacement between t-1 and t and between t and
   t+1 is <= 25 px (in the 512x512 crop frame; threshold chosen from the jump contact
   sheets). Only evaluated where both endpoints are high-confidence; missing neighbors
   (first/last frame) skip that side;
5. motion energy (BEAST, on the aligned rectangle video, 32x32) at or below the video's 99th
   percentile among frames passing 1-4 (drops alignment glitches/blur; frame 0 excluded);
6. < 5% pure-black pixels (max channel == 0) in the aligned rectangle frame (chosen
   aggressively on purpose; see `black_contact_sheets.py`).

Removed relative to the current logic:

- the "both neighbors must survive" rule (`build_candidate_frame_idxs`);
- the +-1 context-frame export;
- the "keep the top 50% by motion energy" cut (replaced by criterion 5's top-1% drop).

Notes:

- Jump check uses the **raw** predictions, before `pose.interpolate_pose` fills anything in.
- Every filter writes its own boolean column into a per-video table (`frame_idx`, one column
  per criterion, plus `eligible`), so rejected frames can be inspected.

### Step 2: embed eligible frames

- Frozen DINOv3 ViT-S/16, CLS token, 224 px, float16, on the aligned 512x512 frames.
- Stored per video, row-aligned to a frame-index array, so new videos can be added without
  redoing others.
- Only eligible frames are embedded.

### Step 3: diversity selection (target 1k frames/video, ~144k total)

- **Per video:** L2-normalize; hierarchical k-means with balanced sampling (fine clusters
  grouped on their centroids; equal number of fine clusters drawn per group; one **random**
  member per chosen cluster). Random member, not nearest-to-centroid and not farthest: Vo et
  al. 2024 and the mighty-mouse experiments find random == nearest and farthest is much
  worse. Videos with <= 1k eligible frames keep everything.
- **Across videos:** video identity dominated earlier embeddings (see
  `docs/latent_space_confounds.md`), so per-video selection comes first. If a second,
  cross-video pass is needed (to equalize per-session/fish budgets), mean-center
  embeddings per session before clustering.
- Output: manifest with `session_id`, `fish_id`, `frame_idx`, `image_path` plus
  `img{frame_idx}.png` per selected frame (anchor only, no context).

### Step 4: diagnostics

- Classifier class histogram of the new pool vs. all eligible frames vs. the old
  `beast_frames` pool (is black aggression no longer over-represented?).
- Coverage on a random sample of eligible frames (including those not embedded for
  selection): nearest-selected cosine distance (mean, p95, p99.9), share of the densest
  state, share of the rarest states, fraction of the 1% most isolated frames retained.
- Contact sheets of the most isolated picks (embeddings cannot tell junk from a rare
  pattern), and of the frames removed by the jump filter.

### Step 5: use the pool

Export as `beast_frames`, train/compare the extractors on it.

## Motion-energy gate history (crop-video sheets suggested dropping it; rectangle sheets reversed that: criterion 5)

Sheets made on the 512x512 crop videos (motion energy there includes whole-body motion)
showed mostly legitimate fast behavior, so the gate was dropped. Redone on the aligned
rectangle videos (`motion_contact_sheets.py`, all 144 videos in
`results/.extraction/motion_sheets_rect/`), the top ~1% is dominated by alignment glitches
(black wedges, warped fragments) and blur, so criterion 5 drops the top 1% per video.

Black-fraction sheets (`black_contact_sheets.py`, `results/.extraction/black_sheets_rect/`)
were used to choose criterion 6 (5%).

Final eligible counts with criteria 1-6: 5.20M of 6.34M frames; per video median 38.5k,
min 11.3k (Day8_Tank3_Cuttle2_Intruder), none below 1k (`results/.extraction/
eligibility_final_summary.csv`). The black filter removes a median of 386 frames per video
but up to 8.4k (tank-6 videos lose the most).

## Prototype results on 512x512 CROPS (superseded, wrong input; kept for reference)

Scripts: `embed_subset.py` (embeds eligible frames with DINOv3-S CLS + runs the skin-pattern
classifier; outputs in `results/.extraction/proto/`), `kmeans_utils.py`, `compare_selection.py`
(`results/.extraction/proto_results.json`). Mean over videos (stochastic methods: 3 seeds):

| method | dark-aggression share | cov p99.9 | densest-state ratio | rare-state ratio | isolated kept |
|---|---|---|---|---|---|
| eligible (all frames) | 0.621 | - | 1 | 1 | - |
| baseline (old recipe) | 0.676 | 0.243 | 0.37 | 1.28 | 0.094 |
| random | 0.619 | 0.243 | 1.03 | 1.00 | 0.039 |
| flat k-means | 0.624 | 0.212 | 0.90 | 1.09 | 0.059 |
| two-level | 0.622 | 0.223 | 0.86 | 1.35 | 0.079 |

- Within-video selection in DINOv3 CLS space does not change the classifier class balance;
  dark aggression is ~62% of eligible frames and of every embedding-based pool. The old
  recipe is slightly worse (0.68).
- Class imbalance is mostly between videos (dark share per video ranges 0.03-0.95), which a
  per-video selection cannot touch.
- Nearest neighbor of a frame (excluding +-100 frames) is in the same video 96% of the time
  (95% after per-video mean-centering; chance 12.5%).

Cross-video two-stage variant (`cross_video_selection.py`; per-video flat k-means with 3k
clusters, then k-means on all pooled centroids into g groups, equal centroids per group, one
random member each; 8 videos, 8k total, pooled metrics, 3 seeds):

| method | dark share | class entropy | cov p99.9 | densest ratio | rare ratio |
|---|---|---|---|---|---|
| eligible pooled | 0.587 | - | - | 1 | 1 |
| random pooled | 0.590 | 1.131 | 0.240 | 1.02 | 0.99 |
| per-video two-level | 0.622 | 1.094 | 0.216 | 0.72 | 1.03 |
| cross-video g=100 | 0.587 | 1.131 | 0.221 | 0.64 | 1.20 |
| cross-video g=400 | 0.617 | 1.085 | 0.214 | 0.71 | 0.99 |
| cross-video g=800 | 0.625 | 1.072 | 0.213 | 0.74 | 0.89 |

Class mix still tracks the eligible mix; per-video allocation ends up near-uniform
(~0.9-1.2k each). DINOv3 CLS geometry does not separate the classifier's pattern classes
well enough for balanced sampling there to rebalance them.

## Prototype results on aligned RECTANGLE videos (current; 8 videos, DINOv3-S CLS)

Same scripts, now with DINOv3 on `results/rectangles/*.mp4` (200x100, stretched to 224) and
motion energy on rectangles; classifier still on the 512x512 crops (outputs in
`results/.extraction/proto_rect/`, `proto_rect_results.json`, `xvid_rect_results.json`).
Per-video 1k picks (mean over videos, 3 seeds): dark share eligible 0.621 / random 0.619 /
flat 0.623 / two-level 0.624 / old recipe 0.689; cov p99.9 random 0.321, flat 0.288,
two-level 0.296, old 0.389; rare-state ratio two-level 1.48, flat 1.13, old 1.32.
Cross-video (8k pooled): dark share 0.598 (g=100), 0.625 (g=400), 0.626 (g=800) vs
eligible 0.587; densest ratio 0.55 / 0.80 / 0.87, rare ratio 1.85 / 1.54 / 1.46.
Class mix still not rebalanced; coverage and rare-state share improve vs random.
Nearest-neighbor same-video rate 96% (raw) / 96% (video-centered).

Sheets for all 144 videos (6 rows incl. 90-95 pct) are in `results/.extraction/motion_sheets_rect/`.
Motion-energy contact sheets on rectangles (`motion_contact_sheets.py`) are very different
from the crop ones: the top ~1% contains clear junk (black wedges, warped/misplaced
rectangle fragments, heavy blur), so the motion-energy gate is worth reconsidering.

## Decisions / resolved items

- Pose CSVs (`/media/mattw/CUTTLE/pose-estimation/cuttle-test/models/iter-1.2_eks/video_preds/`,
  one per crop video, EKS-smoothed `x`,`y`,`likelihood` for `neck` and `tail` only) are in the
  512x512 crop frame, so the jump threshold applies directly (fixed at 25 px).
- Videos with < 1k eligible frames: print a warning and take all of them.
- Photometry: embeddings (and anything pixel-derived) wait until the photometry work is
  done, since it may change the pixels. Eligibility tables depend only on pose, blank-frame
  lists and rectangles, so they can be computed now; only embeddings depend on pixels.
- Thresholds are tuned with contact sheets before any eligibility code is written (see
  "Threshold tuning" below).

## Still open

- (resolved) 1k/video budget is feasible everywhere under criteria 1-6.
- Embedding input: full aligned frame vs. masked/cropped body.
- Hierarchical k-means sizes (fine clusters, groups) per video; set from eligible-frame
  counts, mirroring the mighty-mouse thresholds.

## Threshold tuning

1. Keypoint-jump contact sheets (`jump_contact_sheets.py`): 5 random videos, one 5x10 sheet
   each; rows are jump-size bins (>50, 40-50, 30-40, 20-30, 10-20 px). Result: threshold
   fixed at 25 px.
2. Motion-energy contact sheets: gate dropped (above).

## Class-quota selection (current plan)

Eligibility criteria 1-6, day 1 excluded (132 videos, 4.72M frames). Classifier: 2026-11-07
two-head model, pattern head only, on 512x512 crops (`results/classifications/
2026-11-07_classifier_model/`); DINOv3-S CLS on rectangles, every frame
(`results/embeddings/dinov3_vits16/`). Square-root class quotas (no confidence floor, no
per-video cap), then per class: stage 1 = per-video spherical k-means (~3x the video's
share of the class quota), stage 2 = k-means on pooled centroids into groups, equal draws
per group, one random member per chosen centroid. Scripts: `final_eligibility.py`
(`results/.extraction/eligibility_final/`), `select_by_class.py`, `evaluate_selection.py`
(per-video allocation, within-class coverage and state ratios vs a random baseline, confidence,
temporal redundancy, contact sheets; writes `report.md`/`report.json`/`eval_sheets/` next to
the manifest).

## Planned files (all under `scratch/extraction/`)

- `eligibility.py`: per-video eligibility table (Step 1).
- `embed_eligible.py`: DINOv3 embeddings of eligible frames (Step 2).
- `select_frames.py`: per-video hierarchical k-means selection (Step 3).
- `diagnostics.py`: class histograms, coverage metrics, contact sheets (Step 4).

## Promotion to the CLI

When the settings are final, follow the "Adding a new pipeline stage" checklist in
`CLAUDE.md`: update `extract.py` (and remove the context-frame/neighbor logic),
`cli/cmd_extract.py`, `paths.py`, `README.md`, `docs/PHASES.md` (Phase 3 text), a
`docs/DECISIONS.md` entry (why the selection changed), and the mirrored tests. Embedding
reuse with `cuttle embed` should be considered at that point.
