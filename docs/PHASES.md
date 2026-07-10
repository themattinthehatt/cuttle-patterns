# Project Phases

## Goal

Cluster the visual patterns cuttlefish display during social interaction, using an
unsupervised approach, and build tools to explore the resulting clusters.

Collaborators provide one video per fish per session, with the other animal and the tank
background masked out (black). Two fish per session, 72 sessions, ~72,000 frames per
session (fps and resolution TBD once data arrives) — roughly 10.4M individual frames
across both fish streams.

Raw pattern (pixel) clustering would conflate two independent sources of variation: the
skin pattern itself, and the animal's orientation/pose in the frame. We remove the second
source with a dedicated preprocessing phase (egocentric alignment) rather than trying to
learn orientation invariance directly in the embedding model (e.g. via contrastive
augmentation) — see [DECISIONS.md](DECISIONS.md).

Phases below are meant to be worked roughly in order, but expect iteration — especially
between Phase 2 (alignment quality) and Phase 6 (does the clustering look sane).

---

## Phase 0: Infrastructure setup — done

**Goal:** every machine that runs this code (all Linux) can find data and write results
without hardcoded paths.

- Per-machine config file at `~/.cuttle-patterns/config.yaml`, e.g.:
  ```yaml
  data_dir: /media/mattw/poseinterface/cuttle/data
  results_dir: /media/mattw/poseinterface/cuttle/results
  ```
- `cuttle_patterns/config.py` provides `load_config()` / `save_config()`, reading and
  writing that file (`load_config` raises `FileNotFoundError`/`ValueError` with
  actionable messages if the file is missing or missing required keys). Tested in
  `tests/test_config.py`.
- `cuttle setup` (via `cmd_setup.py`) interactively prompts for `data_dir`/`results_dir`
  and writes/updates the config file, confirming before overwriting an existing one —
  the documented way to create this file rather than hand-editing yaml.
- `beast-backbones` and `pyyaml` added to `pyproject.toml` dependencies.
- No data or large artifacts committed to git; everything lives under `data_dir` /
  `results_dir`.
- Top-level `README.md` with setup instructions (conda env + `pip install -e ".[dev]"`)
  and links to these docs.
- CI: `.github/workflows/tests.yml` (CPU tests via `pytest -m "not gpu"`) and
  `.github/workflows/lint.yml` (ruff), reviewed and simplified — lint no longer installs
  the project's runtime dependencies (torch included), since ruff only needs to be
  installed itself to check the source tree.

**Open questions (deferred, not blocking):**
- Exact set of keys the config needs beyond `data_dir` / `results_dir` (e.g. a separate
  fast local scratch dir for training, if `data_dir` is a slower shared mount) — revisit
  once Phase 4 makes IO speed on `data_dir` a concrete concern.

---

## Phase 1: Data ingestion & inventory — done (for the first delivered session)

**Goal:** know exactly what we have once the collaborators' videos land in `data_dir`.

- Raw file naming convention (confirmed): `data_dir/session-{session_id}_cuttle-
  {fish_id}.mp4` with an accompanying `session-{session_id}_cuttle-{fish_id}.txt` listing
  blank-frame indices (one integer per line, no header). First delivered example: session
  1, fish 1 — 512x512, 24 fps, 44,400 frames, 18,068 flagged blank.
- `cuttle_patterns/ingest.py` builds the manifest: `find_raw_videos` /
  `read_video_info` (via OpenCV) / `read_blank_frame_indices` / `build_manifest`, one row
  per video with `session_id`, `fish_id`, `video_path`, `blank_frames_path`, `n_frames`,
  `n_blank_frames`, `fps`, `width`, `height`. Written to
  `results_dir/manifests/ingest.parquet`.
- Exposed as `cuttle ingest` via the CLI (see below) rather than a bare script.
- Sanity check implemented: warns (does not fail) if a blank-frame index falls outside
  the video's frame range, or if a video's `.txt` file is missing.

**CLI:** `cuttle_patterns/cli/` mirrors the structure of a previous project
(github.com/themattinthehatt/crittercam): `main.py` is the entry point, registered as the
`cuttle` console script; it auto-discovers `cmd_*.py` modules in the same directory and
dispatches to whichever one's `register()` added the matching subparser. Each
`cmd_<name>.py` owns its own argparse wiring and a thin `cmd_<name>(args)` handler that
calls into the real logic living in a top-level module (e.g. `cmd_ingest.py` calls
`cuttle_patterns.ingest.build_manifest`). Future phases' CLI-exposed steps (align,
extract-frames, train, embed, cluster, serve, ...) should follow the same pattern — see
[DECISIONS.md](DECISIONS.md).

**Open questions:**
- Only one session/fish pair has been delivered so far (72 total expected); revisit the
  "~44,000 frames per session" assumption and cross-session resolution/fps consistency
  once more sessions land.

---

## Phase 2: Egocentric alignment preprocessing

**Goal:** produce, for every input video, a derived video where the cuttlefish's body is
consistently positioned and oriented, so pattern — not pose — dominates pixel variation.

We only have the mp4 itself to work with — no separate mask file, and standard H.264 mp4
(confirmed via `ffprobe` on the delivered file) has no alpha channel, so there's no
transparency/mask data to recover from the container. The "masked" background is just
black RGB pixels baked into the frame.

The rectangle is fixed at a 2:1 aspect ratio (long axis along the body) to keep box shape
from adding its own noise on top of body-position/orientation noise.

### Phase 2a: rectangle inscription via classical CV — implemented, CLI-exposed

Single-frame geometry lives in `cuttle_patterns/preprocessing/inscribe.py`; per-video
orchestration in `cuttle_patterns/preprocessing/align.py` and
`cuttle_patterns/preprocessing/overlay.py`. Exposed via two CLI commands:

- `cuttle inscribe [--data-dir] [--results-dir] [--output-dir] [--video-path] [--thresh]
  [--aspect] [--canonical-height]` — for every raw video in `data_dir` (or one, via
  `--video-path`): computes the per-frame rectangle trajectory, linearly interpolates
  over frames with no detected body, warps each frame's rectangle into a fixed canonical
  size (`aspect * canonical_height` × `canonical_height`, default 200×100), and writes
  `{output_dir}/{video_name}.mp4` (the aligned crop video) plus `.csv` (per-frame corner
  geometry + an `is_interpolated` flag). `output_dir` defaults to
  `results_dir/rectangles`.
- `cuttle overlay [same flags]` — QC tool. For each video, if `{video_name}.csv` doesn't
  exist yet, runs the same inscribe logic to produce it; then draws each frame's
  (interpolated) rectangle on the *raw* frame — green if directly detected, orange if
  interpolated — and writes `{output_dir}/{video_name}_overlay.mp4`.

Tests: `tests/preprocessing/{test_inscribe,test_align,test_overlay}.py` and
`tests/cli/{test_cmd_inscribe,test_cmd_overlay}.py`. There's also a standalone
visualization script predating the CLI, `scratch/run_inscribe_v1.py`, which samples a
reproducible random set of frames (fixed seed) from a raw video and writes annotated
PNGs to `{results_dir}/inscribe/{version}/` — still useful for quick single-frame
iteration without running a full video through the CLI.

Per-frame single-image pipeline (`inscribe.py`):
1. Recover the body mask as the complement of the frame's largest connected background
   blob (see "mask recovery" below — this is not simple intensity thresholding).
2. Estimate the body's principal axis via PCA on the mask's foreground pixel coordinates
   (centroid + major-axis angle) — first-pass approach, see [DECISIONS.md](DECISIONS.md).
3. Rotate the mask upright (major axis horizontal) via an affine warp, keeping the
   inverse transform to map results back to the original frame.
4. Seed a candidate rectangle center from the distance-transform peak (center of the
   largest inscribed circle) in the rotated mask.
5. Grow a fixed 2:1 rectangle from that seed: binary-search the largest scale that stays
   fully inside the mask (checked in O(1) via an integral image / summed-area table), and
   coordinate-ascent-refine the center, alternating the two until convergence.
6. Map the rectangle's corners back to the original frame via the inverse rotation.

**Mask recovery (step 1), iterated against real data:**
- v1 thresholded at intensity 10. Worked, but produced small/off-center boxes whenever
  the animal displayed dark skin patterning — chromatophore blotches near black in value
  were misclassified as background, riddling the mask with holes the rectangle-growing
  step couldn't cross.
- v2 tightened the threshold to intensity 1 (`gray > 0`), after confirming the background
  is pure 0 (sampled corner pixels, no compression noise) while most "hole" pixels inside
  the body were nonzero (1-9). Recovered most holes for free, but left small islands of
  body pixels that render at literal 0.
- v3 (current) replaced intensity thresholding with a background-complement approach:
  label connected components of near-black (`<= thresh`, default 0) pixels, keep only the
  single largest as the true background, and treat everything else — including isolated
  zero-valued patches on the animal, whether or not they're topologically enclosed — as
  body. Parameter-free relative to hole-filling/morphological alternatives; assumes one
  dominant background region per frame (holds as long as the body doesn't split the frame
  into disconnected background pockets).

**Known limitations — status as of the last real-data check:**
- Head/tail direction is unresolved — PCA gives an axis, not a sign, so the rectangle is
  orientation-aligned but not yet canonically "head right." Being addressed by Phase 2b
  below.
- Arm posture bias is the **dominant** failure mode: PCA is free to find the largest
  rectangle anywhere across the full mantle+arm-crown blob, and it frequently lands
  straddling the arm crown rather than centered on the mantle. Confirmed visually across
  a 20-frame sample. Being addressed by Phase 2b below.
- Occlusion (one fish crossing in front of the other, masking the far fish's pixels to
  black) is likely improved as a side effect of the v3 mask recovery — an occluding blob
  fully inside the target fish's silhouette forms its own small background component and
  gets absorbed into the body, the same mechanism that fixes dark-patterning holes — but
  not yet verified against a real occlusion frame. Still unresolved if the occluding blob
  touches the image edge and merges with the true background component. Accepted as fine
  either way — the rectangle is allowed to contain occluded (black) pixels; downstream
  pattern analysis handling of those pixels is out of scope for this phase.

### Phase 2b: pose-informed refinement — implemented, CLI-exposed

**Status:** a first labeling attempt at the originally-planned 4-keypoint scheme (tail,
neck, two lateral mantle-width points) found the two lateral "width" points hard to
label consistently. The design below replaces that plan with a 2-keypoint scheme and
supersedes the old one (see [DECISIONS.md](DECISIONS.md)). Real per-frame tail/neck
predictions for session-01/cuttle-01 landed (from a pose-estimation model trained outside
this codebase) and the integration below is implemented: `tail`/`neck` are optional
keyword args on the existing `inscribe_rectangle` (not a parallel function) — when both
are given it uses `body_orientation_signed` instead of PCA and cuts the rotated mask at
the neck via `cut_mask_at_neck` before sizing; when omitted, behavior is unchanged from
Phase 2a. Same pattern one level up: `compute_corner_trajectory` takes optional
`tail_xy`/`neck_xy` per-frame arrays, and `align_video` takes an optional `pose_path`.

**Keypoints needed (2, not 4):**
- `tail`: tip of the mantle, opposite the head.
- `neck`: the head/body (mantle-neck) transition point. Anatomically fixed regardless of
  arm posture (unlike an arm/head-tip keypoint), which is why it's usable as a mask-cut
  boundary — see step 3 below. Obtain via a lightweight pose model (`beast-backbones`
  infra, or DeepLabCut/SLEAP as a fallback — see [DECISIONS.md](DECISIONS.md)).

**Algorithm (supersedes the original 4-point ellipse-fit plan):**
1. Compute a **signed** axis angle from `neck - tail` (instead of PCA's unsigned major
   axis), so head-right orientation falls out directly — no separate direction-
   disambiguation step needed.
2. Rotate the v3 background-complement mask upright using this signed angle (reuses
   `rotate_mask_upright` unchanged, just fed a pose-derived angle instead of a PCA one).
3. Project the `neck` point through the same affine transform to get its rotated
   x-coordinate, and zero out every mask pixel past it (the head/arm side) — a **new**
   masking step, not present in Phase 2a. This is what actually fixes arm bias:
   everything downstream only ever sees the mantle.
4. Feed the cut, rotated mask into the **existing, unchanged** Phase 2a sizing machinery
   — `seed_from_distance_transform` + `grow_rectangle` (binary search + coordinate
   ascent) — then map corners back via the inverse rotation, same as Phase 2a step 6.

So Phase 2b is not a parallel implementation — it only replaces the orientation source
(Phase 2a step 2) and inserts one new masking step before the existing sizing logic
(before Phase 2a step 4). Implemented as: `body_orientation_signed(tail, neck)` in
`inscribe.py`, a signed counterpart to `body_orientation(mask)`; `cut_mask_at_neck(mask,
neck_x)`; and `tail`/`neck` as optional kwargs on `inscribe_rectangle` itself, rather than
a separate entry point — the mask recovery, rotation, seeding/growing, and corner mapping
are identical either way, so branching inside one function avoided duplicating them.
`compute_corner_trajectory` (in `align.py`) similarly takes optional `tail_xy`/`neck_xy`
per-frame arrays and passes them straight through per frame.

Per-frame pose predictions are read from `cuttle_patterns/preprocessing/pose.py`:
`load_pose_predictions` parses the CSV a pose-estimation model actually produces — the
standard multi-header format (three header rows: scorer/bodyparts/coords), not a
simplified schema — and `interpolate_pose` linearly interpolates (flat extrapolation at
the edges, same convention as `interpolate_corners`) over any frame where either
keypoint's likelihood is below 0.9. `interpolate_pose` trusts that the pose CSV has
exactly one row per video frame (confirmed on the one real file seen so far) rather than
taking an explicit frame count to reconcile against — `align_video` no longer needs to
open the video up front just to learn its frame count before calling it.
`align_video` takes an optional `pose_path` and ORs this interpolation mask into the one
already written for undetected/cut-empty frames. `cuttle inscribe`/`cuttle overlay` add
`--pose-dir` (default `results_dir/pose`, looked up per video as `{video_name}.csv`) and
`--pose-path` (single-video override); a video with no matching pose file falls back to
the Phase 2a PCA path with a printed message, so a batch run works over a mix of videos
with and without pose predictions.

**Why this is better than the original 4-point ellipse-fit plan, not just simpler:**
using the real (cut) mask for sizing instead of fitting a synthetic ellipse also
eliminates a known limitation of the old plan — a symmetric ellipse couldn't capture the
mantle's true taper (narrower at the tail, wider near the neck); the mask-based approach
gets the real shape for free.

**Known trade-off:** loses keypoint-derived occlusion robustness specifically for the
*width* dimension — if the mask itself is corrupted right at the mantle in the one
already-unresolved edge case (an occluding blob touching the image border), width sizing
would still be affected, where a pure keypoint-derived ellipse would have been immune.
Accepted as a narrow, already-known risk (see the occlusion bullet under Phase 2a), not a
new one.

### Temporal smoothing — implemented

**Status:** visual QC via `cuttle overlay` on session-01/cuttle-01 (the 1:35-1:40 window)
showed the rectangle jittering frame-to-frame — center jumps averaging ~13px (max ~53px)
— even where the body's actual pose barely changes, tracking the fins' beat rather than
real motion. `align.py` applies one of two smoothers to the final corner trajectory
(after `interpolate_corners` and after ORing in the pose interpolation mask), mutually
exclusive via `align_video`'s `smoothing_window`/`smoothing_sigma` args (raises
`ValueError` if both given) and `cuttle inscribe`/`cuttle overlay`'s `--smoothing-window`/
`--smoothing-sigma` (an argparse mutually-exclusive group):

- `smooth_corners` — centered rolling median, default window 9 frames (1 disables it).
  The historical default when neither flag is given. Robust to a one- or two-frame
  outlier (rejects it outright) at the cost of a "steppy" trajectory that doesn't fully
  flatten continuous jitter.
- `smooth_corners_gaussian` — Gaussian filter, `--smoothing-sigma` (standard deviation in
  frames; defaults to 2.0 if the flag is given with no value). Blends the whole window
  rather than snapping to one observed value, which tracks continuous, quasi-periodic
  jitter (fin beats) more smoothly than the median at comparable strength, at the cost of
  blending in rather than rejecting a single genuinely-bad frame.

Measured on the 1:35-1:40 window (mean/max frame-to-frame center jump): raw ~13px/~53px;
median (w=9) ~2.6px/~9px; gaussian (σ=1.5, comparable strength to the median) ~2.7px/~8px
but with lower variance; gaussian (σ=2.0, the CLI default) ~1.25px/~4px. No visible lag
against the body at any of these settings on this clip.

**Open question:** both the window (9 frames) and sigma (2.0) were tuned by eye/on this
one noisy clip; revisit either if they lag behind genuinely fast movements (e.g. escape
jets) elsewhere in the dataset.

### Remaining Phase 2 work

- ~~Final rotate/crop/warp + derived-video writing~~ — done, see Phase 2a above
  (`align_video` / `cuttle inscribe`).
- Canonical crop size (`--canonical-height`, default 100; width = `round(aspect *
  height)`) is still an arbitrary placeholder — revisit once the distribution of
  inscribed-rectangle sizes across the full (eventually 72-session) dataset is known.
- ~~Phase 2b's pose-data plumbing~~ — done, see Phase 2b above (`--pose-dir`/`--pose-path`
  on `cuttle inscribe`/`cuttle overlay`). Only session-01/cuttle-01 has pose predictions
  so far; revisit the 0.9 likelihood threshold once more sessions' predictions land and
  their confidence distribution is known.

---

## Phase 3: Representative frame extraction

**Goal:** turn aligned videos into a training set of still frames for BEAST.

- Define a sampling strategy per video: start with simple uniform subsampling at a fixed
  interval as a baseline; consider motion- or change-based keyframe selection later if
  uniform sampling over/under-represents static periods.
- Output: a frame manifest (session, fish, source frame index, path/tensor) under
  `results_dir`.

**Open questions:**
- Sampling rate/interval — depends on fps (unknown) and how quickly patterns change.
- Whether to weight sampling toward moments of pattern change (harder, deferred).

---

## Phase 4: BEAST training

**Goal:** train a BEAST backbone (ViT with masked autoencoding + temporal contrastive
loss) on our own unlabeled aligned frames, per the paper's experiment-specific pretraining
design — not reusing a checkpoint trained on a different dataset.

- Install/import `beast-backbones`; use its training entry point (library call or CLI)
  rather than reimplementing the ViT/MAE/contrastive loop.
- Prepare data in whatever input format `beast-backbones` expects (frames + temporal
  neighbor structure for the contrastive sampling) — to be confirmed once we're working
  against the package.
- Train on the cloud multi-GPU machine (code should assume a local-style multi-GPU
  workstation — no SLURM/job-array abstractions needed, see
  [DECISIONS.md](DECISIONS.md)).
- Checkpointing and basic experiment tracking (what config/data produced which
  checkpoint).

**Open questions:**
- `beast-backbones` package's exact data-loading contract and CLI surface (need to read
  its docs/source once the data pipeline is ready to feed it).

---

## Phase 5: Embedding extraction

**Goal:** get a 768-d BEAST embedding for every frame in the training set (and,
eventually, any frame we care to embed).

- Run the trained backbone over frames to produce embeddings.
- Store embeddings alongside frame metadata (session, fish, frame index, video path) in
  a structured format (Parquet/HDF5) under `results_dir`, keyed so they can be joined
  back to the manifest from Phase 3.
- Structure this so alternative embeddings (non-BEAST) can be added side by side later.

---

## Phase 6: Dimensionality reduction & clustering

**Goal:** turn 768-d embeddings into something explorable and clustered.

- Apply UMAP and/or t-SNE to project embeddings to 2D.
- Run k-means in the 2D projection as the initial clustering approach.
- Store 2D coordinates + cluster labels per frame, joined to the same frame keys as
  Phase 5.

**Open questions:**
- Cluster in the 2D projection (simpler, what's currently planned) vs. in the original
  768-d space (arguably more principled, but less directly tied to what's visualized) —
  revisit if 2D clusters look unstable relative to structure visible in the high-d space.

---

## Phase 7: Interactive visualization

**Goal:** a UI to explore the embedding — a dot per frame, hover shows the corresponding
aligned frame image, colored by cluster.

- Built on Plotly (specific app framework — Dash vs. something lighter — still to be
  decided when we get here, see [DECISIONS.md](DECISIONS.md)).
- Must support swapping in different embeddings/projections (BEAST now, others later)
  without rebuilding the UI.
- Likely filters: by session, by fish, by cluster.

---

## Phase 8: Iteration & analysis

**Goal:** use the tool from Phase 7 to sanity-check everything upstream, and actually
characterize the patterns found.

- Visually audit alignment quality (Phase 2) and frame sampling (Phase 3) using the
  clusters/outliers surfaced by the UI.
- Characterize discovered clusters (are they biologically meaningful pattern states?).
- Loop back to earlier phases as needed (e.g. swap in pose-based alignment, adjust
  frame sampling, try alternative embeddings).
