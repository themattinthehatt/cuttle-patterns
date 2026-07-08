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

## Phase 0: Infrastructure setup

**Goal:** every machine that runs this code (all Linux) can find data and write results
without hardcoded paths.

- Per-machine config file at `~/.cuttle-patterns/config.yaml`, e.g.:
  ```yaml
  data_dir: /media/mattw/poseinterface/cuttle
  results_dir: /home/mattw/cuttle-patterns-results
  ```
- A config-loading module in `cuttle_patterns` that reads this file (clear error if
  missing, pointing to where to create it).
- `beast-backbones` added to `pyproject.toml` dependencies.
- No data or large artifacts committed to git; everything lives under `data_dir` /
  `results_dir`.

**Open questions:**
- Exact set of keys the config needs beyond `data_dir` / `results_dir` (e.g. a separate
  fast local scratch dir for training, if `data_dir` is a slower shared mount).

---

## Phase 1: Data ingestion & inventory

**Goal:** know exactly what we have once the collaborators' videos land in `data_dir`.

- Define the expected raw directory/file naming convention (per session, per fish) —
  TBD until we see the real delivered structure.
- Build a manifest (session id, fish id, video path, frame count, fps, resolution) as a
  CSV/Parquet file under `results_dir`.
- Sanity checks: frame-count consistency with collaborators' reported ~72,000, resolution
  consistency across sessions, corrupt/truncated video detection.

**Open questions:**
- Is the "black background" purely the composited mask, or is an alpha/mask channel or
  separate mask file also provided? This affects how we recover the foreground mask in
  Phase 2 (currently assuming: threshold near-black pixels).
- Actual fps — needed to reason about motion between frames for Phase 3 sampling.

---

## Phase 2: Egocentric alignment preprocessing

**Goal:** produce, for every input video, a derived video where the cuttlefish's body is
consistently positioned and oriented, so pattern — not pose — dominates pixel variation.

Steps per frame:
1. Recover the foreground mask (near-black background thresholding, or provided mask).
2. Estimate the body's principal axis via PCA/ellipse fit on the mask (first-pass
   approach — see [DECISIONS.md](DECISIONS.md)).
3. Disambiguate head vs. tail along that axis (PCA gives an axis, not a direction).
4. Inscribe a rectangle in the body aligned to the principal axis; rotate/crop/warp to a
   fixed canonical size and orientation.
5. Write the aligned crop to a derived video, mirroring the input session/fish structure
   under `data_dir` (or a `derived/aligned/` subtree).

**Open questions / risks:**
- Head/tail disambiguation: PCA alone is 180°-ambiguous. Likely needs a rule (e.g.
  mantle vs. arm-crown mass asymmetry) plus temporal smoothing to avoid frame-to-frame
  flips. Flagged in [DECISIONS.md](DECISIONS.md) as the first thing to sanity-check
  visually once real data arrives.
- Ellipse fit may be noisy when arms are splayed vs. contracted (body shape isn't a
  fixed ellipse). If this proves too noisy, fall back to pose keypoints (DeepLabCut/
  SLEAP) for a more robust axis — deferred until we see how bad it is.
- Canonical crop size/aspect ratio: needs to accommodate the most extended arm posture
  across the dataset without excessive padding for contracted postures.

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
