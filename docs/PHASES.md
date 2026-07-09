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
- `cuttle_patterns/config.py` provides `load_config()`, reading and validating that file
  (raises `FileNotFoundError`/`ValueError` with actionable messages if the file is
  missing or missing required keys). Tested in `tests/test_config.py`.
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

Steps per frame:
1. Recover the foreground mask via near-black RGB thresholding on the decoded frame.
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
