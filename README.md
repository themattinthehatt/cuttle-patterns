# cuttle-patterns

Unsupervised analysis of visual patterns displayed by cuttlefish during social
interaction: egocentric alignment of segmented cuttlefish videos, self-supervised
embedding via BEAST, and interactive tools for exploring the resulting pattern clusters.

## Docs

```
README.md                                    # setup + full `cuttle` CLI reference
AGENTS.md                                    # development guidelines for coding agents (CLAUDE.md imports it)
docs/
├── PHASES.md                                # living design/status narrative, by phase
├── DECISIONS.md                             # append-only decision log (the why)
├── data_layout.md                           # raw data and results layout, known data issues
├── pose_estimation.md                       # external tail/neck pose model status
├── latent_space_confounds.md                # AE -> MSPS-VAE -> masked-MSPS-VAE confound timeline
└── implementation_notes/
    ├── embedder.md                          # `cuttle embed` backbones/readouts (canonical)
    ├── eval_plan.md                         # historical eval-harness design
    └── msps_vae.md                          # MSPS-VAE model/sampler/config reference
cuttle_patterns/eval/README.md               # eval harness how-to-run (current scope, metrics)
cuttle_patterns/preprocessing/extract/README.md  # frame selection (`cuttle extract-*`) guide
cuttle_patterns/paths.py                     # results_dir tree (module docstring)
```

## Setup

Requires the `ffmpeg` binary on `PATH` (e.g. `apt-get install ffmpeg` /
`brew install ffmpeg`) — `cuttle overlay` shells out to it for H.264-encoded output.

```bash
conda create -n cuttle python=3.12
conda activate cuttle
pip install -e ".[dev]"
```

`beast-backbones v2.0.0` (required) isn't on PyPI yet, so install it from a git clone first — see
the [BEAST README](https://github.com/paninski-lab/beast#installation) for the exact steps
(`pip install lightning poetry-core` then `pip install -e . --no-build-isolation`).

Each machine needs a local config file at `~/.cuttle-patterns/config.yaml` pointing to
where data and results live. Create/update it with:

```bash
cuttle setup
```

which prompts for `data_dir` and `results_dir`, e.g.:

```yaml
data_dir: /path/to/cuttle/data
results_dir: /path/to/cuttle/results
```

## Pipeline: preprocessing

The steps below run in order after `cuttle setup`. Every subcommand reads
`data_dir`/`results_dir` from the config file by default; override either with
`--data-dir`/`--results-dir` if needed. Output paths mentioned below are relative to
`results_dir` — see `cuttle_patterns/paths.py` for the exact on-disk layout of every
pipeline stage's output.

### `cuttle ingest`

Scans `data_dir` for raw `Day{day}_Tank{tank}_Cuttle{n}_{role}_crop.mp4`/
`_black_frames.txt` pairs and writes a manifest of what's there (frame counts, fps,
resolution, flagged-blank counts) under `manifests/`.

```bash
cuttle ingest
```

### `cuttle inscribe`

Inscribes an egocentric rectangle in the cuttlefish's body on every frame, then warps it
into a fixed-size, upright crop video. For each raw video, writes the aligned crop and a
`.csv` of per-frame rectangle corners (plus an `is_interpolated` flag for frames where
nothing was detected directly and the rectangle had to be filled in) under `rectangles/`.

```bash
cuttle inscribe
```

It runs in one of two modes, chosen automatically per video:

- **PCA-based (default).** Estimates body orientation via PCA on the segmentation mask.
  Needs no extra inputs, but is biased by extended arms — the rectangle can end up
  straddling the arm crown instead of staying centered on the mantle.
- **Pose-informed.** Used automatically for any video with a matching per-frame
  tail/neck pose-prediction CSV — cuts the mask at the neck before sizing the rectangle,
  so it only ever sees the mantle. By default, `cuttle inscribe` looks for predictions at
  `results_dir/pose/{video_name}.csv`; point it elsewhere with `--pose-dir`. Videos with
  no matching pose file fall back to the PCA-based mode with a printed message, so it's
  safe to run over a mix of videos with and without predictions. See
  [docs/pose_estimation.md](docs/pose_estimation.md) for how these pose predictions are
  produced.

Either mode can still leave the rectangle jittering frame-to-frame during rapid body
motion (fin beats in particular); the corner trajectory is smoothed to damp this, via
either a Gaussian filter (`--smoothing-sigma`) or a centered rolling median
(`--smoothing-window`) — mutually exclusive, see `--help` for defaults. The Gaussian
tracks continuous, quasi-periodic jitter (e.g. fin beats) more smoothly since it blends
the whole window rather than snapping to one observed value; the median is more robust
to an occasional single-frame garbage detection, since it rejects rather than blends it
in.

To process one video at a time (e.g. while iterating on `--thresh`/`--aspect`), use
`--video-path`, optionally paired with an explicit `--pose-path`:

```bash
cuttle inscribe --video-path /path/to/Day1_Tank2_Cuttle1_Resident_Crop.mp4 \
  --pose-path /path/to/Day1_Tank2_Cuttle1_Resident_Crop_pose.csv
```

Pass `--skip-existing` to leave a video alone (no re-inscription) if its
`{video_name}.mp4`/`.csv` already exist in `output_dir`, so a batch run can be safely
re-run over a directory that's only partially processed.

### `cuttle overlay` (optional QC)

Draws each frame's (interpolated) rectangle on top of the corresponding *raw* frame —
green if directly detected, orange if interpolated — so inscription quality can be
checked visually before moving on. Reuses `cuttle inscribe`'s `.csv` if it already
exists for a video; otherwise runs the same detection first (accepting the same
`--pose-dir`/`--pose-path`/`--thresh`/`--aspect` flags as `cuttle inscribe`, used only if
it has to compute the CSV itself).

If a matching pose-prediction CSV is found (`--pose-dir`/`--pose-path`, same lookup as
`cuttle inscribe`; see [docs/pose_estimation.md](docs/pose_estimation.md)), each frame's
tail/neck keypoints are also drawn on top of the rectangle — bright pink with a white
border — but only where the raw prediction's likelihood is >= 0.9; lower-confidence
keypoints are left undrawn rather than interpolated.

```bash
cuttle overlay
```

Writes an `_overlay.mp4` alongside the aligned crop under `rectangles/`, H.264-encoded
(via `ffmpeg`) since these are full raw-resolution videos and can otherwise get large;
tune size vs. quality with `--crf` (lower is higher quality/larger file). As
with `cuttle inscribe`, pass `--skip-existing` to leave an already-written overlay video
alone rather than re-encoding it.

### `cuttle extract` (frame selection, in stages)

Selects the still frames BEAST trains on from the aligned crop videos (under `rectangles/`),
as a sequence of separately runnable, resumable stages. `cuttle extract` itself only prints
the stage list; run them in order (`cuttle <stage> --help` for options and defaults,
`cuttle_patterns/paths.py` for exact output paths; the working guide, with gotchas and what to
check, is `cuttle_patterns/preprocessing/extract/README.md`):

```bash
cuttle extract-eligibility --pose-dir /path/to/pose/predictions
cuttle extract-classify --model-dir /path/to/classifier_dir
cuttle extract-dino-cls
cuttle extract-kmeans-video --name sqrt-132k --classifier-name <classifier_dir name> --budget 132000
cuttle extract-kmeans-pooled --name sqrt-132k
cuttle extract-report --name sqrt-132k
cuttle extract-export --name sqrt-132k
```

Each stage prints the next one when it finishes. The long stages (`extract-eligibility`,
`extract-classify`, `extract-dino-cls`) skip videos that are already done, write a
`progress_shard{i}of{n}.json` next to their output, and take `--shard`/`--n-shards` to split
videos across processes. Algorithm-level detail is in [docs/PHASES.md](docs/PHASES.md)
(Phase 3); why it works this way is in [docs/DECISIONS.md](docs/DECISIONS.md).

#### `cuttle extract-eligibility`

Per aligned video, computes which frames are usable: not blank, all tail/neck keypoints
above 0.9 likelihood, rectangle not too small for the body, no keypoint jump over 25 px
between neighboring frames, motion energy not in the video's top 1% (alignment glitches,
blur), and less than 5% pure-black pixels. One table per video, one boolean column per
criterion, so thresholds can be re-tuned (`--jump-max`, `--me-percentile`, `--black-max`)
without guessing which frames each one removed. `--pose-dir` is required (crop-frame pose
CSVs); a video missing its pose or rectangle CSV is skipped with a warning.

#### `cuttle extract-classify`

Runs the two-head skin-pattern classifier (`--model-dir`: weights plus `behaviors.json`) over
every frame of the 512x512 crop videos, keeping only the pattern head. One table per video
with the label, confidence and per-class probabilities of every frame; `--save-features` also
keeps the 512-d penultimate features. Output goes to `classifications/{model name}/`.

#### `cuttle extract-dino-cls`

Computes DINOv3 CLS features (fp16 autocast, stored as float16) for every frame of the aligned
rectangle videos, one `.npy` per video where row i is frame i. It uses the same backbone and
preprocessing as `cuttle embed`, applied to whole videos instead of exported stills.
`--arch` and `--resolution` pick the model; output goes to `embeddings/dinov3_{arch}/`.

#### `cuttle extract-kmeans-video`

Stage 1 of the selection. Splits `--budget` frames across the classifier's classes with
quotas proportional to `count ** --quota-power` (0.5, square root, by default), then runs
k-means per video and per class on the DINOv3 features. Day 1 is left out by default
(`--exclude-days`). Because stage 1 depends on the quotas, changing the budget or settings
needs `--overwrite` (or a new `--name`).

#### `cuttle extract-kmeans-pooled`

Stage 2. Per class, pools every video's stage-1 centroids, groups them, draws a balanced
number of centroids per group (`--picks-per-group`) and takes one random frame from each.
Writes the selection manifest and a summary under `extraction/selections/{name}/`.

#### `cuttle extract-report`

Diagnostics for a selection: per-video allocation, per-class coverage and state ratios against
a random baseline of the same size, classifier confidence, temporal redundancy, and contact
sheets of random and most-isolated picks (`eval_sheets/`). Run it before exporting.

#### `cuttle extract-export`

Writes the selected frames as `beast_frames/{video}/img{frame_idx}.png` (only the selected
frames, no context frames) with a per-video `selected_frames.csv`, and the combined
`manifests/extract.parquet` (`session_id`, `fish_id`, `frame_idx`, `image_path`) that
`cuttle train`/`cuttle embed` read.

## Pipeline: extract embeddings

### `cuttle train` / `cuttle predict`

Thin wrappers around BEAST's own `beast train`/`beast predict` CLI (`beast-backbones`
must be installed, which it is as a dependency of this package) — `cuttle` just resolves
`results_dir` the usual way and passes it to BEAST via its own `--data`/`--output`
flags, so the checked-in configs under `configs/` (e.g. `configs/beast_resnet_ae.yaml`)
can stay machine-agnostic instead of hardcoding a `data_dir`.

```bash
cuttle train --config configs/beast_resnet_ae.yaml --model-name resnet-ae-v1
cuttle predict --model-name resnet-ae-v1 --save-latents
```

`configs/beast_resnet_ae.yaml` is a ResNet18 autoencoder — cheaper to train than the
ViT + MAE + temporal-contrastive architecture that's the eventual target (see
[docs/DECISIONS.md](docs/DECISIONS.md)), so it's the first backbone trained to get the
rest of the pipeline running end to end; a ViT config will follow once that's validated.

`cuttle train` saves under `beast_models/{model_name}` (`--model-name` is required, no
default); `cuttle predict` looks a model back up by that same name. Both
default `--input-dir` to `results_dir/beast_frames` (i.e. the training-frame set from
step 4, not full videos — full-video inference is a later step, once a checkpoint is
trained). `--gpus`/`--nodes`/`--overrides` on `cuttle train`, and `--batch-size`/
`--save-latents`/`--save-reconstructions`/`--output-dir` on `cuttle predict`, pass
straight through to BEAST's own flags of the same purpose.

`cuttle predict` writes under `{model_dir}/image_predictions/{input_dir.stem}` by
default — per-frame embeddings as `latents/{...}/{frame_stem}.npy` when
`--save-latents` is passed, reconstructed images when `--save-reconstructions` is.

### `cuttle embed`

Runs a frozen, pretrained DINOv3 ViT over exported frames — no training required — as an
alternative embedding source alongside `cuttle train`/`cuttle predict`'s from-scratch
BEAST backbones. Requires a Hugging Face token with access to the gated DINOv3 weights
(`hf auth login --token <token>`).

```bash
cuttle embed --backbone vitb16 --resolution 224 --readout cls
```

`--backbone` (`vits16`/`vitb16`/`vitl16`) × `--resolution` (any multiple of 16, e.g.
224/448) × `--readout` (`cls`/`meanpatch_uniform`/`meanpatch_taper`/`gram` — the last
configurable via `--gram-k`/`--gram-weights`) select the embedder; `--model-name`
defaults to `{backbone key}_{readout name}` (e.g. `dinov3_vitb16_224_cls` — see
[docs/implementation_notes/embedder.md](docs/implementation_notes/embedder.md) for the
full key format, including VGG-19).

Writes under `beast_models/{model_name}/`, duck-typed as a BEAST model directory — same
downstream contract as `cuttle predict --save-latents`, but as a single combined
`embeddings.npy` + `manifest.parquet` per run rather than one file per frame (writing
millions of tiny files is extremely slow on some filesystems) — so
`cuttle reduce`/`cuttle cluster`/`cuttle serve` work against it exactly like a trained
BEAST checkpoint. See
[docs/implementation_notes/embedder.md](docs/implementation_notes/embedder.md) for the
full embedder design.

## Pipeline: post-processing

### `cuttle reduce`

Projects the per-frame latents from `cuttle predict --save-latents` to 2D via UMAP, so
they can be compared across hyperparameter settings and, eventually, visualized.

```bash
cuttle reduce --model-name resnet-ae-v1
```

Reads every latent `cuttle predict --save-latents` wrote for the model
(`--predictions-name` defaults to `beast_frames`) and writes one row per frame —
`umap_x`, `umap_y`, plus per-frame metadata — under the model's `reduce/` directory.
`--n-neighbors`/`--min-dist` are encoded into the output filename so different sweeps
don't overwrite each other; `--metric`/`--random-state` are also exposed but aren't
part of the filename — see `--help` for defaults.

### `cuttle cluster`

Assigns a discrete cluster label to every frame, clustering the raw per-frame latents
directly (not the 2D UMAP projection from `cuttle reduce`) — k-means is the only method
so far.

```bash
cuttle cluster --model-name resnet-ae-v1 --n-clusters 10
```

Reads the same latents `cuttle reduce` does (`--predictions-name`, default
`beast_frames`, selects which predicted frame set) and writes one row per frame —
`cluster`, plus per-frame metadata — under the model's `clusters/` directory.
`--n-clusters` is required (no sensible universal default) and is encoded into the
output filename so different sweeps don't overwrite each other; `--random-state` is
also exposed but isn't part of the filename.

### `cuttle clusterview`

For each distinct `cluster` label in a `cuttle cluster` output, randomly samples member
frames and plots them in a grid, one PNG per cluster — a quick visual QC pass before
reaching for the interactive explorer.

```bash
cuttle clusterview --model-name resnet-ae-v1 --cluster-run kmeans_k10
```

Reads the parquet `cuttle cluster` wrote for `--cluster-run` and writes one PNG per
cluster alongside it. `--n-frames` sets how many member frames are sampled per cluster;
`--n-cols` sets the grid width, with row count derived from `--n-frames`; `--seed`
controls the per-cluster sampling — see `--help` for defaults.

### `cuttle serve`

Launches the interactive embedding explorer: a Bokeh dashboard with a dot per frame,
hover to see the corresponding aligned frame image, colored by any UMAP reduction's
cluster/classification attributes. The Model dropdown lists every entry under
`results_dir/beast_models/` — a trained BEAST checkpoint, an MSPS-VAE run, or a
`cuttle embed` output all work the same way.

```bash
cuttle serve
```

`--port` sets the port the app and its frame-image route are served on; `--no-show`
skips auto-opening a browser tab.
