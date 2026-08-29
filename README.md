# cuttle-patterns

Unsupervised analysis of visual patterns displayed by cuttlefish during social
interaction: egocentric alignment of segmented cuttlefish videos, self-supervised
embedding via BEAST, and interactive tools for exploring the resulting pattern clusters.

## Docs

- [docs/PHASES.md](docs/PHASES.md) — project phases/roadmap
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log
- [docs/pose_estimation.md](docs/pose_estimation.md) — pose model used for pose-informed
  `cuttle inscribe`

## Setup

Requires the `ffmpeg` binary on `PATH` (e.g. `apt-get install ffmpeg` /
`brew install ffmpeg`) — `cuttle overlay` shells out to it for H.264-encoded output.

```bash
conda create -n cuttle python=3.12
conda activate cuttle
pip install -e ".[dev]"
```

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

## Pipeline

The first few steps after `cuttle setup`, run in order. Every subcommand reads
`data_dir`/`results_dir` from the config file by default; override either with
`--data-dir`/`--results-dir` if needed.

### 1. `cuttle ingest`

Scans `data_dir` for raw `Day{day}_Tank{tank}_Cuttle{n}_{role}_crop.mp4`/
`_black_frames.txt` pairs and writes a manifest of what's there (frame counts, fps,
resolution, flagged-blank counts) to `results_dir/manifests/ingest.parquet`.

```bash
cuttle ingest
```

### 2. `cuttle inscribe`

Inscribes an egocentric rectangle in the cuttlefish's body on every frame, then warps it
into a fixed-size, upright crop video. For each raw video, writes
`results_dir/rectangles/{video_name}.mp4` (the aligned crop) and `{video_name}.csv`
(per-frame rectangle corners plus an `is_interpolated` flag for frames where nothing was
detected directly and the rectangle had to be filled in).

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
either a Gaussian filter (`--smoothing-sigma`, standard deviation in frames, 2.0 if
given with no value — **the default if neither flag is given**) or a centered rolling
median (`--smoothing-window`, frames, 9 if given with no value; 1 disables smoothing) —
the two are mutually exclusive. The Gaussian tracks continuous, quasi-periodic jitter
(e.g. fin beats) more smoothly since it blends the whole window rather than snapping to
one observed value; the median is more robust to an occasional single-frame garbage
detection, since it rejects rather than blends it in.

To process one video at a time (e.g. while iterating on `--thresh`/`--aspect`), use
`--video-path`, optionally paired with an explicit `--pose-path`:

```bash
cuttle inscribe --video-path /path/to/Day1_Tank2_Cuttle1_Resident_Crop.mp4 \
  --pose-path /path/to/Day1_Tank2_Cuttle1_Resident_Crop_pose.csv
```

Pass `--skip-existing` to leave a video alone (no re-inscription) if its
`{video_name}.mp4`/`.csv` already exist in `output_dir`, so a batch run can be safely
re-run over a directory that's only partially processed.

### 3. `cuttle overlay` (optional QC)

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

Writes `results_dir/rectangles/{video_name}_overlay.mp4`, H.264-encoded (via `ffmpeg`)
since these are full raw-resolution videos and can otherwise get large; tune size vs.
quality with `--crf` (lower is higher quality/larger file, default 28). As with `cuttle
inscribe`, pass `--skip-existing` to leave a video's `{video_name}_overlay.mp4` alone if
it already exists, rather than re-encoding it.

### 4. `cuttle extract`

Selects a diverse, representative set of still frames from the aligned crop videos
(`results_dir/rectangles/{video_name}.mp4`) to train BEAST (Phase 4) on. Per video:

1. Remove frames that are blank, have any tail/neck keypoint likelihood below 0.9, or
   have a rectangle (from `cuttle inscribe`) whose long edge is less than 50% of the
   neck-tail distance — a body clearly longer than the box drawn around it, found via QC
   on real data (`scratch/qc_small_rectangles.py`).
2. Keep only the survivors whose immediate neighbors also survived step 1, so every frame
   BEAST will see as temporal context is itself a valid frame.
3. From that candidate set, select anchor frames during movement via motion-energy
   thresholding, PCA, and k-means — a fork of
   `beast.preprocess.extraction.select_frame_idxs_kmeans` (BEAST v2.0.0) restricted to
   only ever pick anchors from the candidate set.

```bash
cuttle extract --pose-dir /path/to/pose/predictions
```

`--pose-dir` is **required, with no default** — keypoint-likelihood and rectangle-size
filtering are required parts of the algorithm here, not an optional refinement like in
`cuttle inscribe`/`cuttle overlay`. A video with no matching `{video_name}.csv` in
`--pose-dir`, no matching `{video_name}.csv` rectangle geometry in `--input-dir` (written
alongside the video by `cuttle inscribe`), or a missing blank-frames `.txt` in `data_dir`
is skipped/warned about rather than silently including unfiltered frames.

For each video, writes `results_dir/beast_frames/{video_name}/img{frame_idx}.png` for
every selected anchor frame plus its immediate neighbors (context for BEAST's temporal
training), and a `selected_frames.csv` listing just the anchor frames — matching BEAST's
own `extract_frames` output layout. Across all videos, also writes a combined
`results_dir/manifests/extract.parquet` (`session_id`, `fish_id`, `frame_idx`,
`image_path`, one row per selected anchor frame).

`-n`/`--frames-per-video` caps the number of anchor frames selected per video (default 1000) — 
a maximum, not an exact count: a video with fewer surviving candidate frames than
that just uses all of them, with a printed warning. As with the earlier steps,
`--skip-existing` skips a video whose `beast_frames/{video_name}/selected_frames.csv`
already exists, and `--video-path`/`--pose-path` process a single video against an
explicit pose CSV instead of scanning `--input-dir` (default `results_dir/rectangles`).

### 5. `cuttle train` / `cuttle predict`

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

`cuttle train` saves to `results_dir/beast_models/{model_name}` (`--model-name` is
required, no default); `cuttle predict` looks a model back up by that same name. Both
default `--input-dir` to `results_dir/beast_frames` (i.e. the training-frame set from
step 4, not full videos — full-video inference is a later step, once a checkpoint is
trained). `--gpus`/`--nodes`/`--overrides` on `cuttle train`, and `--batch-size`/
`--save-latents`/`--save-reconstructions`/`--output-dir` on `cuttle predict`, pass
straight through to BEAST's own flags of the same purpose.

`cuttle predict` writes under `{model_dir}/image_predictions/{input_dir.stem}` by
default — per-frame embeddings as `latents/{...}/{frame_stem}.npy` when
`--save-latents` is passed, reconstructed images when `--save-reconstructions` is.
