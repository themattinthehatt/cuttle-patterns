# cuttle-patterns

Unsupervised analysis of visual patterns displayed by cuttlefish during social
interaction: egocentric alignment of segmented cuttlefish videos, self-supervised
embedding via BEAST, and interactive tools for exploring the resulting pattern clusters.

## Docs

- [docs/PHASES.md](docs/PHASES.md) — project phases/roadmap
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log

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

Scans `data_dir` for raw `session-{id}_cuttle-{id}.mp4`/`.txt` pairs and writes a
manifest of what's there (frame counts, fps, resolution, flagged-blank counts) to
`results_dir/manifests/ingest.parquet`.

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
  safe to run over a mix of videos with and without predictions.

Either mode can still leave the rectangle jittering frame-to-frame during rapid body
motion (fin beats in particular); the corner trajectory is smoothed to damp this, via
either a centered rolling median (`--smoothing-window`, frames, default 9; 1 disables
smoothing) or a Gaussian filter (`--smoothing-sigma`, standard deviation in frames,
default 2.0 if given with no value) — the two are mutually exclusive. The Gaussian tends
to track continuous, quasi-periodic jitter (e.g. fin beats) more smoothly since it blends
the whole window rather than snapping to one observed value; the median is more robust to
an occasional single-frame garbage detection, since it rejects rather than blends it in.

To process one video at a time (e.g. while iterating on `--thresh`/`--aspect`), use
`--video-path`, optionally paired with an explicit `--pose-path`:

```bash
cuttle inscribe --video-path /path/to/session-01_cuttle-01.mp4 \
  --pose-path /path/to/session-01_cuttle-01_pose.csv
```

### 3. `cuttle overlay` (optional QC)

Draws each frame's (interpolated) rectangle on top of the corresponding *raw* frame —
green if directly detected, orange if interpolated — so inscription quality can be
checked visually before moving on. Reuses `cuttle inscribe`'s `.csv` if it already
exists for a video; otherwise runs the same detection first (accepting the same
`--pose-dir`/`--pose-path`/`--thresh`/`--aspect` flags as `cuttle inscribe`, used only if
it has to compute the CSV itself).

```bash
cuttle overlay
```

Writes `results_dir/rectangles/{video_name}_overlay.mp4`, H.264-encoded (via `ffmpeg`)
since these are full raw-resolution videos and can otherwise get large; tune size vs.
quality with `--crf` (lower is higher quality/larger file, default 28).
