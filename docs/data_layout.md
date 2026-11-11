# Data layout

Where the data lives on disk, how files are named, and the known inconsistencies in the
raw data. Facts here were checked against the data on 2026-11-10/11; re-verify before
relying on a specific number. For the layout of a `results_dir` tree, see the module
docstring of `cuttle_patterns/paths.py` (the single source of truth for it).

Root: `/media/mattw/CUTTLE/`

    CUTTLE/
    ├── raw_data/            # READ-ONLY. crop videos, bbox csvs, canvas_sides.csv
    ├── raw_data_samples/    # READ-ONLY. downscaled full-frame stills per tank
    ├── results/             # pipeline outputs, one results_dir per iteration
    ├── classification/      # skin-pattern classifier models
    └── pose-estimation/     # external tail/neck pose model files

`raw_data`, `raw_data_samples` and existing `results/.iter-*` dirs are never moved,
deleted or modified. Derived data goes in a new results dir (see below).

## raw_data

12 days x 6 tanks x 2 cuttlefish. Per tank (`Day{D}_Tank{T}`):

| File | Contents |
|------|----------|
| `Day{D}_Tank{T}_bboxes.csv` | one row per (frame, obj_id): mask-space and original-resolution bounding boxes |
| `Day{D}_Tank{T}_Cuttle{N}_{Role}_Crop.mp4` | 512x512 crop video of one animal |
| `Day{D}_Tank{T}_Cuttle{N}_{Role}_black_frames.txt` | one frame index per line (frames flagged black) |

- `Role` is `Resident` or `Intruder`; which of Cuttle1/Cuttle2 holds which role varies by
  tank (e.g. Day 1 Tank 2 has Cuttle2 as Intruder, Day 2 Tank 1 the opposite).
- `canvas_sides.csv`: one row per crop video (`day`, `tank`, `cuttle`, `role`, `video`,
  `resized`, `box`, `canvas_side`, `scale_video_per_original`, `n_frames`, edge stats).
  `day`/`tank` are written `DAY_1`/`Tank_1`, unlike the `Day1`/`Tank1` filenames.
- `identity_mapping.txt`: 72 lines, one per tank-day, mapping to the original recording
  names (`2025-06-02_RIassay_Day1_Tank1_R1-I1_exp1`, ...).
- `iter-1/`: subdirectory holding a second copy of crop videos and black-frame files
  (289 files); the top level is the reference.
- `bboxes.csv` columns: `day, tank, chunk, frame, obj_id`, mask-space size
  (`mask_w`, `mask_h` = 771x434), `merged`, `n_components`, `raw_*` / `main_*` boxes
  (mask space), and `orig_raw_*` / `orig_main_*` boxes (original 3856x2170 px).

### Crop-to-original mapping

A crop is a square window of `canvas_side` original pixels (from `canvas_sides.csv`),
centered on the bbox center, resized to 512. Use `orig_raw_*` or `orig_main_*` boxes
according to the `box` column. `obj_id` equals the Cuttle number. Verified on 135 of 144
videos (crop/bbox size ratio == 512 / canvas_side); the exceptions are listed under known
issues.

When mapping a crop frame back to the full frame, pick the bbox for each frame by
overlap/size match, not by `obj_id` alone (ids flip in a few videos, below).

## raw_data_samples

`raw_data_samples/Day{D}_Tank{T}/Day_{D}_tank_{T}_frame_{NNNNNN}.png`: about 1000
full-tank stills per tank (the 72 dirs mirror the 72 tank-days), downscaled to 771x434
(original / 5). The sample label `N` corresponds to video frame `N - 1`. Some Day 1 samples
were stored at 1156 px wide and later resized to 771; use the manifest `out_w`, not
`src_w`, for scale.

## results

Each `results/<name>/` is a self-contained `results_dir`. The tree inside one (frames,
models, manifests, clusters, classifications, eval, ...) is defined in
`cuttle_patterns/paths.py`; this table only says what each iteration dir is:

| Dir | Contents |
|-----|----------|
| `.extraction`, `.iter-0.0` | early extraction experiments and the first iteration; kept for history |
| `.iter-1.0` | first full iteration: Day 1 Tanks 2-6, Day 2 Tanks 1-6, Day 3 Tanks 2-6, 2000 frames x 32 videos |
| `.iter-1.0-photometry` | photometric-correction experiment on the same frames: `params/`, `beast_frames_*` (corrected and control frames), `variants/{orig,raw,final,wrong,norm}/` (self-contained results dirs, 25,756 identical frames, 26 videos) |
| `manifests`, `rectangles` | top-level dirs from before the dot-prefixed iteration dirs |

A variant dir for an ablation follows the same `paths.py` tree: its own
`manifests/extract.parquet` and frames, plus symlinks (not copies) to anything unchanged,
such as `classifications/`.

## Known issues (as of 2026-11-11)

Crop, csv and canvas inconsistencies:

- **Day 1 Tank 1**: both videos' scale and obj_id disagree with the csv.
- **Day 5 Tank 4, Day 6 Tank 3, Day 8 Tank 1**: the matching obj_id flips once mid-video
  during a blank gap; the canvas values look swapped.
- **Day 10 Tank 3 Cuttle 2**: crop scale about 3% off the csv value.
- **Day 1 Tank 2**: Resident/Intruder role labels swapped between csv and filenames.
- **Re-made crops** (`resized=True` in `canvas_sides.csv`, mp4s modified 2026-11-10): Day 1
  Tank 3 (both), Day 1 Tank 4 (both), Day 3 Tank 2 Cuttle 2, Day 3 Tank 6 Cuttle 1. They
  no longer reproduce the iter-1.0 frames (mean abs diff about 33 vs about 3 for
  untouched crops), and their bbox info is not usable for the old crops.
- **52 csv entries** list crop file names with a `_Crop_resized` suffix that the files on
  disk do not have.

Sample frames:

- **Day 1 Tank 3, Day 2 Tank 2** sample frames are rotated 180 degrees relative to the same
  tank on other days.

Image content:

- **Day 1 Tanks 2, 4, 6 are fogged**: floor brightness about 0.6 vs about 0.41 elsewhere,
  bluer, and animals' black level (L* 2nd percentile) about 14-25 vs 3-7. Tanks 3 and 5 look
  normal. A photometric correction removed the effect on embeddings only for these
  sessions; the decision was to exclude Day 1 instead, see
  [DECISIONS.md](DECISIONS.md).
- A raised dark level alone is not unique to Day 1 (Day 10 Tanks 2 and 4 are also
  elevated); not yet checked in animal pixels.

Questions about the crop/csv problems were sent to the data collaborator on 2026-11-10;
update this list when they reply.
