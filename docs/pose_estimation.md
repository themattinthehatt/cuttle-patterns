# Pose Estimation

Tail/neck keypoint predictions consumed by the pose-informed path of `cuttle inscribe`
(see [PHASES.md](PHASES.md) Phase 2b) are produced outside this codebase, using
[Lightning Pose](https://github.com/paninski-lab/lightning-pose).

## Environment

Separate conda env named `pose`, with `lightning-pose` and `lightning-pose-app`
pip-installed into it (kept separate from this repo's own `cuttle` env).

## Labeling

20 frames labeled from each video (tail, neck) via the Lightning Pose app, sampled from a subset 
of the videos in Day1-Day3 — 720 labeled frames total.

## Training

Three supervised models planned (random seeds 0, 1, 2), each:
- backbone: `vits_dino`
- data augmentation: `dlc-top-down`
- 500 epochs

Models are named `{iteration}_vits-dino_seed-{n}`, where `iteration` tracks successive
rounds of the active-learning loop below (`iter-1.0` = trained on the initial 720 frames;
later iterations add frames selected from `iter-1.0`'s QC review, etc.).

**Status as of 2026-08-28:** `iter-1.0_vits-dino_seed-{0,1,2}` are all trained, with video
inference run for all three. A second round, `iter-1.1_vits-dino_seed-{0,1,2}`, is
currently training (see Active learning below for what frames were added). Project lives
at `/media/mattw/CUTTLE/pose-estimation/cuttle-test/` (Lightning Pose project directory):
`models/{model_name}/video_preds/{video_name}.csv` holds per-video predictions in the
standard 3-header format `cuttle_patterns/preprocessing/pose.py` parses;
`CollectedData.csv` at the project root is the ground-truth labeled-frame manifest, one
row per label as `labeled-data/{video_name}/img{frame_idx:08d}.jpg`. Frame indices are
0-based and consistent across this file, each `_black_frames.txt`, and the `video_preds`
CSV row order.

## Active learning: selecting new frames to label

**Status (round 1, complete):** ran `scripts/pose_plot_outliers.py` against
`iter-1.0_vits-dino_seed-{0,1,2}` over the 32 videos with predictions available at the
time (Day1-Day3; the 6 Day4 videos were skipped, no inference run yet). Manually paged
through the QC images and recorded the selected frame indices in
`{project_dir}/qc/selected_frames.txt` (one line per video: `{video_name} -
{frame_idx}, {frame_idx}, ...`) — 335 frames total across those 32 videos. Those frames
have since been labeled and added to `CollectedData.csv`, and round 2 training
(`iter-1.1_vits-dino_seed-{0,1,2}`) is underway using the expanded label set.

`scripts/pose_plot_outliers.py` (not yet promoted into `cuttle_patterns/` + the `cuttle`
CLI — see the `scripts/` vs `scratch/` distinction in [DECISIONS.md](DECISIONS.md)) ranks
candidate frames by cross-model prediction disagreement:

1. For each video, load tail/neck predictions from whichever of the given `--models` have
   a `video_preds` CSV for it; skip the video if fewer than 2 do (variance is meaningless
   with a single point).
2. Score each frame as `max over {tail, neck} of (var(x across models) + var(y across
   models))` (population variance, no likelihood weighting).
3. Exclude frames already in `CollectedData.csv` or flagged blank.
4. Take the `--top-k` (default 100) highest-scoring frames per video.
5. Write a QC image per selected frame — raw frame (no inscribed rectangle), every
   available model's tail (circle) / neck (square) prediction overlaid in a color fixed
   to that model's position in `--models` (consistent across every video/frame, with a
   legend burned into the image), drawn regardless of likelihood since disagreement is
   the signal being surfaced — to
   `{project_dir}/qc/{video_name}/rank{rank:03d}_frame{frame_idx:08d}_var{score:.1f}.png`.

Adding a frame to the actual labeling queue (extracting it into `labeled-data/` and
appending to `CollectedData.unlabeled.jsonl`) stays a manual step in the Lightning Pose
app after paging through the QC images — the script only selects and renders candidates.

## Open questions

- How the three seeded models are combined/selected for the predictions actually
  consumed by `cuttle inscribe` (ensembling vs. picking one) — not yet decided.
