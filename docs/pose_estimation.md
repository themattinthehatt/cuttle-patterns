# Pose Estimation

**Doc type:** living status doc — edited in place each active-learning round.

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

**Status as of 2026-11-10:** `iter-1.0`, `iter-1.1`, and `iter-1.2`
(`_vits-dino_seed-{0,1,2}` each) are all trained (see Active learning below for what
frames were added in each round), with video inference run for all nine models. `iter-1.1`
and `iter-1.2` inference cover every video (Day1-Day12, 144 videos). The label set grew
720 (`iter-1.0`) -> 1018 (`iter-1.1`) -> 1295 (`iter-1.2`) frames; the `iter-1.2` increment
comes from the round 2 QC review below. `iter-1.2_eks` holds the ensemble-smoothed
predictions (see Ensemble Kalman Smoother below). Project lives at
`/media/mattw/CUTTLE/pose-estimation/cuttle-test/` (Lightning Pose project directory):
`models/{model_name}/video_preds/{video_name}.csv` holds per-video predictions in the
standard 3-header format `cuttle_patterns/preprocessing/pose.py` parses (each model dir also
has `*_temporal_norm.csv` companions, which are not predictions);
`CollectedData.csv` at the project root is the ground-truth labeled-frame manifest, one
row per label as `labeled-data/{video_name}/img{frame_idx:08d}.jpg`. Frame indices are
0-based and consistent across this file, each `_black_frames.txt`, and the `video_preds`
CSV row order.

## Active learning: selecting new frames to label

**Status (round 1, complete):** ran `scripts/pose_plot_outliers.py` against
`iter-1.0_vits-dino_seed-{0,1,2}` over the 32 videos with predictions available at the
time (Day1-Day3; the 6 Day4 videos were skipped, no inference run yet). Manually paged
through the QC images and recorded the selected frame indices in
`{project_dir}/qc/iter-1.0/selected_frames.txt` (one line per video: `{video_name} -
{frame_idx}, {frame_idx}, ...`) — 335 frames total across those 32 videos. Those frames
have since been labeled and added to `CollectedData.csv`, and round 2 training
(`iter-1.1_vits-dino_seed-{0,1,2}`) on the expanded label set is complete.

**Status (round 2, complete):** ran `scripts/pose_plot_outliers.py --min-day 4` against
`iter-1.1_vits-dino_seed-{0,1,2}` over the 108 videos from Day4-Day12 (Day1-Day3 were
already covered in round 1), with `--output-dir {project_dir}/qc/iter-1.1`. Each video got
100 top-variance frames plus 10 random mid-percentile (`_mid`) frames. Selected frame
indices were recorded in `{project_dir}/qc/iter-1.1/selected_frames.txt` (265 frames), the
frames were labeled and added to `CollectedData.csv`, and round 3 training
(`iter-1.2_vits-dino_seed-{0,1,2}`, same recipe, 1295 labels) is complete.

`scripts/pose_plot_outliers.py` (not yet promoted into `cuttle_patterns/` + the `cuttle`
CLI — see the `scripts/` vs `scratch/` distinction in [DECISIONS.md](DECISIONS.md)) ranks
candidate frames by cross-model prediction disagreement. `--min-day N` restricts it to
videos from Day N onwards, and `--output-dir` (default `{project_dir}/qc`) is where the
per-video QC folders go — use a per-round subfolder like `qc/iter-1.1`:

1. For each video, load tail/neck predictions from whichever of the given `--models` have
   a `video_preds` CSV for it; skip the video if fewer than 2 do (variance is meaningless
   with a single point).
2. Score each frame as `max over {tail, neck} of (var(x across models) + var(y across
   models))` (population variance, no likelihood weighting).
3. Exclude frames already in `CollectedData.csv` or flagged blank.
4. Take the `--top-k` (default 100) highest-scoring frames per video.
5. Append `--n-mid` (default 10) frames drawn at random from the 45th-55th percentile band
   (`--mid-percentiles`) of the video's remaining candidate scores, so the labeling set
   isn't only the very hardest frames. Sampling is seeded from `--seed` (default 0)
   combined with the video name, so a video's picks don't depend on which other videos
   are processed. They continue the rank numbering after the top-k and carry a `_mid`
   filename suffix.
6. Write a QC image per selected frame — raw frame (no inscribed rectangle), every
   available model's tail (circle) / neck (square) prediction overlaid in a color fixed
   to that model's position in `--models` (consistent across every video/frame, with a
   legend burned into the image), drawn regardless of likelihood since disagreement is
   the signal being surfaced — to
   `{output_dir}/{video_name}/rank{rank:03d}_frame{frame_idx:08d}_var{score:.1f}[_mid].png`.

Adding a frame to the actual labeling queue (extracting it into `labeled-data/` and
appending to `CollectedData.unlabeled.jsonl`) stays a manual step in the Lightning Pose
app after paging through the QC images — the script only selects and renders candidates.

## Ensemble Kalman Smoother

The three seeded models of a round are combined with the
[Ensemble Kalman Smoother](https://github.com/paninski-lab/eks) (`ensemble-kalman-smoother`
package, `eks`), run single-view (no multi-view PCA) with a fixed `smooth_param` of 1000.
That value is large enough that the smoother barely smooths, so the output is close to a
per-frame median across the three models (on one test video the mean absolute difference
from the median was ~0.2 px), with EKS-derived uncertainty on top.

`scripts/pose_run_eks.py` (not yet promoted into `cuttle_patterns/` + the `cuttle` CLI) runs
it over every video that has a `video_preds` CSV in all of the given models
(`*_temporal_norm.csv` files are ignored; videos missing from some models are skipped with a
warning), and writes `{project_dir}/models/{output_name}/video_preds/{video_name}.csv`:

```
python scripts/pose_run_eks.py \
    --models iter-1.2_vits-dino_seed-0 iter-1.2_vits-dino_seed-1 iter-1.2_vits-dino_seed-2 \
    --output-name iter-1.2_eks
```

`--smooth-param` overrides the 1000 default; existing outputs are skipped unless
`--overwrite` is passed. Requires `ensemble-kalman-smoother` in the active env. The output
keeps the standard `x`/`y`/`likelihood` columns per keypoint (scorer `ensemble-kalman_tracker`),
so it loads with `cuttle_patterns/preprocessing/pose.py` like any model's predictions, and
adds `x_ens_median`, `x_ens_var`, `x_posterior_var` (and `y_` equivalents) columns.

**Status:** `iter-1.2_eks` is the target output directory (Lightning Pose app already created
`iter-1.2_eks/ensemble.yaml` listing the three members and `smooth_param: 1000.0`); the script
has been verified on one video but not yet run over all 144.

## Open questions

- Whether `cuttle inscribe` should consume the `iter-1.2_eks` predictions (vs. a single
  seed's) — not yet decided.
