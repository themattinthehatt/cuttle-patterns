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

Three supervised models trained (random seeds 0, 1, 2), each:
- backbone: `vits_dino`
- data augmentation: `dlc-top-down`
- 500 epochs

## Open questions

- How the three seeded models are combined/selected for the predictions actually
  consumed by `cuttle inscribe` (ensembling vs. picking one) — not yet decided.
- Active learning loop for selecting new frames to label beyond the initial 720 — planned,
  not yet designed.
