# Decisions

Lightweight decision log for this project. Each entry: what we decided, why, what we
considered instead, and current status. Add new entries at the top. See
[PHASES.md](PHASES.md) for how these play out across the project timeline.

---

## Egocentric alignment via derived videos, not learned invariance

**Date:** 2026-07-08
**Status:** decided

**Decision:** Preprocess raw segmented videos into egocentrically-aligned derived videos
(inscribe a rectangle in the body, rotate/crop to a canonical frame) before any embedding
step, rather than relying on the embedding model to learn orientation invariance.

**Why:** Orientation is a nuisance variable that would otherwise dominate naive pixel-
space or embedding-space clustering (two frames of the same pattern in different
orientations look very different to a clustering algorithm). Handling it with explicit
geometric preprocessing is simpler and more controllable than approaches like contrastive
learning with orientation-invariant augmentations, which add training complexity and
don't guarantee invariance.

**Alternatives considered:** contrastive/self-supervised training with rotation
augmentation to induce invariance — rejected as unnecessarily complex for a problem
solvable geometrically.

---

## Orientation estimation: PCA/ellipse fit first, pose keypoints if needed

**Date:** 2026-07-08
**Status:** decided (first pass); may revisit

**Decision:** Estimate each frame's body axis via PCA/ellipse fit on the segmentation
mask, rather than starting with a trained pose model.

**Why:** No manual labeling required, fastest way to unblock the alignment pipeline
(Phase 2). Good enough to validate the overall pipeline end to end.

**Trade-off / known risk:** PCA gives an axis, not a direction — head/tail must be
disambiguated separately (e.g. via mass asymmetry between mantle and arm-crown, plus
temporal smoothing). The ellipse fit may also be noisy when arm posture changes the
body's effective shape.

**Fallback:** if orientation noise measurably hurts downstream clustering, switch to
pose keypoints (DeepLabCut/SLEAP) tracking e.g. mantle tip and arm-crown center for a
more robust axis. Deferred until we can see how bad the PCA approach actually is on
real data.

---

## Embedding backbone: BEAST, trained from scratch on our data

**Date:** 2026-07-08
**Status:** decided

**Decision:** Use BEAST (BEhavioral Analysis via Self-supervised pretraining of
Transformers) as the embedding model, via the `beast-backbones` pip package (added to
`pyproject.toml`), calling its library functions/CLI rather than reimplementing the
ViT + MAE + temporal-contrastive training loop in this repo.

**Why:** BEAST is designed to be trained per-experiment on a lab's own unlabeled video
(per the paper, this is the intended usage — general-purpose checkpoints trained on
other species/setups aren't expected to transfer well). We train our own backbone on our
aligned cuttlefish frames rather than fine-tuning an existing checkpoint.

**Alternatives considered:** none seriously — this was the intended embedding approach
from the start of the project.

---

## UI framework: Plotly, exact app structure deferred

**Date:** 2026-07-08
**Status:** partially decided

**Decision:** Build the interactive embedding explorer (Phase 7) on Plotly. Whether that
means Dash, a lighter Plotly-based setup, or something else built around Plotly figures
is intentionally left open until we're actually building the UI.

**Why:** Plotly's hover/event model is a natural fit for "hover a dot, show the
corresponding frame image" — the core interaction we need. The specific app framework
choice matters less right now than getting the data pipeline (Phases 0-6) working.

---

## Compute: code targets a local-style multi-GPU workstation

**Date:** 2026-07-08
**Status:** decided

**Decision:** Write training/processing code as if targeting a local multi-GPU
workstation (standard multi-GPU PyTorch, no SLURM/job-array/cluster-scheduler
abstractions), even though execution actually happens on a cloud-hosted multi-GPU
machine.

**Why:** The cloud environment is set up to behave like a local workstation from the
code's perspective, so there's no need for cluster-orchestration complexity.

---

## Data location and per-machine config

**Date:** 2026-07-08
**Status:** decided

**Decision:** Raw data will live at `/media/mattw/poseinterface/cuttle` (not yet
populated as of this writing). Since code will run on multiple (always Linux) machines
with potentially different mount points, each machine gets a local config file at
`~/.cuttle-patterns/config.yaml` specifying `data_dir` and `results_dir`, loaded by a
config module rather than hardcoding paths anywhere in the codebase.

**Why:** Avoids hardcoded paths and machine-specific branches in code; keeps the
data/results location a one-line-per-machine setup step.
