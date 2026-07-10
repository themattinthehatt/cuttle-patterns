# Decisions

Lightweight decision log for this project. Each entry: what we decided, why, what we
considered instead, and current status. Add new entries at the top. See
[PHASES.md](PHASES.md) for how these play out across the project timeline.

---

## Phase 2b keypoint scheme: tail + neck only, mask-cut over ellipse-fit

**Date:** 2026-07-10
**Status:** decided (design); implementation blocked on pose labeling/training

**Decision:** Use two pose keypoints (tail tip, head/body "neck" transition point)
rather than the originally planned four (tail, neck, two lateral mantle-width points).
Use the neck point as a mask-cutting boundary (zero out everything past it — the
head/arm side) combined with the signed tail→neck vector for orientation, then reuse
Phase 2a's existing mask-based sizing pipeline (distance-transform seed + integral-image
rectangle growth, via `seed_from_distance_transform`/`grow_rectangle`) on the resulting
mantle-only mask — rather than fitting a synthetic ellipse from four keypoints. Full
technical plan in [PHASES.md](PHASES.md) Phase 2b.

**Why:** A first real labeling attempt at the original 4-point scheme found the two
lateral "width" points difficult to label consistently — much less anatomically
well-defined landmarks than tail-tip and neck-transition. Dropping them isn't just a
labeling-effort compromise: reusing the real (cut) mask for sizing instead of an ellipse
also eliminates a known limitation of the original plan (a symmetric ellipse couldn't
capture the mantle's true taper), so the new design is a strict improvement, not a
fallback.

**Alternatives considered:** the original 4-point ellipse-fit plan (see "Orientation
estimation" entry below) — rejected for the reasons above.

**Trade-off / known risk:** loses keypoint-derived occlusion robustness specifically for
the width dimension — if the mask is corrupted right at the mantle in the one
already-unresolved edge case (an occluding blob touching the image border), width sizing
is still affected, where a pure keypoint-derived ellipse would have been immune. This
narrow risk was already accepted as part of the mask-recovery decision below; nothing
new here.

---

## Body mask recovery: background-complement, not intensity thresholding

**Date:** 2026-07-10
**Status:** decided

**Decision:** Recover each frame's body mask by labeling connected components of near-
black (`<= thresh`, default 0) pixels, keeping only the single largest as the true
background, and treating everything else as body — rather than a plain intensity
threshold plus largest-foreground-component selection.

**Why:** The background is confirmed pure black (pixel value 0, verified via sampled
corner pixels, no compression noise). But the cuttlefish's own dark chromatophore
patterning also renders at or near 0, so pixel intensity alone can't distinguish body
from background. A naive threshold (tried at both 10 and 1) left the mask riddled with
holes wherever the animal displayed dark patterning, which shrank and mis-centered the
Phase 2a inscribed rectangle (see [PHASES.md](PHASES.md) Phase 2a). Splitting on
connected components of near-black pixels instead correctly reclassifies isolated dark
patches on the body as foreground, since they aren't connected to the true background
blob, regardless of whether they're fully enclosed.

**Alternatives considered:**
- `scipy.ndimage.binary_fill_holes` on the thresholded mask — only fixes holes fully
  enclosed by foreground; missed holes that touch the mask's outer boundary (common near
  the arm crown).
- Morphological closing — bridges gaps regardless of shape, but the kernel size needed
  to close the branchier chromatophore-driven gaps also uniformly rounds off real
  anatomical concavities (e.g. the notch between the mantle and arm crown).
- Convex hull of the mask — parameter-free but too permissive: it also fills real empty
  space between splayed arms, risking the rectangle landing in open water rather than on
  the animal.

**Trade-off / known risk:** assumes a single dominant background region per frame (holds
as long as the body doesn't split the frame into disconnected background pockets).

---

## CLI structure: auto-discovered `cmd_*.py` modules, mirroring `crittercam`

**Date:** 2026-07-09
**Status:** decided

**Decision:** Expose pipeline steps as subcommands of a single `cuttle` console script
(`cuttle ingest`, and future `cuttle align`, `cuttle extract-frames`, etc.), implemented
under `cuttle_patterns/cli/`: `main.py` builds the root argparse parser and auto-
discovers every `cmd_*.py` file in the same directory via `Path.glob('cmd_*.py')`,
importing each and calling its `register(subparsers)`; each `cmd_<name>.py` owns its
argparse wiring (`register`) and a thin `cmd_<name>(args)` handler that delegates to real
logic in a top-level module (e.g. `cmd_ingest.py` → `cuttle_patterns.ingest.build_manifest`).

**Why:** Matches an existing project (github.com/themattinthehatt/crittercam) the user
already has conventions and muscle memory for. Adding a new pipeline step is just adding
one `cmd_<name>.py` file — no central registry to edit, `main.py` never changes.

**Alternatives considered:** none — explicitly requested to match the prior project's
pattern.

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

**Date:** 2026-07-08 (arm-bias confirmed on real data 2026-07-10)
**Status:** decided (first pass); arm-bias confirmed — Phase 2b prioritized

**Decision:** Estimate each frame's body axis via PCA/ellipse fit on the segmentation
mask, rather than starting with a trained pose model.

**Why:** No manual labeling required, fastest way to unblock the alignment pipeline
(Phase 2). Good enough to validate the overall pipeline end to end.

**Trade-off / known risk:** PCA gives an axis, not a direction — head/tail must be
disambiguated separately. Confirmed on real data (session-01 sample, see
[PHASES.md](PHASES.md) Phase 2a): once the mask-recovery fix above stopped fragmenting
the mask with dark-patterning holes, arm posture became the dominant failure mode — PCA
frequently finds its largest rectangle straddling the arm crown rather than centered on
the mantle, since splayed arms are part of the same connected component as the mantle.

**Fallback:** switch to pose keypoints for a more robust axis. Originally scoped as a
4-keypoint ellipse fit; superseded by a 2-keypoint (tail + neck) mask-cut design — see
the "Phase 2b keypoint scheme" entry above and [PHASES.md](PHASES.md) Phase 2b for the
current plan. Prioritized now that the PCA failure mode is confirmed on real data rather
than hypothetical.

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

**Decision:** Raw data lives at `/media/mattw/poseinterface/cuttle/data` (populated as of
2026-07-09 with the first session), with a separate sibling `/media/mattw/poseinterface/
cuttle/results` for everything this codebase generates (manifests, embeddings,
checkpoints). Since code will run on multiple (always Linux) machines with potentially
different mount points, each machine gets a local config file at
`~/.cuttle-patterns/config.yaml` specifying `data_dir` and `results_dir`, loaded by a
config module rather than hardcoding paths anywhere in the codebase.

**Why:** Avoids hardcoded paths and machine-specific branches in code; keeps the
data/results location a one-line-per-machine setup step.
