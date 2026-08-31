# Decisions

Lightweight decision log for this project. Each entry: what we decided, why, what we
considered instead, and current status. Add new entries at the top. See
[PHASES.md](PHASES.md) for how these play out across the project timeline.

---

## Small-rectangle filter: exclude frames where the box is much shorter than the body

**Date:** 2026-08-28
**Status:** decided, implemented

**Decision:** Add a third criterion to `cuttle extract`'s frame-filtering step (alongside
blank frames and low keypoint likelihood): exclude a frame if, among frames where both
tail/neck keypoints meet the likelihood threshold, the inscribed rectangle's long edge is
less than 50% of the neck-tail distance. Implemented as
`extract.compute_small_rectangle_mask`, called from `extract.compute_filtered_frame_mask`
whenever both a pose CSV and a rectangle-geometry CSV are available for the video.

**Why:** While QC'ing `cuttle extract`'s output on real data, a scratch script
(`scratch/qc_small_rectangles.py`) sampling example frames from each video's
`_overlay.mp4` surfaced a cheap, reliable defect signal: frames where `cuttle inscribe`'s
rectangle is clearly too short for the body it's supposed to bound, visible immediately
by eye against the overlay's keypoints. Across all 32 real videos available at the time,
6,921/1,437,351 frames (0.5%) were flagged at the 50% threshold — worth excluding from
BEAST's training set (bad alignment geometry would otherwise leak orientation/scale noise
into pattern-focused training) at negligible cost to how much data remains.

**Alternatives considered:** fixing the underlying `cuttle inscribe` sizing bug that
produces these rectangles — not pursued yet, since the QC script's job was to first
quantify and visualize the problem, not diagnose its root cause; filtering is the correct
short-term mitigation regardless of what that root cause turns out to be, and doesn't
block making progress on Phase 3/4 while it's investigated.

**Trade-off / known risk:** 50% was chosen directly from the QC script's default (an
initial guess later tightened from 75% after visual review of examples at that looser
threshold), not derived from a labeled ground truth of "small enough to hurt training" —
revisit if the flagged/kept split still looks wrong on visual spot-check. The check only
runs where likelihood is already high, so it can't catch a genuinely small rectangle
alongside a genuinely wrong (but confidently predicted) keypoint pair; that failure mode
is unaddressed by this or any other current filter.

---

## `cuttle train`/`cuttle predict`: subprocess wrappers around BEAST's own CLI

**Date:** 2026-08-28
**Status:** decided, implemented

**Decision:** Wrap BEAST's own `beast train`/`beast predict` CLI commands in
`cuttle train`/`cuttle predict` (`cuttle_patterns/cli/cmd_train.py`/`cmd_predict.py`) by
building the equivalent `beast` argv and running it as a subprocess (stdout/stderr
inherited, so BEAST's own training/inference logs stream live) — not by calling
`beast.api.model.Model` directly in-process. `cuttle` resolves `results_dir` the usual
config-or-override way and passes it to BEAST via its own `--data`/`--output` flags
(`beast train`) and `--model`/`--input` (`beast predict`), so the checked-in configs
under `configs/` can stay machine-agnostic instead of committing one machine's absolute
`data_dir`. Models are saved to `results_dir/beast_models/{model_name}` — `--model-name`
is required on `cuttle train` (no default) and looked up again by `cuttle predict`, so
the two commands share one naming scheme. `cuttle train` also passes through BEAST's
`--gpus`/`--nodes`/`--overrides` flags, since real training targets the multi-GPU cloud
machine (see the "Compute" entry below).

**Why:** A subprocess wrapper is a few dozen lines and reuses everything BEAST's own CLI
commands already do correctly (config loading/validation via pydantic, output-dir setup,
its own logging setup) — calling `Model` directly would mean reimplementing that
scaffolding in `cuttle_patterns` for no real benefit, and would couple this repo to
BEAST's internal API surface rather than its public CLI contract, which is more likely to
stay stable across `beast-backbones` versions.

**Alternatives considered:** calling `beast.api.model.Model.from_config(...).train(...)`/
`Model.from_dir(...).predict_images(...)` directly — rejected per above; capturing
subprocess output instead of inheriting stdout/stderr — rejected, since BEAST's training
logs are long-running and meant to be watched live, not buffered until the process exits.

**Trade-off / known risk:** errors surface as BEAST's own CLI error output plus a bare
nonzero exit code propagated from `cuttle`, not a `cuttle`-specific error message (except
the two fail-fast checks `cuttle` does itself: `beast` missing from `PATH`, and
`cuttle predict`'s model directory not existing). `cuttle train` warns but doesn't block
when `--model-name` points at a non-empty existing directory, since BEAST's CLI doesn't
support resuming (a fresh run just starts training from scratch again) — re-running with
the same name is on the user to intend.

---

## Backbone sequencing: train a ResNet18 autoencoder before the ViT

**Date:** 2026-08-28
**Status:** decided, implemented (config only — training itself run manually via BEAST's
own CLI, not yet executed against real data)

**Decision:** Train a ResNet18 autoencoder (`configs/beast_resnet_ae.yaml`,
`model_class: resnet`, no contrastive loss) as the first BEAST backbone, rather than
going straight to the ViT + MAE + temporal-contrastive design that's the eventual target
(see "Embedding backbone" below and [PHASES.md](PHASES.md) Phase 4).

**Why:** The ViT is meaningfully more expensive to train. Getting the full pipeline
(Phase 4 training → Phase 5 embedding extraction → Phase 6 clustering → Phase 7
visualization) running end to end against a cheap backbone first surfaces pipeline/data
issues (data-loading contract, output format, downstream code assumptions) without
paying the ViT's training cost while still iterating on those issues.

**Alternatives considered:** training the ViT first, per the original Phase 4 plan —
not rejected outright, just deferred: still the intended production backbone, to be
trained once the rest of the pipeline is validated against the ResNet-AE's embeddings.

**Trade-off / known risk:** the ResNet-AE has no temporal-contrastive loss, so it won't
exercise `beast-backbones`'s contrastive-specific data requirements (temporal neighbor
sampling) — those remain unvalidated against our data until the ViT config is actually
trained. Downstream code (Phase 5 embedding storage, Phase 6 clustering) should avoid
hardcoding the 768-d ViT embedding size, since the ResNet-AE's `num_latents: 16` gives a
different dimensionality.

---

## Phase 3 frame selection: filter-then-candidate-restrict BEAST's own kmeans selection

**Date:** 2026-08-28
**Status:** decided, implemented

**Decision:** For `cuttle extract` (Phase 3), select BEAST training frames per video in
three steps — (1) remove blank/low-likelihood frames, (2) keep only survivors whose
immediate neighbors also survived, as the candidate set, (3) run a fork of BEAST's
`select_frame_idxs_kmeans` (motion-energy threshold → PCA → k-means)
restricted to that candidate set — rather than calling BEAST's function directly and
filtering its output afterward. `select_frame_idxs_kmeans`'s only subsetting knob is
`frame_range`, a contiguous fractional window (e.g. `[0.25, 0.75]`); it can't express an
arbitrary/non-contiguous allowed-frame set, so filtering *after* the fact wouldn't have
worked either — the high-motion-energy percentile and the k-means cluster centers
themselves need to be computed only over allowed frames, or a disallowed frame could still
end up as the nearest-frame-to-a-cluster-center pick. `select_frame_idxs_kmeans_restricted`
(`cuttle_patterns/preprocessing/extract.py`) is that fork: same algorithm, but the
percentile/PCA/k-means steps only ever see `candidate_idxs`. Everything else BEAST
offers — `compute_video_motion_energy`, `export_frames` — is reused unmodified; the one
private helper upstream (`_run_kmeans`) is not imported (its logic — `KMeans(...,
n_init='auto')` — is three lines, inlined directly with `random_state=0` instead of
upstream's global `np.random.seed(seed)` + unseeded `KMeans`).

**Why:** Keeps the diversity-selection algorithm identical to BEAST's own (same
motion-energy-driven PCA/k-means logic Phase 4 training will implicitly assume), while
still guaranteeing no blank or low-confidence-pose frame — and no frame whose exported
temporal-context neighbor would be blank/low-confidence — ever ends up in the training
set. Reusing `pose.interpolate_pose`'s existing `is_interpolated` return value (already
exactly "this frame's likelihood was too low") for the likelihood-filter mask avoided
writing new comparison logic for something already computed correctly in Phase 2b.

**Alternatives considered:** monkeypatching or subclassing BEAST's `extract_frames`/
`select_frame_idxs_kmeans` — rejected, more fragile than a straight fork given the
function's small size and that a private helper (`_run_kmeans`) would still need
reaching into; filtering `select_frame_idxs_kmeans`'s *output* against the allowed set —
rejected as described above, since it doesn't stop a disallowed frame from
being what a cluster center resolves to, only from being an initial high-motion-energy
candidate.

**Trade-off / known risk:** `--pose-dir` has no default and a video with no matching pose
CSV is skipped entirely (warned, not extracted with blank-only filtering) — a deliberate
asymmetry with `cuttle inscribe`/`cuttle overlay`'s optional-pose-with-PCA-fallback
design, since likelihood filtering is required here, not an optional refinement.
`frames_per_video`, the motion-energy percentile thresholds, and `resize_dims` (32,
hardcoded) are all carried over from BEAST's defaults/upstream logic unchanged and
untuned against real aligned videos — revisit once real data is available (see
[PHASES.md](PHASES.md) Phase 3 open questions).

---

## `scripts/` directory: working scripts not yet promoted to `cuttle_patterns/` + the CLI

**Date:** 2026-08-27
**Status:** decided, implemented

**Decision:** New top-level `scripts/` directory for code that's integral to the project
but not yet crystallized enough to live in `cuttle_patterns/` + the `cuttle` CLI (tested
library functions, `cmd_*.py` argparse wiring — see the CLI-structure entry below) — e.g.
`scripts/pose_plot_outliers.py`, which ranks candidate frames for pose-labeling by
cross-model prediction disagreement (see [pose_estimation.md](pose_estimation.md)).
Distinct from the existing `scratch/` (one-off, throwaway single-frame visualization,
e.g. `scratch/run_inscribe_v1.py` — see Phase 2a in [PHASES.md](PHASES.md)): `scripts/`
is for things meant to be run repeatedly and iterated on as real tooling, just without
the argparse/test/docstring overhead of a full `cuttle` subcommand yet.

**Why:** `scratch/` already had a settled meaning (throwaway single-frame experiments
predating the CLI) that doesn't fit ongoing active-learning/QC tooling meant to be rerun
as new model predictions land. A separate directory keeps that distinction legible
instead of overloading `scratch/`'s existing meaning.

**Alternatives considered:** folding this into `scratch/` — rejected to keep the
"throwaway" vs. "real but not yet crystallized" distinction visible; promoting straight
into `cuttle_patterns/` + a `cuttle` subcommand — rejected as premature before the
approach (e.g. the disagreement-score formula, output format) has been used for real.

**Trade-off / known risk:** no enforced graduation path yet — a script could linger in
`scripts/` indefinitely instead of being promoted once it stabilizes. Acceptable for now;
revisit if `scripts/` accumulates content that's clearly proven out.

---

## Raw filename scheme: `Day{n}_Tank{m}_Cuttle{k}_{role}`, string session/fish ids

**Date:** 2026-08-27
**Status:** decided, implemented

**Decision:** Parse raw video/blank-frames filenames as
`Day{day}_Tank{tank}_Cuttle{n}_{role}_crop.mp4` /
`Day{day}_Tank{tank}_Cuttle{n}_{role}_black_frames.txt` (`crop`/`Crop` both seen, matched
case-insensitively), rather than the original `session-{id}_cuttle-{id}.mp4`/`.txt`.
`session_id` is now the `Day{day}_Tank{tank}` prefix (e.g. `Day1_Tank2`) and `fish_id` is
the `Cuttle{n}_{role}` suffix (e.g. `Cuttle1_Resident`/`Cuttle2_Intruder`) — both plain
strings, not the small integers `ingest.py` previously parsed and cast with `int(...)`.
Since the video and blank-frames filenames now have different suffixes (`_crop`/`_black_
frames`, not just a different extension on an otherwise-identical stem), `ingest.py`
rebuilds the blank-frames path from the parsed `session_id`/`fish_id` rather than
`video_path.with_suffix('.txt')`.

**Why:** Collaborators changed their delivery naming scheme once more sessions started
landing (confirmed by inspecting the current contents of the raw data drive, which now
has 38 videos across Day1-4/Tank1-6 instead of the single originally-delivered
session/fish pair). Only `cuttle_patterns/ingest.py` (the one place that actually parses
session/fish ids out of a filename, via `FILENAME_PATTERN`) needed changing —
`preprocessing/align.py`, `overlay.py`, and the CLI commands built on top of them only
ever use `video_path.stem`/`.name` opaquely for output naming, so they're unaffected by
the scheme itself.

**Alternatives considered:** keeping `session_id`/`fish_id` as ints by extracting just the
numeric day/tank/cuttle-number components and dropping the tank/role text — rejected,
since the role (`Resident`/`Intruder`) isn't derivable from the cuttle number alone (see
`Day2_Tank5_Cuttle1_Intruder`/`Cuttle2_Resident`, where the usual `Cuttle1_Resident`/
`Cuttle2_Intruder` pairing is flipped), so it has to be captured, not discarded.

**Trade-off / known risk:** `session_id`/`fish_id` are now strings everywhere downstream
(manifest, any future code that groups/filters by them) rather than ints — no code beyond
`ingest.py` depended on the int type as of this change, but new code should not assume
either column is numeric.

---

## Rectangle-trajectory smoothing: median and Gaussian, mutually exclusive

**Date:** 2026-07-10
**Status:** decided, implemented

**Decision:** Smooth the final per-frame rectangle corner trajectory with either a
Gaussian filter (`align.smooth_corners_gaussian`, `--smoothing-sigma`, standard deviation
in frames, 2.0 if given with no value — **the default when neither flag is given**) or a
centered rolling median (`align.smooth_corners`, `--smoothing-window`, 9 frames if given
with no value; 1 disables smoothing), rather than fixing this upstream in mask/pose
geometry. The two CLI flags are symmetric (`nargs='?'` + `const` on both, so either can
be typed bare for its recommended value or with an explicit number) and mutually
exclusive — an argparse mutually-exclusive group on the CLI, a `ValueError` in
`align_video` if both are passed as non-None.

**Why:** QC on session-01/cuttle-01 (1:35-1:40) showed the rectangle jittering
frame-to-frame — well above baseline — driven by fin-beat oscillation rather than real
body movement (visually, the box shifted between frames where the body's pose was
essentially unchanged). Smoothing the final trajectory (not the pose keypoints, and not
the mask) is the narrowest fix: the jitter shows up in the sized rectangle regardless of
orientation source (PCA or pose-informed), so it needs to happen after both paths
converge — the same place `interpolate_corners` already runs. The median was implemented
first as the safer, outlier-robust default (rejects a one- or two-frame glitch outright).
The Gaussian was added afterward once the median's own output still looked "jumpy" on the
same clip: since the fin-beat noise is continuous/quasi-periodic rather than sparse
spikes, blending the whole window (Gaussian) tracks it more smoothly than snapping to one
observed value (median) at comparable strength — see the measurements in
[PHASES.md](PHASES.md) (Temporal smoothing). Once the Gaussian's results looked better on
this clip, the default (what runs when neither flag is given) was switched from the
median to the Gaussian at σ=2.0.

**Alternatives considered:** smoothing the pose keypoints before orientation/mask-cut —
rejected, since fin-driven jitter also affects the PCA-only path and the mask-based
sizing step, not just pose; picking one smoother instead of offering both — rejected,
since the median's outlier robustness and the Gaussian's smoother continuous tracking are
genuinely different trade-offs and it's cheap to expose both.

**Trade-off / known risk:** both the 9-frame median window and the 2.0 Gaussian sigma
were tuned by eye against the one noisy clip checked so far; a genuinely fast real
movement (e.g. an escape jet) could get smoothed more than intended by either — revisit
if that shows up elsewhere in the dataset. The Gaussian is less robust to a single
genuinely-bad frame than the median, since it blends the outlier in rather than
rejecting it.

---

## Phase 2b pose plumbing: real pose-CSV format, --pose-dir/--pose-path, 0.9 likelihood

**Date:** 2026-07-10
**Status:** decided, implemented

**Decision:** Consume the pose CSV a pose-estimation model actually writes — the standard
multi-header format (three header rows: scorer, bodyparts, coords; one data row per
frame) — rather than inventing a simplified `frame_idx, tail_x, tail_y, neck_x, neck_y`
schema as originally sketched in `docs/PHASES.md` Phase 2b. `pose.load_pose_predictions`
selects columns by the `bodyparts` level only (ignoring `scorer`, a model/run-specific
label), so it works regardless of which pose model produced the file. Interpolate
(flat-extrapolate at the edges, same convention as `align.interpolate_corners`) over any
frame where either keypoint's likelihood is below 0.9. `interpolate_pose` trusts the pose
CSV has exactly one row per video frame (holds for the one real file seen so far) instead
of taking an explicit frame count to reconcile against, which also lets `align_video` skip
opening the video up front just to learn its frame count. `cuttle inscribe`/`cuttle
overlay` gain `--pose-dir` (default
`results_dir/pose`, looked up per video as `{video_name}.csv`, mirroring the existing
`--output-dir` pattern) and `--pose-path` (single-video override, mirroring
`--video-path`); a video with no matching pose file falls back to the Phase 2a PCA path
with a printed message.

**Why:** Real predictions for session-01/cuttle-01 landed with per-frame likelihoods
around 0.998-0.999, confirming the format is the pose-estimation tool's native output
rather than something worth re-deriving. 0.9 is a reasonably strict cutoff given that
baseline — chosen without yet having seen a real low-confidence frame to calibrate
against, so it may need revisiting once more sessions' predictions land. The
`--pose-dir`/`--pose-path` split mirrors the `--output-dir`/`--video-path` pattern already
used by both commands, so batch runs auto-discover predictions per video while a single
video can still be pointed at an explicit file.

**Alternatives considered:** a separate `inscribe_rectangle_from_pose` /
`compute_corner_trajectory_from_pose` entry point mirroring the PCA-based ones —
rejected, since it would have duplicated the mask-recovery/rotation/seeding/growing/
corner-mapping logic that's identical in both paths; `tail`/`neck` as optional kwargs on
the existing functions keeps one implementation with a two-way branch instead.

**Trade-off / known risk:** the 0.9 threshold is currently unvalidated against any
actual low-confidence frame (all real predictions seen so far are >0.98); revisit once
more sessions' pose predictions land and some genuinely uncertain frames show up.
Trusting the pose CSV's row count against the video's frame count (rather than
reconciling an explicit count) means a mismatch would surface as an `IndexError` in
`compute_corner_trajectory` rather than a clear error message — acceptable since the one
real file pairing confirmed an exact match; revisit if that stops holding once more
sessions land.

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
**Status:** superseded — see "UI framework: Bokeh, programmatic Server" below.

**Decision:** Build the interactive embedding explorer (Phase 7) on Plotly. Whether that
means Dash, a lighter Plotly-based setup, or something else built around Plotly figures
is intentionally left open until we're actually building the UI.

**Why:** Plotly's hover/event model is a natural fit for "hover a dot, show the
corresponding frame image" — the core interaction we need. The specific app framework
choice matters less right now than getting the data pipeline (Phases 0-6) working.

---

## UI framework: Bokeh, programmatic Server, not the bare `bokeh serve` CLI

**Date:** 2026-08-31
**Status:** decided

**Decision:** Build the interactive embedding explorer (Phase 7, `cuttle serve`) on Bokeh
instead of Plotly, with the app launched via a programmatic `bokeh.server.server.Server`
(`cuttle_patterns/dashboard/launch.py`) rather than the `bokeh serve` CLI subprocess
pattern `cuttle train`/`cuttle predict` use for BEAST's own CLI.

**Why:** Bokeh's WebGL scatter (`figure(output_backend='webgl')`) comfortably handles the
target ~200-300k points; a bare Canvas-backed Plotly/Bokeh scatter would not. A
programmatic `Server` lets `launch.py` register an extra static-file route
(`/images/...` → `results_dir/beast_frames/`) that a Bokeh `HoverTool` HTML tooltip can
reference directly (`<img src="/images/@image_relpath">`) — the browser requests and
caches each image by URL on hover, no per-hover server round trip. The `bokeh serve` CLI
doesn't expose a way to add that route, which is why this deviates from the
subprocess-wrapper pattern used elsewhere; `cuttle serve` still calls `run_server`
directly rather than shelling out, since it wraps our own code, not an external CLI.

**Trade-off:** switching the color-by attribute uses a plain Python `on_change` callback
that recomputes a hex-color column and pushes it to the `ColumnDataSource` over the
websocket (`source.patch`), rather than precomputing one color column per attribute
server-side and swapping between them with a client-side `CustomJS` callback. The latter
would make color switching instant with zero data resent; the former is simpler code at
the cost of a brief (sub-second to ~1s at 300k rows) lag per switch. Accepted for now —
revisit if that lag is annoying in practice.

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
