# Frame selection for BEAST training (`cuttle extract-*`)

Picks the frames used as `beast_frames`: a diverse pool that is not dominated by the most common
skin pattern. This file is the working guide: how the pipeline fits together, how to run it,
what to check, and what has and has not been validated. Command flags are in the top-level
`README.md` (and `--help`); the rationale is the "Class-quota frame selection" entry in
`docs/DECISIONS.md`; the mechanism summary is Phase 3 in `docs/PHASES.md`; the exploratory
record is `scratch/extraction/`.

## Why this exists

The old `cuttle extract` kept the top 50% of frames by motion energy, ran PCA + flat k-means
on 32x32 pixels, and took the frame nearest each centroid. Dense states (dark aggression)
took most of the cluster budget. Prototypes showed that k-means in DINOv3 space alone also
does not rebalance classes: the dark-aggression share stayed ~62% of every embedding-based
pool, because class imbalance is mostly *between* videos and the DINOv3 CLS geometry does not
separate the classifier's patterns. So the pool is built from classifier labels first
(class quotas) and diversity second (k-means within each class).

## Stages

| # | command | module | reads | writes (under `results_dir`) |
|---|---|---|---|---|
| 1 | `extract-eligibility` | `eligibility.py` | rectangle videos, pose CSVs, blank-frame lists | `extraction/eligibility/{video}.parquet` |
| 2 | `extract-classify` | `classify.py` | **crop** videos, classifier dir | `classifications/{classifier dir name}/{video}.parquet` |
| 3 | `extract-dino-cls` | `dino_cls.py` | rectangle videos | `embeddings/dinov3_{arch}/{video}.npy` |
| 4 | `extract-kmeans-video` | `kmeans.py` | 1-3 | `extraction/kmeans_video/{name}/{video}.npz`, `config.yaml` |
| 5 | `extract-kmeans-pooled` | `kmeans.py` | 4 | `extraction/selections/{name}/manifest.parquet`, `summary.json` |
| 6 | `extract-report` | `report.py` | 1-5, rectangle videos | `report.md`, `report.json`, sheets next to the manifest |
| 7 | `extract-export` | `export.py` | 5, rectangle videos | `beast_frames/{video}/img{idx:08d}.png`, `manifests/extract.parquet` |

Stages 1-3 are independent of each other and can run in parallel (use `--shard/--n-shards`
to split videos across processes). Stages 4-7 run in order. `paths.py` has the exact path
constants. Each command prints a `next:` hint.

### Inputs: crops vs rectangles (easy to get wrong)

- **Rectangle videos** (`results/rectangles/*_Crop.mp4`, aligned 200x100): eligibility
  (motion energy, black fraction), DINOv3 features, the report's sheets, and the export.
- **Crop videos** (`data_dir/**/*_Crop.mp4`, 512x512): the skin-pattern classifier, and the
  pose CSVs' coordinate frame (the keypoint-jump threshold is in crop pixels).
- Only `*_Crop.mp4` rectangle videos count; the `.csv` and `_overlay.mp4` files next to them are
  ignored. Early prototype results were computed on crops by mistake and are superseded.
- Row `i` of every per-video table/array is frame `i` of that video, so eligibility,
  classifications and embeddings line up by index (a length mismatch raises).

## Eligibility (stage 1)

A frame is eligible only if all of these hold:

1. not blank (`*_black_frames.txt`);
2. neck and tail keypoint likelihood >= 0.9;
3. the inscribed rectangle's long edge is >= 50% of the neck-tail distance;
4. no keypoint jump: displacement to either neighbor <= 25 px (checked only where both
   endpoints are high-confidence; uses the *raw* predictions, before interpolation);
5. motion energy (rectangle video, 32x32) <= the video's 99th percentile among frames passing
   1-4; frame 0 is never eligible (no previous frame);
6. < 5% pure-black pixels (max channel == 0) in the rectangle frame.

Each criterion has its own boolean column (True = fails), so rejected frames can be
inspected. Thresholds were picked from contact sheets (`scratch/extraction/*_contact_sheets.py`):
the top ~1% of rectangle-video motion energy is mostly alignment glitches and blur, and the
black threshold is deliberately aggressive. There is no neighbor/context-frame requirement and
no context-frame export. Result on the full set: 5.20M of 6.34M frames eligible; per-video
median 38.5k, min 11.3k.

## Selection (stages 4-5)

1. **Quotas.** Per class: `quota ∝ count^0.5` (square-root; `--quota-power`), scaled so the
   total is `--budget`, water-filled so no class exceeds its eligible count. Classes come
   from the classifier's pattern head only (Dark aggression, Leopard, Pale aggression, White,
   Yellow-brown). No confidence floor (low-confidence frames are often the interesting
   ones) and no per-video cap.
2. **Stage 1, per (video, class).** Spherical k-means on the DINOv3 CLS features of the
   video's eligible frames of that class, with ~`--fine-mult` (3) times the video's share of
   the class quota as clusters. Writes centroids plus members.
3. **Stage 2, per class, pooled over videos.** k-means on the pooled centroids into
   `quota / --picks-per-group` (10) groups; equal draws per group (water-filled), and for each
   chosen centroid one **random** member frame. Random member, not nearest or farthest: Vo et
   al. 2024 and the mighty-mouse experiments find random == nearest and farthest is much worse.
4. Day 1 (`--exclude-days`, default 1) is skipped throughout: the fogged tank was dropped in
   the photometry decision (`docs/DECISIONS.md`).

Selection is deterministic for a given seed: the centroid update uses fixed-order segment sums,
because GPU `index_add_` was nondeterministic.

Expected class mix of eligible frames (excl. Day 1): Leopard 55%, Dark aggression 36%, Pale 5%,
Yellow-brown 2%, White 1%. At a 132k budget the quotas are about Leopard 53.8k, Dark 43.3k,
Pale 16.6k, Yellow-brown 10.4k, White 7.9k. Per-video allocation ends up ~380-2,070 frames
(median ~950); White is concentrated in a few videos (top 5 ≈ 44% of its quota).

## Running it

```bash
cuttle extract-eligibility --pose-dir <pose dir>            # ~2 min/video
cuttle extract-classify --model-dir <classifier dir>        # skip if classifications exist
cuttle extract-dino-cls                                     # skip if embeddings exist
cuttle extract-kmeans-video --name sqrt-132k \
    --classifier-name 2026-11-07_classifier_model --budget 132000
cuttle extract-kmeans-pooled --name sqrt-132k
cuttle extract-report --name sqrt-132k                      # read this before exporting
cuttle extract-export --name sqrt-132k
```

All stages skip outputs that already exist (resumable) and write atomically; `--overwrite`
redoes them. For long jobs run detached with a log, per `CLAUDE.md`. Stages 1-3 (the long
ones) also write `progress_shard{i}of{n}.json` (done/total, current video) in their output
directory. One `--name` is one variant: to try a different budget or
setting, use a new `--name` rather than overwriting.

## Gotchas

- **Settings are pinned per name.** `extract-kmeans-video` records its settings in
  `config.yaml` and raises if you rerun with different ones (budget, powers, classifier,
  embedding) unless `--overwrite`. Use a new `--name` instead when comparing variants.
- **`--overwrite` and shared inputs.** Never run `extract-classify` or `extract-dino-cls` with
  `--overwrite` against the real `classifications/` or `embeddings/` directories unless you
  mean to recompute them. When testing, use a temporary `--results-dir`; do not symlink to the
  real directories and write through them.
- **Read-only data.** `raw_data`, `raw_data_samples` and existing `results/.iter-*` are
  read-only. Everything here writes to new directories.
- **Missing inputs.** `extract-kmeans-video` errors if a non-excluded video lacks its
  eligibility table, classifications or embeddings; `--skip-missing` skips those videos
  (check the count it reports before continuing).
- **Resuming DINOv3.** `extract-dino-cls` writes the same format as the old
  `scripts/dinov3_video_cls.py` (float16, `(n_frames, 384)`), so partial output from either
  can be resumed by the command.
- **Classifier version.** Classifications live under the classifier directory's name; pass
  the same name to `extract-kmeans-video --classifier-name`. A new classifier means a new
  directory and a new selection `--name`.
- **Embedding drift.** DINOv3 features depend on the pixels. If the rectangles are regenerated
  (different alignment or photometry), recompute stages 1 and 3 and everything after.
- **Videos with < 1k eligible frames** log a warning and keep what they have; none exist in
  the current data (min 11.3k).

## Checking a selection (`extract-report`)

Looks at: per-video allocation, class mix vs. eligible, within-class coverage (nearest-selected
cosine distance, mean/p95/p99.9) and densest/rarest-state ratios against a random baseline of
the same size, mean classifier confidence, temporal redundancy, and contact sheets. Before
exporting, look at:

- [ ] class mix equals the intended quotas (`summary.json`; they are met exactly, so a
  mismatch means a bug);
- [ ] per-video allocation: every video contributes and none is far outside the expected range
  (first run: 450-2,048, median 967);
- [ ] contact sheets for the **isolated picks** of each class, especially White and
  Yellow-brown: embeddings cannot tell junk from a rare pattern, so this is where blur,
  black wedges and warped rectangles would show up;
- [ ] coverage (p99.9) no worse than the random baseline and the densest-state ratio below 1;
- [ ] temporal redundancy (`near10`, share of picks with another pick from the same video
  within 10 frames) no higher than the random baseline. It is high for the rare classes in
  both columns, so only the selected-vs-random difference means anything.

Read the isolated sheets as a worst case, not a junk rate: they show the least typical frames,
so they say where junk hides, not how much of the pool it is. To estimate a junk rate, look at
the `*_random.jpg` sheets.

### First run (`sqrt-132k`: 132 videos, budget 132,000, defaults, Day 1 excluded)

- Quotas met exactly; per video 450-2,048 (median 967), all 132 videos contribute.
- Coverage p99.9 is better than random in every class but only slightly (Dark 0.255 vs 0.275,
  White 0.251 vs 0.283, Yellow-brown 0.299 vs 0.311); the densest state is thinned to 0.73-0.93
  of random's share.
- The rare-state ratio beats random only for Dark (1.22 vs 0.99); the other classes are at
  random (0.98-1.07). Temporal redundancy is about the same as random.
- Mean confidence of the picks matches the eligible frames (slightly more low-confidence Dark
  frames: 0.12 vs 0.10). The class mix is set by the quotas, and the diversity gain within each
  class is modest, as in the 8-video prototype.
- `White_random.jpg` is clean white pattern. `White_isolated.jpg` is mostly junk: motion blur,
  out-of-focus close-ups, fish edge, a few jagged black wedges from rectangle alignment
  (confidence mostly 0.3-0.5). `Yellow-brown_isolated.jpg` is mixed: valid orange-brown
  patterns plus close-ups of fins and eyes. `Pale_aggression_isolated.jpg` is mostly
  legitimate varied texture. Dark and Leopard isolated sheets were not reviewed.
- Decision: export as is and compare by training BEAST on the pool. Not changed because of
  the report: the confidence rule (no floor, by choice). If the junk matters after training,
  the candidates are a stricter motion-energy gate for blur and the black-fraction gate for
  wedges; estimate the junk rate from the random sheets first.

## What is not validated

- `--picks-per-group` (10), `--fine-mult` (3) and `--quota-power` (0.5) are defaults chosen on
  8 videos, not tuned.
- Nobody has trained BEAST on the pool and compared it with the old `beast_frames`.
- The class quotas are only as good as the classifier; pale aggression has the lowest mean
  confidence (0.65) and 10.7% of frames are below 0.6.
- Embedding input (full aligned frame vs. masked body) was not explored.

## Prototype findings behind the design

On 8 videos (full tables in `scratch/extraction/README.md`):

- Within-video k-means in DINOv3 space (flat or two-level) raised coverage and the rare-state
  share over random picks, but the dark-aggression share stayed ~62% (eligible 62.1%); the old
  recipe was worse, ~69%.
- Class imbalance is mostly between videos (per-video dark share ranges 0.03-0.95), which
  per-video selection cannot fix.
- A frame's nearest neighbor in embedding space (excluding ±100 frames) is in the same video
  96% of the time, so video identity dominates raw DINOv3 geometry; this is why selection is
  per-video first and cross-video only over centroids.
