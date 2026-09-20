# Embedding eval harness: plan

## Purpose

This harness replaces the current loop of finding confounds by eyeballing cluster grids
after each training run. It produces a quantitative scoreboard that compares any
embedding (existing AE/MSPS-VAE checkpoints, frozen pretrained backbones, future
fine-tunes) on equal footing. It asks two questions. How well does the embedding
organize by skin pattern? And how much does it organize by video/individual identity,
the one nuisance variable we can currently measure directly? The qualitative inspection
that found both confounds so far (identity, and rectangle-edge leakage) stays in, but
becomes standardized and comparable across runs. Leakage, lighting, and crop-geometry
nuisance metrics are deliberately out of scope for now — see "Out of scope for now" — so
this v1 only scores pattern agreement vs. identity organization; extend it once that
metadata is available.

The harness is embedding-agnostic. Anything that maps a frame to a vector plugs in
through one interface, and all metrics run on cached embeddings.

## Design principles

**Classifier labels are a sanity floor, not the objective.** The collaborator's 6-class
classifier is the best available proxy for pattern type, but maximizing agreement with it
would reward distilling the classifier, which is the failure mode the project is trying
to avoid. Report classifier agreement, but read it as "high enough to show the embedding
captures coarse pattern," not as the number to push up. The primary signal is *low
identity organization at acceptable pattern agreement*.

**Distinguish "decodable" from "dominant."** Almost any embedding will let a linear probe
decode video identity. That alone isn't the problem. The problem is when identity
accounts for a large share of the embedding's variance, because that's what k-means and
UMAP respond to. The harness reports both kinds of metric and labels them clearly.

**Condition on pattern when measuring identity.** Pattern type and video are correlated:
some sessions are mostly one pattern. An embedding that perfectly captures pattern will
therefore look somewhat identity-organized. Identity metrics should be computed *within
classifier class* wherever possible, so they measure identity structure beyond what
pattern explains.

**Split by video everywhere.** All probes, k-NN evaluations, and any train/test split use
video-grouped splits (`GroupKFold` on `video_name`), so no metric can be satisfied by
recognizing the session.

## Eval frame set

We already have this — it's the frame set `cuttle extract` produced (Phase 3), not
something to build from scratch. Reuse `results_dir/beast_frames/{video_name}/` (the
frames themselves) and `results_dir/manifests/extract.parquet` (`session_id`, `fish_id`,
`frame_idx`, `image_path`, one row per selected anchor frame) as the eval frame set
directly, no new manifest-building step. It's already a diverse, deduplicated,
per-video-capped sample (BEAST's motion-energy/PCA/k-means selection, up to
`frames_per_video` per video), so no new stride/sampling logic is needed either. A
held-out-video split and a frozen/hashed manifest are unnecessary on top of this — each
data/pipeline iteration already lives in its own `results_dir` folder, so there's nothing
to snapshot or hash separately.

## Per-frame metadata

**Identity:** `video_name`, plus `day`/`tank`/`role` (already parsed by
`cuttle_patterns/latents.py:parse_video_name`). Individual ID mapping — see "To verify
in the repo before starting" below.

**Pattern proxy:** the classifier's predicted class, full softmax vector, and
max-probability, already computed for every frame under `beast_frames/` and stored at
`results_dir/classifications/{classifier_name}.parquet` (`../../scripts/classify_skin_pattern.py`),
keyed by `(video_name, frame_number)` — join, don't recompute. Low max-probability frames
(below ~0.5, tune after looking at the distribution) are the "ambiguous" subset used in
later metrics.

**Time:** `frame_number` within video (already in every manifest/latents path). A real
timestamp can be derived from `manifests/ingest.parquet`'s per-video `fps` if needed
later; not required for v1.

Leakage, crop geometry, and lighting/illumination metadata are not currently available in
this pipeline and are dropped from this plan — see "Out of scope for now."

## Embedders

Two tiers, because most of the embeddings this harness needs to score already exist:

**Tier A — already produced by `beast train`/`cuttle predict`.** The plain AE, MSPS-VAE,
and masked MSPS-VAE checkpoints already write per-frame latents under
`beast_models/{model_name}/image_predictions/{predictions_name}/latents/`. The harness
loads these with the existing `../../cuttle_patterns/latents.py` (`load_latents` +
`split_latent_spaces`, which already exposes MSPS-VAE's `z_u`/`z_b`/full `z`) and joins
them to the eval manifest by `(video_name, frame_number)`. No new extraction code needed
for this tier — for MSPS-VAE/masked MSPS-VAE, `z_b` serves as a sanity check: it *should*
score high on identity metrics; if it doesn't, something is wrong with the harness.

**Tier B — needs new extraction code**, since nothing in the repo runs these yet:

```python
class Embedder:
    id: str                      # e.g. "dinov2_vitb14_meanpatch"
    def preprocess(self, frames: np.ndarray) -> torch.Tensor: ...
    def embed(self, batch: torch.Tensor) -> np.ndarray: ...  # (B, D)
```

Cached as `embeddings/{embedder_id}.npy`, row-aligned with `extract.parquet`, with a
sidecar JSON recording checkpoint path, preprocessing, and pooling. Start with:
**calibration baselines** (downsampled raw pixels, random-init ResNet-18) and **one**
pretrained backbone (DINOv2 ViT-B/14, mean-pooled patch tokens) for an external reference
point. I-JEPA and VGG Gram-matrix texture features are real candidates for later — cut
from v1 to keep the embedder list small to start.

A **post-hoc per-video-centered** variant (subtract each video's mean embedding) is a
cheap identity-removal baseline worth adding once the core scoreboard works — v2, not v1.

## Clustering conventions

L2-normalize, then PCA to 64 dimensions (or native dimension if smaller), with no
whitening, and record variance retained. Start with a single **k=16** (matching the
existing `kmeans_k16` convention already used in `iter-1.1_msps-vae_d16` and
`../../scripts/plot_cluster_frames.py`), 5 seeds, mean ± std across seeds. Add k=8/32 later if
k=16 alone doesn't give a clear enough read.

## Metrics

Start with the smallest set that answers the two questions in "Purpose." Everything else
is deferred to v2.

| Metric | What it measures | Want |
|---|---|---|
| AMI(clusters, classifier class) | Coarse pattern agreement | High enough |
| AMI(clusters, video), within class | Identity structure beyond pattern | Low |
| Effective # videos per cluster (exp of entropy) | Cluster identity purity | High |
| Cross-video k-NN class accuracy (k=20, neighbors from other videos only) | Pattern retrieval across sessions | High |
| Linear probe accuracy, classifier class (video-grouped CV) | Pattern decodability | High |

Use AMI rather than NMI, because NMI is biased upward for labelings with many categories,
and `video_name` has many. For the "within class" metric, compute it separately in each
classifier class and report a frame-weighted average.

**Deferred to v2** (real candidates, just not needed to answer the current question):
same-video neighbor rate vs. class-matched expectation, variance-share-by-identity,
ambiguous-subset cross-video neighbor consistency, linear-probe R² for any nuisance
variable once one beyond identity is available.

## Qualitative outputs

Extend `../../scripts/plot_cluster_frames.py` (already exists) rather than replacing it. For
each cluster (k=16, first seed), produce a frame grid that samples **across videos** (at
most 2 frames per video per grid), with each tile annotated by video and classifier
class. This stops a cluster from looking coherent just because it's all one session.

Add **fixed-query neighbor grids**: about 30 query frames chosen once and saved in the
manifest, including around 10 from the ambiguous subset. Each query is shown with its 12
nearest cross-video neighbors, for every embedder. Because the queries never change,
these grids can be compared side by side across backbones and across training
checkpoints.

UMAP plots colored by class and by video are useful but secondary — generate them, don't
treat them as evidence on their own.

## Code layout

Lives inside the `cuttle_patterns` package, not as a separate top-level directory, so
it's covered by the repo's normal install/lint/test setup with no extra wiring:

```
cuttle-patterns/
  cuttle_patterns/
    eval/
      build_manifest.py        # classifier-label join on top of extract.parquet
      load_embeddings.py        # Tier A: load/align existing beast_models/ latents
      clustering.py              # shared L2-normalize + PCA + k-means preprocessing
      metrics.py                  # the five core metrics, pure functions on arrays
      report.py                   # score_embedder / run_report / write_report
      run_core.py                 # CLI entry point (not wired into `cuttle`)
  tests/
    eval/                        # mirrors cuttle_patterns/eval/, like every other module
```

`python -m cuttle_patterns.eval.run_core --classifier-name ... --model-name ...` builds
the manifest, scores every requested model, and writes `{results_dir}/eval/` (see
`cuttle_patterns.paths.EVAL_RELPATH`) containing `metrics.json` and `scoreboard.md` —
see [`../../cuttle_patterns/eval/README.md`](../../cuttle_patterns/eval/README.md)
for the full walkthrough. Tier-B extraction (`embedders/`, `extract.py`) and qualitative
outputs (`qualitative.py`) aren't built yet — deferred, per "Deferred to v2" / "Out of
scope for now" above. Metrics are unit-tested on synthetic embeddings, including a case
where embeddings are one-hot video IDs plus noise, confirming the identity metric fires
and the pattern metric doesn't (`../../tests/eval/test_metrics.py`).

## Build order

1. Build the manifest: join classifier labels onto the existing `extract.parquet`.
2. Implement the five metrics above and the report on the existing MSPS-VAE `z_u`/`z_b`
   (Tier A, no new extraction needed), and verify `z_b` looks identity-heavy and `z_u`
   less so. This validates the harness against known behavior.
3. Add the masked MSPS-VAE checkpoint once trained, and the calibration baselines + one
   pretrained embedder (Tier B).
4. Add qualitative outputs (cluster grids, fixed-query grids).

Step 2 matters most: the harness is only trustworthy once it reproduces the identity
finding that was originally found by eye.

## Out of scope for now

Leakage, crop-geometry, and lighting/illumination nuisance metrics and the counterfactual
sensitivity tests built around them (inset-crop test, photometric perturbation test) —
this pipeline doesn't currently have that per-frame metadata; revisit once/if it does.
Also out of scope: any training or fine-tuning, training-set curation, hyperparameter
search over clustering algorithms other than k-means, and the I-JEPA / VGG Gram-matrix
embedders (candidates for later, not needed to validate the harness itself).
