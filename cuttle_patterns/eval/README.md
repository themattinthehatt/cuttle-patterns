# Embedding eval harness

Scores any per-frame embedding on two questions: how well does it organize by skin
pattern, and how much does it organize by video/individual identity instead? Replaces
eyeballing cluster grids after each training run with a quantitative scoreboard. Full
design rationale (why these metrics, what's deliberately out of scope, what a v2 would
add) lives in [`docs/eval_plan.md`](../../docs/eval_plan.md) — this file is just the
how-to-run.

**Not wired into the `cuttle` CLI.** Everything here is called directly, either from
`cuttle_patterns/eval/run_core.py` or from a Python shell — see below.

## Current scope (v1 / "core")

Only what's needed to answer the two questions above, for embeddings already produced
by the existing `cuttle train`/`cuttle predict` pipeline:

- `build_manifest.py` — joins classifier predictions onto the existing
  `cuttle extract` frame set.
- `load_embeddings.py` — loads a trained model's latents (optionally splitting an
  MSPS-VAE checkpoint into `z_u`/`z_b`/`z_all`) and aligns them to the manifest.
- `clustering.py` — the shared L2-normalize + PCA + k-means preprocessing every
  embedder is scored through.
- `metrics.py` — the five core metrics (see table below).
- `report.py` — runs the metrics for a list of embedders and writes the scoreboard.
- `run_core.py` — a ready-made script wiring the above up against real
  `results_dir` models.

**Deliberately not built yet:** scoring a pretrained backbone like DINOv2 (would need
new extraction code, not just loading existing latents), and the qualitative outputs
(cluster grids, fixed-query neighbor grids). Both are real v2 work, not forgotten —
see `docs/eval_plan.md`'s "Deferred to v2" / "Out of scope for now" notes.

## Metrics

| Metric | What it measures | Want |
|---|---|---|
| `ami_cluster_class_mean/std` | AMI(clusters, classifier class) — coarse pattern agreement | High enough |
| `ami_cluster_video_within_class_mean/std` | AMI(clusters, video), computed within classifier class — identity structure beyond pattern | Low |
| `effective_videos_per_cluster_mean/std` | exp(entropy) of each cluster's video distribution, frame-weighted mean | High |
| `cross_video_knn_accuracy` | majority-vote k-NN class accuracy, neighbors restricted to other videos | High |
| `linear_probe_accuracy` | video-grouped CV linear-probe accuracy for classifier class | High |

`*_mean/std` metrics are aggregated over 5 k-means seeds at k=16 (`prepare_for_clustering`
L2-normalizes then PCA-reduces first — see `clustering.py`). None of these are meant to
be maximized in isolation — read them together, per `docs/eval_plan.md`'s design
principles. What each one actually is, and how to read it:

**`ami_cluster_class_mean/std`** — adjusted mutual information (AMI) between the
k-means cluster assignment and the classifier's predicted pattern label. 0 means no
better than random, 1 means perfect agreement (it can dip slightly negative for
near-random labelings). This is a **sanity floor, not the objective**: the classifier's
6 hand-labeled classes are a coarse, imperfect proxy for "abstract pattern," and scoring
*too* well here is itself a yellow flag — it suggests the embedding has distilled the
classifier rather than found structure beyond it, including the rare/ambiguous patterns
the classifier can't label in the first place. A low score means clusters don't even
track the coarse classes we already have labels for, which is a real problem; a middling
score is expected; treat a very high score as something to double-check, not celebrate.

**`ami_cluster_video_within_class_mean/std`** — the same AMI, but between clusters and
`video_name`, computed *separately within each classifier class* and then
frame-weighted-averaged — not raw AMI(cluster, video) over the whole dataset. The
within-class conditioning matters because pattern and video are correlated (some
sessions are mostly one pattern), so an embedding that perfectly captured pattern would
look somewhat identity-organized even with zero actual identity confound; conditioning
on class isolates identity structure *beyond* what pattern already explains. This is the
metric that most directly targets the project's original failure mode
(video-identity-dominated clustering, see `docs/latent_space_confounds.md`) — **want it
low**. A subspace shaped by a triplet loss on video identity (e.g. MSPS-VAE's `z_b`)
scoring high here is expected and is in fact the harness's own sanity check (see below);
a pattern subspace (`z_u`) scoring this high means the identity confound isn't actually
fixed.

**`effective_videos_per_cluster_mean/std`** — for each cluster,
`exp(entropy(video distribution within that cluster))`: 1.0 if every frame in a cluster
comes from the same video, up to the true number of distinct videos in the cluster if
frames are spread evenly across them. Frame-weighted mean across clusters. A second,
more intuitive view of the same identity question as the AMI-within-class metric above
— "how many videos does a typical cluster actually mix together," rather than an
information-theoretic score — useful for eyeballing "is this basically one video per
cluster" at a glance. **Want it high.**

**`cross_video_knn_accuracy`** — for each frame, look at its nearest neighbors in
embedding space, restricted to a *different* video (see `metrics.py`'s
`cross_video_knn_accuracy` docstring for the exact search-window/fallback behavior),
take a majority vote of their classifier-predicted class, and score whether that vote
matches the frame's own predicted class. This tests whether pattern-type is actually
retrievable *across sessions* — the practically useful thing an embedding is for
(finding other frames with the same pattern, not just other frames of the same fish).
**Want it high.** Unlike the AMI metrics it doesn't depend on k-means or a choice of k
at all, so it's a useful independent check when the clustering-based metrics look
surprising.

**`linear_probe_accuracy`** — fit a multinomial logistic regression to predict
classifier class from the raw embedding, under video-grouped cross-validation
(`GroupKFold` on `video_name`, so no fold can pass by memorizing a session), and report
accuracy on the held-out folds. A standard, model-based measure of how linearly
decodable pattern-type is, complementary to the clustering-based metrics (which depend
on k-means finding the right structure unsupervised). **Want it high** — but a high
probe accuracy only shows pattern is *linearly decodable*, not that it *dominates* the
embedding's geometry the way k-means/UMAP would see it (per the eval plan's "distinguish
decodable from dominant" principle — the same reasoning that motivates conditioning the
identity metrics on class rather than trusting decodability alone). Read it alongside
`ami_cluster_video_within_class`/`effective_videos_per_cluster`, not as a substitute for
them.

## Running it

### Quick start

```bash
python -m cuttle_patterns.eval.run_core \
  --classifier-name iter-1.1_classifier_d512 \
  --model-name iter-1.1_resnet-18_d16:ae \
  --model-name iter-1.1_msps-vae_d16:msps_vae
```

`cuttle_patterns.eval` is part of the normal package install (`pip install -e ".[dev]"`),
so this resolves from anywhere, not just the repo root.

- `--classifier-name` is the stem of `results_dir/classifications/{name}.parquet`
  (`scripts/classify_skin_pattern.py`'s output) to join as the pattern-type proxy.
- `--model-name {model_name}:{kind}` is repeatable; `kind` is `ae` for a
  single-subspace model (one scoreboard row, id = `model_name`) or `msps_vae` to
  expand into three rows (`{model_name}_z_u`, `{model_name}_z_b`,
  `{model_name}_z_all`). `model_name` is the name passed to `cuttle train
  --model-name`/`cuttle predict --model-name`.
- `results_dir` is read from `~/.cuttle-patterns/config.yaml`, same as every `cuttle`
  subcommand.
- Writes `{results_dir}/eval/scoreboard.md` and `{results_dir}/eval/metrics.json`
  (see `cuttle_patterns.paths.EVAL_RELPATH`) by default — same `results_dir`-relative
  convention as every other pipeline stage's output (`beast_models/`,
  `classifications/`, ...). Pass `--out-dir` to write somewhere else instead, and
  prints the scoreboard to stdout either way.
- **Merges, doesn't overwrite:** rows are keyed by `embedder` id. If `metrics.json`
  already exists at the output dir, a rerun keeps every existing row except ones whose
  `embedder` id is in this run's `--model-name` list, which get replaced. So scoring a
  new model doesn't drop the others — pass just the new `--model-name` to add it, no
  need to re-list every embedder you've already scored.

A real run against two checkpoints (a ResNet-18 AE and an MSPS-VAE) takes a couple of
minutes — most of the cost is the linear-probe/k-NN metrics running once per embedder
over the full eval frame set (currently ~32k frames).

### From a Python shell

For anything `run_core.py`'s CLI doesn't cover (a single metric, a different k, a
model not produced by `beast train`), call the functions directly:

```python
from cuttle_patterns.eval.build_manifest import build_eval_manifest
from cuttle_patterns.eval.load_embeddings import EmbedderSpec, load_embedder_matrix
from cuttle_patterns.eval.report import score_embedder
from cuttle_patterns.config import load_config

config = load_config()
manifest = build_eval_manifest(config.results_dir, 'iter-1.1_classifier_d512')

spec = EmbedderSpec(
    id='msps_vae_z_u',
    model_dir=config.results_dir / 'beast_models' / 'iter-1.1_msps-vae_d16',
    subspace='unsupervised',  # or 'background' / 'all'
)
result = score_embedder(spec, manifest)
```

## Sanity check

An MSPS-VAE checkpoint's `z_b` subspace should score high on
`ami_cluster_video_within_class_mean` and low on `effective_videos_per_cluster_mean`
— it's shaped by a triplet loss keyed on video identity, so if it doesn't look
identity-heavy here, something is wrong with the harness, not the model. Confirmed
against `iter-1.1_msps-vae_d16`: `z_b` scored 0.81 / 2.6 there vs. `z_u`'s 0.19 / 17.6
on the same two metrics.

## Tests

`tests/eval/` mirrors `cuttle_patterns/eval/`, same convention as every other module in
the repo. Run with:

```bash
pytest tests/eval
```

They also run as part of the normal repo-wide `pytest` (no separate CI wiring needed
since no test here needs a GPU or a new dependency).
