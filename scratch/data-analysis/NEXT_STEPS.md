# Next steps: data-analysis session backlog

Things discussed and deliberately tabled during the exploratory session that produced
the scripts in this directory. Results referenced here are on the iter-1.1 data (32
videos, 12 individuals R1-R6/I1-I6, 3 sessions per individual), now archived under
`{results_dir}/.iter-1.1`. A new batch is incoming (target: 12 sessions per individual).

## 1. Clustering algorithms (replacing / supplementing k-means)

**Why k-means is limiting:**

- Its objective favors clusters of similar size and spread. It splits the large
  black-streaky mass and merges rare patterns into their nearest neighbor.
- It is run on raw 500-2000 dimensional embeddings. `cuttle cluster` does no reduction;
  only the eval harness applies PCA-64 first. Distances in that regime are dominated by
  many small, noisy directions.
- It has no hierarchy, and k=16 and k=32 runs are not nested.

Whatever the algorithm, first reduce to about 30-64 PCA dimensions.

**Candidates, in order of preference:**

1. **Leiden community detection on a k-NN graph** (PCA space, k ≈ 15-30). This is the
   single-cell RNA-seq standard for ~100k points, a continuum of states, and populations
   of very different sizes.
   - Cluster size is unconstrained; a tight group of rare-pattern frames becomes its own
     community.
   - The resolution parameter gives a multi-resolution view (not a strict tree).
   - PAGA draws a connectivity graph between clusters, which may show which patterns
     transition into which.
   - Packages: `leidenalg` + `igraph`, or `scanpy`.
2. **Over-cluster, then merge hierarchically.** Run k-means with k ≈ 200-500 to get
   micro-clusters, then Ward or average-linkage agglomerative clustering on their
   centroids.
   - Direct agglomerative clustering on 94k points needs about 35 GB for the distance
     matrix; two stages keeps it cheap.
   - The result is a true dendrogram you can cut at any level. Rare patterns keep their
     own micro-clusters until high in the tree, and it is easy to show in the GUI.
3. **HDBSCAN on a low-dimensional UMAP** (about 10 dimensions, min_dist=0).
   - Handles clusters of varying density and size. It has a built-in hierarchy (the
     condensed tree), and `cluster_selection_method='leaf'` gives fine-grained
     clusters.
   - Downsides: it labels a large fraction of points as noise (often 20-50% on
     continuum-like data), results depend on `min_cluster_size`, and it inherits UMAP's
     distortions.
   - Packages: `sklearn.cluster.HDBSCAN`, or `hdbscan` for tree plots and soft
     membership.
4. **Gaussian mixture model** (full covariance) on PCA dimensions. Clusters can differ
   in size and shape, and `BayesianGaussianMixture` switches off unneeded components.
   It is unstable much above 30-50 dimensions and has no hierarchy.

**Plan for any of them:**

- **Near-duplicate frames will pull graph methods toward clustering by video.** A
  frame's nearest neighbors are mostly frames from a few seconds earlier or later in
  the same video. Fixes to test:
  - subsample frames in time before building the graph;
  - exclude same-video neighbors within a time window;
  - require some neighbors to come from other videos, as the harness's cross-video
    k-NN already does.

  Test this first; it may decide whether graph methods work at all.
- **Extend the eval harness to score any labeling**, not just k-means at k=16. It needs
  a policy for HDBSCAN's noise label and comparisons at matched granularity or across a
  sweep.
- **Rare-pattern recall.** Nothing currently measures whether rare patterns get their
  own clusters. Hand-label a few dozen frames of each known rare pattern, then score
  cluster purity and recall on them. This is the most direct test of the goal.
- **Suggested first prototype:** Leiden (with the temporal-neighbor fix) and
  over-cluster-then-merge, on `ps_s3o4w7_192_nopix` and one DINOv3 embedding.

## 2. White balancing / photometric normalization

The user is getting the needed data. Motivation:

- **Raw frames:** within a pattern class, video η² is 0.55 for a*, 0.51 for black level,
  0.49 for saturation, 0.48 for sharpness, and 0.45 for b*. Session explains most of the
  color and black-level effect.
- **MSPS composites:** `z_b` controls a* (η² 0.45), saturation (0.33), and b* (0.30).

**Needed:**

- Raw, unmasked frames. The crop videos are masked onto black, so the white tank
  background is gone. A few empty-tank frames per session are enough.
- For new recordings, if still possible: a white card or color checker in frame at the
  start of each session, and exposure and white balance locked, not automatic.
- Otherwise, find out whether the camera settings were recorded.

**Order:** do cluster-level analyses (e.g. the c11/c21/c28 breakdowns) only *after*
white balancing, since clusters may change.

## 3. Photometric subspace `z_p` for the MSPS-VAE (design note, not started)

**The problem with a plain invariance term:** asking `z_u` to be invariant to
photometric augmentation doesn't work as first proposed. Two augmented views share a
session, so the variance can't simply be pushed into `z_b`, which must keep them close.

**User's proposal:** add a third subspace `z_p`. Minimize the MSE between the `z_u` and
`z_b` of two augmented views of the same image, and leave `z_p` free, so it learns to
encode the augmentation differences.

**Issues to design around:**

- Scale collapse: `z_u` and `z_b` can shrink to satisfy the MSE; normalize or add a
  variance floor.
- `z_p` may absorb more than photometrics unless it is constrained. Options: low
  dimension, or supervision on the known augmentation parameters.
- The two views need identical geometry, so apply only photometric transforms to the
  pair.
- The existing beast `default` imgaug preset has only Fliplr and CropAndPad; photometric
  transforms must be added through the config.

Hold off until preprocessing (section 2) is decided, since the models will probably be
retrained on the new data anyway.

## 4. Within-individual session analyses (wait for 12 sessions per individual)

**Done on iter-1.1** (`subset_analysis.py`, local per-individual UMAPs, which the user
chose deliberately): I2 and R3 × three models.

- Blocked k-NN session decoding (chance 0.33): masked 0.74/0.62, unmasked 0.76/0.60,
  VGG 0.95/0.79 (I2/R3).
- The intruder is higher than the resident in every model, which fits a tank/camera
  contribution. The resident is still well above chance, so there is also a day-level
  component.

**Wait for the new data:**

- run on all 12 individuals;
- chance-adjusted aggregation `(acc − chance)/(1 − chance)`, residents vs intruders,
  with a permutation test;
- pairwise session separability, to separate day effects from tank effects;
- visual inspection of session-specific clusters;
- freeze blocked within-individual session decoding as a before/after benchmark (an
  eval-harness candidate).

**Doable now or on the new data, to narrow scope** (suggested order a → c → b → d):

- a. a `--metrics-only` mode, to prune the list of backbones cheaply;
- b. a within-session control: decode the first half vs the second half of a session,
  as a baseline for drift within a day;
- c. subtract each session's mean latent and re-decode. If accuracy collapses toward
  chance, the session effect is a shift that per-session correction could remove; if not,
  it is structural;
- d. correlate pairwise session separability with photometric differences, using
  `.cache/raw_frame_stats.parquet`.

## 5. Portilla-Simoncelli texture embedding (follow-ups)

**Done:** `compute_ps_embeddings.py` computes grayscale PS statistics (3 scales, 4
orientations, correlation width 7, at 96×192; 529 statistics) plus a Lab color block.
It writes three variants in the `cuttle embed` layout:

| variant | contents |
|---|---|
| `ps_s3o4w7_192_lab` | PS + Lab color |
| `ps_s3o4w7_192` | PS only |
| `ps_s3o4w7_192_nopix` | PS without pixel statistics |

All three are scored in the eval harness. In the current scoreboard, no embedder is
clearly best overall:

| embedder | eff. videos ↑ | video AMI \| class ↓ | pattern AMI | x-video k-NN ↑ | probe ↑ |
|---|---|---|---|---|---|
| msps-vae_d16 z_u | 17.6 | 0.186 | 0.101 | 0.780 | 0.697 |
| ps_s3o4w7_192_nopix | 14.9 | 0.221 | 0.152 | 0.838 | 0.855 |
| dinov3_224 meanpatch_taper | 14.2 | 0.224 | 0.208 | 0.852 | 0.860 |
| vgg19_12345 gram_k28 | 13.1 | 0.269 | 0.216 | 0.852 | 0.865 |
| classifier_d512 | 12.3 | 0.273 | 0.234 | 0.882 | 0.982 |
| dinov3_224 gram_k64 | 11.7 | 0.316 | 0.208 | 0.847 | 0.862 |
| resnet-18_d16 | 9.0 | 0.466 | 0.161 | 0.822 | 0.815 |
| msps-vae_d16 z_b | 2.6 | 0.813 | 0.130 | 0.762 | 0.777 |

Removing the color block, then the pixel statistics, lowers video AMI from 0.269 to
0.249 to 0.221, at almost no cost to pattern retrieval.

**Tabled:**

- **Per-statistic-group session η²**, to drop the groups that carry the most session
  information. The user skipped this for now.
- **DISTS as a cluster-quality metric** that works for any model: compare
  within-cluster vs between-cluster similarity on sampled pairs. It is available in
  `piq`. It is VGG-based, so not independent of the VGG embedder.
- **SSIM / MS-SSIM for autoencoder reconstruction quality:** `po.metric.ssim` and
  `ms_ssim`.
- **Promote to `cuttle_patterns/embedders/`** if PS stays in the running. `plenoptic` is
  installed in the `cuttle` env but not listed in `pyproject.toml`.
- **Edge effects:** PS assumes a uniform texture, and its FFT-based pyramid wraps around
  at the crop edges. If that matters, try the existing taper in
  `embedders/spatial_weights.py`.

## 6. Known caveats / housekeeping

- **7 frames in `beast_frames` are completely black:**
  - Day1_Tank4 Resident, frames 44310, 44327, 44411, 44892, 44970
  - Day1_Tank4 Intruder, frame 33845
  - Day3_Tank6 Intruder, frame 44196

  This looks like an extraction bug that affects every model. Another 7 textured frames
  get NaN PS low-pass autocorrelations. All 14 are imputed in the PS embeddings.
- **`.cache/` files are keyed by model name only**, not by `results_dir`. The PS
  raw-stats cache only checks row count, so a rerun on the new batch could silently
  reuse the old statistics. Clear `.cache/` before rerunning on new data. The user chose
  not to change this yet.
- **The iter-1.1 BEAST model configs hard-code
  `data_dir: /media/mattw/CUTTLE/results/beast_frames`.** After the move to `.iter-1.1`,
  that points at the new batch. This matters only for retraining or re-predicting from
  those configs.
- **`common.RESULTS_DIR` follows `~/.cuttle-patterns/config.yaml`.** Point `results_dir`
  there at `.iter-1.1` to reproduce iter-1.1 analyses.
- **Move `eval/` with the archive.** `metrics.json` merges rows rather than overwriting,
  so otherwise new-batch scores land in the same table as the iter-1.1 ones.
