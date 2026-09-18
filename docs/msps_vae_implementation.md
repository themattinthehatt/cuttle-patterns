# MSPS-VAE Implementation Plan

**Status:** implemented, on beast's `msps-vae` branch
(`~/Dropbox/github/paninski-lab/beast`) — see "Where this lives" below for why it's there
and not in this repo. `configs/beast_msps_vae.yaml` in this repo mirrors beast's
`configs/msps_vae.yaml`. A real performance bug was found and fixed post-implementation
(see "Implementation gotcha" below) before the first real training run. This doc tracks
the design and stays here (rather than in the beast repo) because it's the record of *why*
we're doing this for the cuttlefish project specifically; the beast-side branch/PR should
link back to it.

## Motivation

UMAP projections of both trained backbones so far (the ResNet18 autoencoder and an
exploratory ViT + temporal-InfoNCE run) cluster primarily by `video_name` rather than by
overall pattern type (black-with-white-streaks vs. all-white vs. brown-blobs, etc.) — the
opposite of what Phase 6/7 need. For the ViT + InfoNCE path specifically, the root cause is
identifiable: BEAST's `ContrastBatchSampler` pairs each anchor with a temporal ±1 neighbor
as the positive and treats every other batch member (including same-video frames) as a
negative, which makes "which video is this" a valid — and much easier — shortcut than
pattern semantics for satisfying the InfoNCE objective (see Robinson et al., *Can
Contrastive Learning Avoid Shortcut Solutions?*, NeurIPS 2021).

We don't have labels for pattern type, and constructing genuine cross-individual
"same-pattern" positive pairs for a standard contrastive approach isn't currently possible
without either such labels or a bootstrapped/self-training scheme. What we *do* have for
free, on every frame, is session/video identity (`video_name`, already the parent directory
of every exported frame — see `BaseDataset._get_single_item` in
`beast/data/datasets.py`, which already surfaces `video=img_path.parts[-2]` per item). The
MSPS-VAE (Whiteway et al., *Partitioning variability in animal behavioral videos using
semi-supervised variational autoencoders*, PLOS Computational Biology 2021) is built for
exactly this shape of problem — reusing identity/session labels to explicitly separate
identity-driven variability from behavior/content-driven variability — and reframes the
whole approach as reconstruction-based rather than contrastive, sidestepping the
cross-individual-positive-pair problem entirely.

We're reusing only the multi-session partitioning idea from that paper, not the
pose-supervised branch — no pose labels are relevant to this use case, so our version has
two latent subspaces, not three.

## Architecture

Base: the existing ResNet18 autoencoder (`beast/models/beast_resnet/`), reusing
`ResNetEncoder`/`ResNetDecoder` unchanged. The single flat `LatentMapping` bottleneck is
replaced by two:

- **`z_u`** (unsupervised/pattern subspace): the majority of latent capacity; this is
  what Phase 5/6 (`cuttle reduce`/`cuttle cluster`) should ultimately read from.
- **`z_b`** (background/identity subspace): a small number of dimensions, shaped by the
  triplet loss below to organize by session/individual identity.

Decoder input is `concat(z_u, z_b)`, same as the current single-`z` design.

**Orthogonality: structural, not a loss term.** Rather than a soft `||UU^T − I||`
orthogonality penalty, `z_u`/`z_b` are produced as fixed, non-trainable slices of a single
random orthogonal matrix, drawn once at model construction and frozen for the rest of
training — the approach from an earlier PS-VAE implementation
(`~/Dropbox/github/paninski-lab/behavenet/behavenet/models/vaes.py`, `ConvAEMSPSEncoder`),
itself based on Li et al., *Latent Space Factorisation and Manipulation via Matrix Subspace
Projection*, 2019. Concretely, in that code: a single `m = scipy.stats.ortho_group.rvs(dim=
n_latents)` is drawn in `__init__`, and the encoder's per-subspace linear layers (`A`/`B`/`C`
there, for supervised/unsupervised/background) get their weights set directly from
disjoint row-slices of `m`, with `requires_grad=False`. Since `m`'s rows are orthonormal by
construction, any such row-partition is orthogonal by construction too — no loss needed to
enforce it, and it can't drift or destabilize during training. That codebase's own history
confirms the soft penalty was worth dropping: the `gamma`/`subspace_overlap` loss term is
present in `vaes.py` but fully commented out, superseded by this fixed-matrix approach
once it proved more stable.

Our version needs only two slices (`z_u`, `z_b`), no `A`/`D` (no supervised/label branch).
On the decoder side this needs no special inverse-transform step: `concat(z_u, z_b)` is
just an orthogonal rotation of the shared pre-partition feature vector, and the existing
`LatentMapping(source='latents')` decoder module can be reused completely unchanged — it
already learns an arbitrary linear map back to decoder feature space, which subsumes
whatever inverse rotation is needed.

**Loss terms:**
1. Reconstruction MSE over the full frame (unchanged from the current ResNet-AE).
2. Triplet loss on `z_b` only: pulls same-session pairs together, pushes different-session
   pairs apart. See sampler design below for how pairs are chosen — this is the part
   adapted most heavily from the base MSPS-VAE design based on cuttlefish-specific
   concerns.

That's it — two loss terms, not three. No adversarial/gradient-reversal component, and no
total-correlation penalty within `z_u` (present in the base PS-VAE to decorrelate the
supervised/unsupervised split) — not needed for a two-subspace version and can be added
later if `z_u`'s own dimensions turn out to be entangled in some unhelpful way.

## Sampler design (the main cuttlefish-specific deviation from the paper)

A naive "any frame from the same session is a positive" triplet, as in the paper, risks a
specific failure mode: a single individual's displayed pattern can change dramatically
within one session (e.g. black-streaky → rocky), so two frames from the same video can be
less similar in content than two frames of the *same* coarse pattern type from two
*different* individuals. Standard hard-negative-style triplet mining would concentrate
gradient pressure on exactly these conflicting cases (large within-session distance, small
between-session distance for a superficially similar pattern), risking either distorting
`z_u` to force unrelated pattern states together, or degrading reconstruction fidelity to
satisfy the margin.

**Mitigation — temporally-local positives:**
- Positive: a randomly chosen frame from the *same video*, within a **±1000 frame** window
  of the anchor (starting guess; at the original 24fps delivery this is ~42s — plausibly
  short enough that most sampled pairs stay within one pattern state, but this is not
  derived from any measured transition timescale and should be revisited if diagnostics
  below suggest otherwise).
- Negative: a randomly chosen frame from any *different* video present in the batch. No
  hard/semi-hard negative mining to start — unlike the positive side, "different video"
  is always a correct claim regardless of how visually similar the pair happens to be, so
  mining here would only affect convergence speed, not correctness. Batch composition just
  needs to guarantee several distinct videos are represented per batch (standard
  metric-learning batch construction), not a specific mining strategy.
- **Fallback if training stalls or `z_b` doesn't separate well in the leakage-probe
  diagnostic:** batch-hard mining (Hermans et al., *In Defense of the Triplet Loss for
  Person Re-Identification*, 2017) — computed directly from the pairwise distance matrix
  the loss already needs, so no new sampling infrastructure required to add this later.

## Where this lives (beast repo, new branch)

New model package, following the existing per-model convention documented in
`beast/models/__init__.py` and `docs/developer_guide.md`:

```
beast/models/msps_vae/
    msps_vae_config.py   # Pydantic config (model_params below)
    msps_vae_model.py    # MspsVae LightningModule + OrthogonalSplit; reuses
                          #   ResNetEncoder/ResNetDecoder/LatentMapping from beast_resnet
    msps_vae_train.py    # train entry point (delegates to beast.train.train)
    __init__.py
```

Registered via the standard three lines in `beast/models/registry.py`'s `_register_all()`,
and added to the `ModelConfig` discriminated union in `beast/config.py` (falls back to the
shared `BeastConfig`/`TrainingConfig`/`OptimizerConfig`, same as `resnet`/`vit` — no
divergent training schema needed).

The encoder-side change is `OrthogonalSplit`, a small module holding two frozen
`nn.Linear` layers (`to_unsupervised`, `to_background`) whose weights are set once from
row-slices of a random orthogonal matrix and frozen (`requires_grad=False`) — see the
"structural, not a loss term" note above. Built via `torch.nn.init.orthogonal_(matrix,
generator=torch.Generator().manual_seed(seed))`, not `scipy.stats.ortho_group` (avoids
adding a scipy dependency), seeded by the model's own `orthogonal_matrix_seed` config
field (default 42, independent of the training seed so it's stable across seed sweeps).
The decoder-side `LatentMapping` is reused as-is (see above), so `OrthogonalSplit` plus
the existing `LatentMapping(source='encoder')` feeding it is the only new module needed
for the bottleneck itself.

New sampler infrastructure landed in `beast/data/samplers.py` alongside
`ContrastBatchSampler`: `extract_windowed_positive_pool` (±window same-video candidate
pool, generalizing `extract_anchor_indices`'s exact-±1-neighbor pairing) and
`TripletBatchSampler` (same rank/world-size-aware DDP pattern and `used`-set epoch
bookkeeping as `ContrastBatchSampler`, but drawing from that windowed pool instead of a
fixed offset), plus `triplet_collate_fn` (like `contrastive_collate_fn`, but also carries
`video` through per item — needed so the loss can pick different-video negatives from
within the batch). `beast/data/datamodules.py`'s `BaseDataModule` generalized its old
`use_sampler: bool` flag to `sampler_kind: Literal['none', 'contrastive', 'triplet']` (+
`positive_window`) to dispatch to the right sampler/collate pair; `beast/train.py` routes
`model_class == 'msps_vae'` to `sampler_kind='triplet'`.

**Implementation gotchas found post-implementation, all in `TripletBatchSampler`:**

1. **O(n) candidate-pool membership check.** The per-anchor positive-pool filter was
   copy-pasted from `ContrastBatchSampler`'s identical-looking line, `p in
   self.dataset_indices` — a plain Python list membership test. In `ContrastBatchSampler`
   this is cheap because its candidate pool is at most 2 items (idx_offset=1). Our
   windowed pool is up to ~2000 items (±1000 frames), so the same check runs ~1000x more
   often per anchor, against a list that can be tens of thousands of entries long.
   Benchmarked on a 50k-frame split: **~65 seconds of pure CPU overhead for a single
   batch's worth of anchor draws** with the list. This is what caused a first training
   attempt to run ~10x slower than the plain ResNet-AE baseline. First fix: a `set` for
   O(1) lookup (~15ms/batch). Second pass (see next item) removed the check entirely —
   turned out to be dead code once local positions stopped needing sortedness (every
   value in `pos_indices[i]` is, by construction, always a valid local position).

2. **`_sequential_split` wasn't actually required — it was a symptom of sorting local
   positions.** `Subset.__getitem__(j)` fetches via `self.indices[j]` in whatever order
   the Subset was constructed with. But the sampler built its local-position bookkeeping
   via `sorted(dataset.indices)`, which only agrees with `Subset.__getitem__` when
   `dataset.indices` is already ascending — true for `_sequential_split`'s contiguous
   ranges, false for `random_split`'s shuffled permutation. This forced `BaseDataModule`
   to give any sampler-based model (`sampler_kind != 'none'`) a *sequential* train/val
   split — first N frames train, rest val/test, by sorted path order — instead of a
   random one, since swapping in `random_split` without fixing this would have silently
   mispaired images (local position 42 in the sampler's bookkeeping ≠ position 42 in what
   `Subset` actually returns). This had a real, non-obvious consequence for msps-vae
   specifically: checked empirically on the real ~94k-frame/32-video dataset, the
   sequential split put 31 videos in train and only 2 in val (1 of them never seen in
   train at all) — vs. resnet's random split, where all 32 videos appear in both, a much
   easier near-interpolation task. **This made `val_loss` not directly comparable between
   the ResNet-AE and MSPS-VAE runs.** Fixed by dropping the `sorted()` call —
   `extract_windowed_positive_pool` doesn't need global sortedness, it already sorts by
   frame number *within* each video internally, so local position just needs to stay
   consistent with whatever order the Subset gives it, sorted or not. `BaseDataModule`
   now routes `sampler_kind='triplet'` through `random_split` (same as `'none'`);
   `sampler_kind='contrastive'` (`ContrastBatchSampler`, untouched) still uses
   `_sequential_split`, since it has the identical dormant dependency and wasn't part of
   this fix.

3. **`num_batches` was silently discarding ~half the dataset every epoch.** Each
   successful pair only marks 2 frames used (the anchor and its chosen positive — unlike
   `ContrastBatchSampler`, which marks up to 3), so the achievable pair count is
   `anchors_per_replica // 2`, and the achievable batch count is that many pairs divided
   by pairs-per-batch (`batch_size // 2`). The original formula divided by `batch_size`
   (total images per batch) instead of `batch_size // 2` (pairs per batch) — an
   accidental extra ~2x undercount, inherited from eyeballing `ContrastBatchSampler`'s own
   (differently-justified) `// 3` without re-deriving it for this sampler's own
   consumption pattern. Confirmed on the real dataset: before the fix, 512-batch epochs
   covered only ~50% of available frames; after, ~98% (171 of a resnet-equivalent 175
   batches — the shortfall is real, not an off-by-one: occasional pool depletion near
   video edges/short videos means the literal `// 2` ideal can't always be hit exactly,
   confirmed by measuring 174 actually achievable vs. 175 declared with no margin at all,
   so a small 2% conservative margin was kept rather than the literal ideal formula — see
   the code comment for why `__len__` must never exceed what `__iter__` actually
   produces).

This is deliberately *not* built as new code in `cuttle_patterns` reusing beast
internals — `cuttle train` shells out to `beast train` as a subprocess (see "cuttle
train/cuttle predict: subprocess wrappers around BEAST's own CLI" in `DECISIONS.md`),
specifically to avoid coupling to beast's internal API surface. A model registered only
from the `cuttle_patterns` process wouldn't be visible to that subprocess. Building it in
beast keeps that existing architecture untouched on the cuttle-patterns side.

## Config surface (new `model_class: msps_vae`)

New `model_params` relative to `configs/beast_resnet_ae.yaml`, with the values chosen for
the first real run (`configs/beast_msps_vae.yaml` here, `configs/msps_vae.yaml` in beast):
- `num_latents_unsupervised: 12` (`z_u` size)
- `num_latents_background: 4` (`z_b` size) — total 16, deliberately matching the
  already-trained `iter-1.1_resnet-18_d16` ResNet-AE's `num_latents: 16`, so that AE's
  latents can be used as a like-for-like scale reference (see calibration below) and so
  the two runs are otherwise comparable.
- `triplet_margin: 1.0` — checked against the `d16` AE's actual latent scale (per-dim std
  ≈1.0-1.4, mean pairwise L2 distance ≈5-6.5 over the full 16 dims) rather than picked
  blind: scaling that down to a 4-dim orthogonal projection (`sqrt(4/16)` factor, assuming
  a roughly isotropic pre-split vector) gives an expected `z_b` pairwise distance of
  ≈2.5-3.25, so a margin of 1.0 is the same order of magnitude (~30-40% of typical
  distance) — plausible, not badly miscalibrated, without needing an actual MSPS-VAE run
  to check first.
- `triplet_weight: 0.1` — calibrated against the `d16` AE's actual converged MSE, read
  directly from its TensorBoard logs (`train_mse_epoch`/`val_mse` at epoch 800: ≈0.026-
  0.028), not assumed. Combined with a bootstrapped estimate of mean triplet loss at
  margin=1.0 (≈0.4, using the `d16` AE's empirical same-video vs. cross-video pairwise
  distance distributions, scaled to 4 dims, as a stand-in for `d(a,p)`/`d(a,n)`), an
  MSE-matching weight is ≈0.026/0.4 ≈ 0.065-0.15 depending on assumptions —
  0.1 sits in that range. This is a converged/steady-state estimate; early in training MSE
  is much larger (≈4-5, measured directly on a freshly-initialized `MspsVae` forward
  pass) and dominates regardless of this weight, which is the desired behavior
  (reconstruction bootstraps first). `train_mse`/`train_triplet` are logged as separate
  unweighted scalars in `compute_loss`, so the real ratio is directly checkable in
  TensorBoard once training runs rather than trusted from this estimate alone.
- `positive_window: 1000` (frames; the ±1000 starting guess from the Sampler design
  section above, unchanged)
- `orthogonal_matrix_seed: 42` (for reproducibility of the fixed random orthogonal matrix
  — no loss weight needed, since orthogonality is structural, not a training objective;
  see Architecture above)

`data.data_dir` stays `results_dir/beast_frames`, same `BaseDataset` scan as today — no
new data plumbing needed, since `video_name` is already derived per-frame from the
directory structure.

## Spatial loss weighting for rectangle-inscription edge/corner leakage

**Date:** 2026-09-18

**Motivation:** inspecting representative frames per cluster for `iter-1.1_msps-vae_d16`'s
`kmeans_k16` (`scratch/plot_cluster_frames.py`, via the dashboard) surfaced a second
confound in `z_u`, distinct from the identity leakage the `z_b` split above already
addresses: several clusters that are visually the same coarse pattern (e.g. "dark
aggression" — dark base, white streaks/spots) split apart by *where* the inscribed
rectangle's edge happens to catch a sliver of background just outside the body (clusters
4/5/12 in that run: same pattern, but background visible in the lower-left, upper-left,
and along the whole bottom edge respectively). Unlike the identity confound, this isn't
predictable from the rectangle's absolute position or size in the raw frame
(`scratch/compute_rectangle_scale.py` confirmed the inscribed rectangle's aspect ratio is
exactly fixed at construction — `grow_rectangle` only ever searches over a single scale
variable — so there's no per-frame anisotropic stretch to blame either, only a uniform
scale factor that varies with body size/distance/contraction). The leakage is driven
instead by the animal's instantaneous body shape/pose relative to `grow_rectangle`'s
fixed-aspect box fit, which generically fits worst at the box's corners for a
non-rectangular (elongated) body — i.e. it can occur regardless of where or how large the
rectangle is, briefly, whenever the body's shape deviates from filling the box cleanly.

**Decision:** down-weight reconstruction MSE spatially (a fixed, non-trainable per-pixel
weight map multiplied into the loss), rather than crop the input or mask the input
pixels. Reconstruction-loss weighting was chosen specifically because it requires no
change to the input transform at inference time — masking/cropping the input would need
to be applied identically, at extra pipeline cost, across every frame of every video at
inference; a loss-only change touches training alone. Trade-off accepted knowingly: this
only reduces the *incentive* to encode edge/corner content, it doesn't prevent the
encoder from picking it up if doing so is cheap regardless of reward — if the confound
persists after this change, that's the signal to revisit input-side masking despite its
cost, not evidence the diagnosis was wrong. Runner-up profiles (plain Gaussian,
super-Gaussian — both smooth but without the plateau/exact-zero-slope property below) are
in `scratch/plot_loss_weight_profiles.py` alongside the chosen one, for the same reason
this doc keeps runner-up options elsewhere (see Open questions below).

**Chosen profile: raised-cosine (Tukey-style) radial taper.** Flat at weight 1.0 for
`r <= r0`, a cosine taper down to exactly 0 (both value and slope) at `r = 1`, and exactly
0 beyond:

```
w(r) = 1                                            for r <= r0
w(r) = 0.5 * (1 + cos(pi * (r - r0) / (1 - r0)))     for r0 < r <= 1
w(r) = 0                                            for r > 1
```

`r` is the distance from the (square) crop's center, normalized so `r = 1` sits at each
edge's midpoint and corners sit at `r = sqrt(2)` — always past the taper's zero point,
regardless of `r0`, so corners are fully ignored by construction with no corner-specific
logic needed. Chosen over the Gaussian/super-Gaussian alternatives for having an explicit,
interpretable plateau (rather than an approximate one) and for reaching zero with zero
slope at the cutoff — no new discontinuity at the boundary for the model to exploit.
Default `r0 = 0.5`, picked by eye from `scratch/plot_loss_weight_profiles.py`'s comparison
figures (written to `results_dir/beast_frames_qc/loss_mask/`) — both the abstract
radial-profile plot and the same three candidate masks overlaid on ~20 real sampled
frames, resized to square the way the autoencoder sees them.

**Normalization spec:** the weight map is precomputed once (not trainable, no grad) at
whatever square side length the MSPS-VAE's actual input resolution is, then
**renormalized so its mean is 1.0** before being multiplied into the per-pixel squared
error:

```python
weight_map = weight_map / weight_map.mean()          # keeps average weight ≈ 1
loss = ((recon - target) ** 2 * weight_map).mean()
```

Without this, the loss's overall magnitude drops just from adding the map (most of it is
small or zero), which would silently shift the balance against the triplet term's fixed
`triplet_weight` (see Config surface above) — renormalizing keeps the reconstruction
term's scale where it was, so the existing `triplet_weight=0.1`/`margin=1.0` calibration
stays valid and the only real change is *which pixels* contribute to the loss.

**Scope:** training-side loss change only, in `beast/models/msps_vae/msps_vae_model.py`'s
loss computation. No change to `cuttle extract`/`align`, and no change to the input
pipeline at inference — same reasoning as the "Where this lives" section above, keep
beast-side changes contained to what training actually needs.

**Status: implemented**, on beast's `msps-vae` branch. New `model_params` fields:
`use_spatial_loss_weight` (bool, default `false`) switches it on/off, and
`spatial_loss_weight_r0` (default `0.5`) exposes the taper's plateau radius, matching the
Open Questions note above that `r0` may need revisiting. When off, `compute_loss` is
byte-for-byte the original unweighted `mse_loss` call — no behavior change for existing
runs/configs. `configs/msps_vae.yaml` (beast) / `configs/beast_msps_vae.yaml` (here) both
default it off, since it hasn't been validated against a real training run yet; flip it
to `true` to try it on `iter-1.1_msps-vae_d16`'s successor run.

## Evaluation / validation plan

Replacing "eyeball the UMAP colored by `video_name`" with concrete checks:

1. **Leakage probe:** k-NN purity or a small linear probe predicting `video_name` (and,
   once the fish_id mapping work lands, `fish_id`) from `z_u` vs. from `z_b`. `z_b` should
   be far more predictive than `z_u`; track this number across runs/hyperparameters
   instead of relying on visual inspection.
2. **Session-swap reconstruction test** (the paper's own validation method): swap `z_b`
   between two frames from different videos while holding `z_u` fixed, decode, and check
   whether the reconstruction correctly recombines one individual's skin-base/identity
   characteristics with the other's current pattern-state content.
3. **Within-session sub-clustering check:** for a video known (or suspected, from raw
   footage) to display more than one pattern state over its duration, confirm `z_u` splits
   into distinct regions rather than collapsing the whole video into one blob — direct
   evidence the ±1000-frame window is doing its job, distinct from the leakage probe (which
   only checks the identity side, not whether real content variation survived).
4. Re-run the existing UMAP-colored-by-`video_name` comparison against `z_u` from this
   model, alongside the existing ResNet-AE and ViT+InfoNCE embeddings, as the qualitative
   version of the same check that originally surfaced this problem.

## Open questions / things to tune

- **Latent split ratio** between `z_u` and `z_b` — started at 12/4 (see Config surface
  above), chosen for comparability with the existing `d16` ResNet-AE rather than any
  principled derivation; still expect empirical tuning, as in the paper's own mouse-data
  tuning.
- **Triplet weight/margin** — started at `margin=1.0`/`weight=0.1`, calibrated against the
  `d16` AE's actual measured latent scale and converged MSE rather than picked blind (see
  Config surface above) — but still an estimate from a different model's latents, not this
  model's own training dynamics. Governs how tightly sessions must cluster in `z_b`; too
  loose risks identity leaking back into `z_u`, too tight risks forcing genuine
  pattern-state content into `z_b` or degrading reconstruction. Validate primarily via the
  leakage probe and session-swap test (and the live `train_mse`/`train_triplet` ratio in
  TensorBoard), not by eye.
- **±1000-frame positive window** — starting guess, not derived from a measured
  pattern-state transition timescale; revisit if diagnostic #3 above shows positives still
  span real transitions, or if it's unnecessarily conservative.
- **Session (`video_name`) vs. individual (`fish_id`) as the triplet key** — `video_name`
  is available now and is the stricter choice (absorbs day/tank/lighting quirks in
  addition to individual identity); once the fish_id mapping lands, worth checking whether
  keying on `fish_id` instead (looser — only individual identity, not session-level
  artifacts) changes the leakage-probe/sub-clustering results.
- **Random vs. hard-negative mining** — starting with random cross-video negatives (see
  Sampler design above); revisit only if the leakage probe shows `z_b` isn't separating
  well.
- **TC penalty within `z_u`** — present in the base PS-VAE for the supervised/unsupervised
  split, omitted here; add later only if `z_u`'s own dimensions look entangled in a way
  that hurts downstream clustering.
- **Raised-cosine `r0` for the edge/corner loss weighting** (see the section above) —
  `r0=0.5` was picked by eye from comparison figures, not derived; revisit if the leakage
  probe/session-swap checks show it's cutting into real body pattern (too small) or not
  suppressing the edge confound enough (too large). Loss-weighting-only was chosen over
  also masking the input specifically to avoid added inference-time cost; if this alone
  doesn't resolve the confound, that cost becomes worth paying.

**Resolved during implementation:**
- **`torch.nn.init.orthogonal_` vs. `scipy.stats.ortho_group`** — used
  `torch.nn.init.orthogonal_(matrix, generator=...)`, no scipy dependency added; see
  "Where this lives" above.
- **Latent-saving format for `predict_step`** — resolved as a single concatenated
  `latents = concat(z_u, z_b)` tensor (matching BEAST's existing single-flat-tensor
  `predict_step` contract, so `beast/inference.py` needed no changes), with the split
  index documented as `num_latents_unsupervised` in `MspsVae.predict_step`'s docstring.
  **Still pending:** `cuttle_patterns/embeddings.py`'s loader doesn't yet know about this
  split — it will load the full 16-d concatenated vector as-is. Needs a follow-up change
  so Phase 5/6 read `z_u` (the first `num_latents_unsupervised` columns) only, not
  `z_u`+`z_b` concatenated, once training confirms the split index convention is right.

## Relationship to existing tracks

A third parallel backbone-training track, alongside the in-progress 8-latent ResNet-AE
(`configs/beast_resnet_ae.yaml`) and the originally-planned ViT + InfoNCE backbone (see
"Backbone sequencing" in `DECISIONS.md`). MSPS-VAE builds on the ResNet-AE's reconstruction
path specifically (shares `ResNetEncoder`/`ResNetDecoder`), not the ViT/contrastive path —
the two aren't mutually exclusive, but this is now the primary candidate for solving the
video-identity-clustering problem, ahead of further contrastive-sampling changes to the ViT
path.

On the `cuttle_patterns` side this needed only the new `configs/beast_msps_vae.yaml`
(`model_class: msps_vae`, mirroring beast's own `configs/msps_vae.yaml`) — `cuttle train`/
`cuttle predict` already pass `model_class` through opaquely via BEAST's own CLI, no code
changes required there. The one real follow-up on this side is the `embeddings.py` loader
change noted above (Phase 5/6 need to read `z_u` only from the concatenated latents file).

**Status as of the second training restart:** implementation complete and unit-tested on
the CPU-only paths (samplers, datamodule dispatch, model forward/loss/predict_step — see
`tests/models/msps_vae/`, `tests/data/test_samplers.py`, `tests/data/test_datamodules.py`
in beast); GPU integration test (`run_model_test`-style, actually calling `trainer.fit`)
not yet added, deferred while the GPU was occupied by the 8-latent ResNet-AE run and then
the first two `iter-1.1_msps-vae_d16` attempts. Two rounds of real-run diagnosis so far —
see "Implementation gotchas" above for all three fixes (candidate-pool lookup speed,
split strategy, batch-count undercounting). First attempt: ~10x slower than ResNet-AE
(candidate-pool lookup). Second attempt (after that fix): ran correctly, but comparing its
TensorBoard curves against the ResNet-AE's surfaced two more issues — `train_mse_step`
tracked resnet's closely and `train_triplet_step` was decreasing as expected, but
`val_loss` looked substantially worse (turned out to be the split-coverage issue, not a
real quality gap) and `lr-AdamW` looked different on a shared step axis (turned out to be
identical per-epoch, just spread over fewer steps/epoch than resnet's, itself a symptom
of the batch-count undercount). Both root-caused and fixed; training restarted again on
the corrected sampler/split.

## References

- Whiteway, M.R. et al. (2021). *Partitioning variability in animal behavioral videos using
  semi-supervised variational autoencoders.* PLOS Computational Biology. (PS-VAE / MSPS-VAE;
  multi-session background-subspace design and the session-swap validation method.)
- `~/Dropbox/github/paninski-lab/behavenet/behavenet/models/vaes.py` (`ConvAEMSPSEncoder`)
  and `aes.py` (`AEMSP.create_orthogonal_matrix`) — the earlier PS-VAE implementation this
  plan's fixed-random-orthogonal-matrix approach is taken from directly (internal
  reference, not published).
- Li, J. et al. (2019). *Latent Space Factorisation and Manipulation via Matrix Subspace
  Projection.* arXiv:1907.12385. (Origin of the fixed-orthogonal-projection idea, as cited
  in the behavenet code above.)
- Robinson, J. et al. (2021). *Can Contrastive Learning Avoid Shortcut Solutions?* NeurIPS.
  (Why the ViT + temporal-InfoNCE path clusters by video identity.)
- Schroff, F., Kalenichenko, D., Philbin, J. (2015). *FaceNet: A Unified Embedding for Face
  Recognition and Clustering.* CVPR. (Semi-hard negative mining — noted as a fallback, not
  adopted initially.)
- Hermans, A., Beyer, L., Leibe, B. (2017). *In Defense of the Triplet Loss for Person
  Re-Identification.* (Batch-hard mining — the cheap fallback if random negatives prove
  insufficient.)
