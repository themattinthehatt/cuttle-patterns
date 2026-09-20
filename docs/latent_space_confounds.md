# Latent space confounds: AE → MSPS-VAE → masked MSPS-VAE

Working summary of the recurring failure mode in the unsupervised embedding track —
useful clustering requires the latent space to organize by *abstract skin pattern*, but
each backbone tried so far has instead organized substantially around some cheaper,
lower-level signal. Written as a standalone status doc rather than a `DECISIONS.md`
entry, since the underlying problem (see "The recurring pattern" below) isn't resolved
yet — this is where things stand, not a closed decision. Full technical detail on the
MSPS-VAE architecture/sampler/config lives in
[msps_vae_implementation.md](implementation_notes/msps_vae.md); this doc is the higher-level
narrative across all three stages plus open thinking on where to go if the current fix
doesn't hold up.

## Timeline

### 1. Plain ResNet-18 autoencoder

First backbone trained (`configs/beast_resnet_ae.yaml`, `model_class: resnet`). UMAP of
the latents clustered primarily by `video_name` — i.e. by *which individual/session*,
not by skin pattern. An exploratory ViT + temporal-InfoNCE contrastive run had the same
problem, traced to a specific mechanism: `ContrastBatchSampler` pairs each anchor with a
temporal neighbor as the positive and everything else in the batch (including
same-video frames) as the negative, which makes "which video is this" a valid — and
much easier — shortcut than pattern semantics for satisfying InfoNCE. That shortcut
is the direct precedent behind the caution on contrastive approaches below.

### 2. MSPS-VAE: structural identity/pattern split

Reframed as reconstruction-based rather than contrastive, reusing free session/video
labels (Whiteway et al. 2021, PLOS Comp Bio). Two latent subspaces, produced as fixed
non-trainable slices of a random orthogonal matrix (orthogonal by construction, no soft
penalty needed): `z_u` (unsupervised/pattern, what clustering should read) and `z_b`
(background/identity, shaped by a triplet loss keyed on `video_name`). Full rationale in
[msps_vae_implementation.md](implementation_notes/msps_vae.md).

This fixed the *identity* confound. It did not fix a second one, found once real
per-cluster examples were inspected (`scripts/plot_cluster_frames.py`): several clusters
that were visually the same coarse pattern (e.g. "dark aggression" — dark base, white
streaks/spots) split apart by *where* the inscribed rectangle's edge happened to catch a
sliver of background just outside the body — clusters 4/5/12 in the `iter-1.1_msps-vae_d16`
`kmeans_k16` run, differing only in whether the leak sat in the lower-left, upper-left, or
along the bottom edge. Confirmed geometrically not to be predictable from the rectangle's
absolute position or size (`grow_rectangle` only ever fits a single scale variable, so
there's no per-frame anisotropic stretch to blame) — it's driven by the animal's
instantaneous body shape/pose relative to a fixed-aspect box fit, which structurally fits
worst at the box's corners for a non-rectangular (elongated) body.

Two fixes were considered and rejected before landing on the one actually built:
- **Condition the decoder on inscription geometry** (corner coords, rotation, scale) so
  `z_u` has no incentive to encode it — rejected because the real nuisance variable
  (*"does the rectangle include a sliver of background just outside the body"*) isn't the
  same thing as the geometry itself, and isn't otherwise well-defined/labelable.
- **Raw-video temporal-median background subtraction**, warped through the same per-frame
  affine transform used for the egocentric crop, to get a well-defined per-pixel leakage
  signal — rejected because the leakage is pose/shape-driven, not position-driven; a
  static background model can't capture "the box's corner briefly caught body edge because
  the animal's pose deviated from filling a rectangle cleanly."

### 3. Masked MSPS-VAE: spatial loss weighting

Chosen fix: down-weight reconstruction MSE spatially with a fixed, non-trainable
raised-cosine radial taper (full weight at center, zero at/before the square input's
edges and corners), rather than mask or crop the input pixels — see
[msps_vae_implementation.md](implementation_notes/msps_vae.md) for the exact profile,
mean-1-renormalization spec, and calibration against the existing
`triplet_weight`/`triplet_margin`. Loss-weighting was picked over literal input masking
specifically because masking would need to be applied identically, at extra pipeline
cost, to every frame of every video at inference — a loss-only change only touches
training. (The project's own naming for this stage — `use_spatial_loss_weight` in config,
"masked msps-vae" in commit history/branch names — is a slight misnomer worth flagging
here: nothing is masked at the input; it's a loss weight. Worth keeping straight if this
doc or the commit history gets referenced later.)

**Status:** implemented on both sides (`beast`'s `msps-vae` branch,
`cuttle-patterns`'s `configs/beast_msps_vae.yaml`), toggled on
(`use_spatial_loss_weight: true`, `spatial_loss_weight_r0: 0.5`, committed). Not yet
evaluated against a real training run's clusters.

## The recurring pattern

Each fix so far has targeted one *specific instance* of the same general failure mode,
not the failure mode itself: plain MSE reconstruction loss has no notion of "abstract
skin pattern" — it rewards accurate reconstruction of *any* pixel-level structure that's
cheap to encode, and abstraction over pattern-type is not something reconstruction loss
selects for on its own. Identity (`z_b` split) and rectangle-edge geometry (spatial loss
weighting) are the two instances diagnosed and fixed so far. Nothing rules out a next one
surfacing once the current fix is evaluated — lighting/contrast variation, residual
rectangle-aspect/scale variation, or something not yet visually obvious in cluster grids
are all plausible candidates, by the same logic that made the first two findable only
after inspecting real per-cluster examples rather than by design review.

## Current open issue

The underlying goal — clusters organized by abstract pattern type (the coarse categories
the collaborator's supervised classifier hand-labels, plus rarer/ambiguous ones it can't
capture) — has not yet been achieved end to end. Every structural fix applied so far has
been reactive: found by eyeballing real cluster examples after a training run, not
predicted in advance. The masked (loss-weighted) MSPS-VAE run is the next checkpoint
against that goal; if lower-level features (geometry, identity, or something new) are
still visibly organizing the clusters after it, that's the signal the reconstruction-loss
approach has hit a harder limit than these two fixes address, which is what motivates
keeping contrastive and I-JEPA in view below rather than iterating indefinitely on more
reconstruction-side patches.

## Thoughts on contrastive loss

Two variants came up, ranked oppositely from how they might first look:

- **Classifier-guided positive/negative pairs** (use the collaborator's 6-class
  supervised classifier: same predicted class = positive, different class = negative) —
  the lower-ranked option, and deliberately so. This doesn't just risk "human bias" in
  the abstract; it would actively suppress the rare/ambiguous patterns the whole project
  is trying to surface, since those get forced to be "negative" against whatever known
  class they superficially resemble. It would effectively distill the six-class
  classifier into the embedding rather than discover structure beyond it — the opposite
  of the stated goal (finding patterns the supervised model can't capture).
- **Temporal-proximity pairs** (same video, nearby frame = positive; different video =
  negative — unbiased in principle, no human labels involved) — the better-principled
  option, but not a free win: this is *exactly* the sampling shape that produced the
  video-identity shortcut in the original ViT+InfoNCE run (stage 1 above). That's not a
  hypothetical risk, it's a concrete precedent — whatever's the cheapest signal
  consistent with the sampler's pairing scheme is what a contrastive objective will
  learn, and rectangle geometry is a plausible next shortcut for the same reason video
  identity was (both vary slowly and are spatially/temporally local). Any future
  contrastive attempt needs the sampler explicitly audited for shortcut opportunities
  before trusting the objective change alone to fix anything.

Net position: contrastive learning isn't rejected outright, but it sits behind both
reconstruction-side fixes tried so far, given the classifier-guided variant's clear bias
risk and the temporal variant's direct shortcut precedent.

## Thoughts on I-JEPA

Predicting masked latent patches instead of raw pixels is attractive in principle — it
reduces the incentive to memorize low-level pixel/texture detail the way plain MSE does.
But it isn't a guaranteed fix for *this* problem: reconstructing in latent space doesn't
inherently discard a systematic spatial confound like rectangle-edge leakage unless the
masking/context-target sampling strategy is specifically designed to break that
correlation — "latent-space reconstruction" is not automatically "invariant to position."
It's also a substantially bigger engineering lift than either MSPS-VAE fix tried so far
(context/target encoder pair, EMA target update, masking-strategy design), for a dataset
this size, relative to how much is already invested in the MSPS-VAE pipeline. Current
status: kept as a lower-priority fallback, worth reaching for only if the reconstruction-
side fixes (z_b split, spatial loss weighting) and/or a carefully shortcut-audited
contrastive scheme genuinely don't resolve the pattern-vs-lower-level-feature confound.

## Evaluating the masked MSPS-VAE run

Once training completes, the existing evaluation plan in
[msps_vae_implementation.md](implementation_notes/msps_vae.md) (leakage probe, session-swap
reconstruction test, within-session sub-clustering check, plus the qualitative
`scripts/plot_cluster_frames.py` inspection that found both confounds so far) applies
unchanged — no new evaluation machinery needed, just a fresh model to point it at.
