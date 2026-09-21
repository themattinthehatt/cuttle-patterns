# Embeddings implementation

**Doc type:** living implementation reference — edited in place as the embedder
implementation changes.

Reference for the frozen-pretrained-backbone embedders that feed the
[evaluation harness](eval_plan.md) as Tier B. Covers the `Backbone`/`Readout`/`Embedder`
protocol every embedder implements, a DINOv3 backbone with CLS-token, mean-patch, and
Gram-matrix (second-order texture) readouts, a VGG-19 backbone (single-layer and
multi-layer Gram fusion) built for the same protocol, and the `cuttle embed` CLI that
runs any of them over exported frames.

Code: `cuttle_patterns/embedders/{base,dinov3,readouts,spatial_weights,gram_math}.py`,
`cuttle_patterns/embed.py` (the `cuttle embed` extraction loop),
`cuttle_patterns/cli/cmd_embed.py` (CLI wrapper).

## Embedder protocol

An embedder maps a batch of egocentric crops to a batch of fixed-length vectors. It's
composed of a **backbone** (runs the network once and returns tokens) and a **readout**
(turns those tokens into a vector).

**Deliberately not optimized: no cross-readout backbone sharing.** `cuttle embed` runs
one embedder — one backbone + one readout — per invocation, writing its own output
directory. Getting CLS, mean-pooled, and Gram readouts for the same frames means running
the command three times, each redoing the DINOv3 forward pass. This is a deliberate
simplicity-over-compute tradeoff: inference is cheap relative to how often this gets
iterated on, and a single-embedder-per-run CLI is much easier to reason about (one
`--model-name`, one output directory, one failure mode) than a batch script juggling
multiple readouts' fit state and outputs at once.

Embedders own their preprocessing — the harness hands them raw crops in one canonical
format and never normalizes or resizes on their behalf. Embedders also don't
L2-normalize or PCA their outputs for clustering purposes; the eval harness applies its
own standardized clustering conventions (L2-normalize, PCA to 64) uniformly across every
embedding. The only reduction an embedder applies is one that's part of its own
definition — for the Gram readout, the channel projection described below.

Interface (`cuttle_patterns/embedders/base.py`):

```python
@dataclass
class TokenOutput:
    """What a ViT-style backbone returns for one batch."""
    cls: torch.Tensor          # (B, C)
    patches: torch.Tensor      # (B, N, C), N = grid_h * grid_w, row-major
    grid_hw: tuple[int, int]   # (grid_h, grid_w)


class Backbone(ABC):
    key: str                   # e.g. "dinov3_vitb16_448"; shared by all readouts on it

    def preprocess(self, frames: Frames) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor) -> TokenOutput: ...
    def metadata(self) -> dict: ...   # model id, resolution, normalization, dtype


class Readout(ABC):
    name: str                  # e.g. "cls", "meanpatch_taper", "gram_k64_taper"
    fit_passes: int = 0        # number of passes over a fit set; 0 = stateless

    def partial_fit(self, tokens: TokenOutput, pass_idx: int) -> None: ...
    def finalize_pass(self, pass_idx: int) -> None: ...
    def __call__(self, tokens: TokenOutput) -> torch.Tensor: ...   # (B, D)
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict) -> None: ...
    def metadata(self) -> dict: ...


class Embedder:
    """One backbone + one readout: the unit of work for a single `cuttle embed` run."""

    id: str            # f"{backbone.key}_{readout.name}", e.g. "dinov3_vitb16_224_cls"
    dim: int           # readout.dim
    requires_fit: bool # readout.fit_passes > 0

    def embed(self, frames: Frames) -> np.ndarray: ...   # (B, D) float32
```

### Fitting stateful readouts

A readout with `fit_passes > 0` (currently only `gram`) has parameters fit from data
before it can embed anything. Fitting runs `fit_passes` passes over a **fit set**: a
random, per-video-capped sample of frames drawn from the same directory being embedded
(`cuttle_patterns.embed.sample_fit_frame_paths`, seeded for reproducibility). Each pass
calls `partial_fit` on every batch's backbone output, then `finalize_pass` once at the
end of the pass. `cuttle embed` runs this automatically, before the main embedding loop,
whenever `embedder.requires_fit`.

Fitting reruns from scratch on every `cuttle embed` invocation rather than caching state
across runs — the fixed seed and fit-set size make it deterministic, so this always
reproduces the same fitted parameters for the same flags, with no cache-invalidation
story to maintain. `write_embedder_output` still writes the readout's `state_dict()` to
`{model_dir}/readout_state.pt` when non-empty, as a provenance record for later
inspection — nothing in `cuttle_patterns` reads it back.

## CLI: `cuttle embed`

See README.md's `cuttle embed` section for a basic DINOv3 example. A VGG-19 example:

```bash
cuttle embed --backbone vgg19 --vgg-layer relu3_1 --readout gram
# writes to results_dir/beast_models/vgg19_3_448_gram_k64_taper/ by default
```

- `--backbone`/`--resolution`/`--readout` select the embedder: `vits16`/`vitb16`/
  `vitl16` (DINOv3, `--resolution` any multiple of 16, default 224) or `vgg19` (see
  "VGG-19 backbone" below; `--resolution` a multiple of `--vgg-layer`'s downsampling
  factor, default 448) × `cls`/`meanpatch_uniform`/`meanpatch_taper`/`gram` (`cls` is
  DINOv3-only — VGG has no CLS token). Together with the readout's own hyperparameters,
  these determine `embed_dim`.
- `--gram-k`/`--gram-weights {uniform,taper}` configure the `gram` readout (default
  `64`/`taper`; ignored for other readouts) — see "Gram readout" below.
- `--vgg-layer {relu1_1,relu2_1,relu3_1,relu4_1,relu5_1}` selects which VGG-19 layer to
  read from (default `relu3_1`; ignored for `--backbone` other than `vgg19`).
- `--model-name` defaults to `{backbone key}_{readout name}`, where `backbone key`
  already encodes arch/layer/resolution (e.g. `dinov3_vitb16_224_cls`,
  `vgg19_3_448_gram_k64_taper` — VGG's layer name is aliased to a short digit code in
  the key, `relu3_1` → `3`, so the default name stays readable; `metadata()`'s
  `vgg_layer` still records the full name for provenance), overridable.
- `--input-dir` defaults to `results_dir/beast_frames` — every frame under it (anchors
  and their ±1 context neighbors alike), matching `cuttle predict`'s own default.
  `--predictions-name` defaults to `input_dir.stem`.
- `--batch-size` defaults to 128. `--device` defaults to `cuda` if available else `cpu`.

**Output — duck-typed as a BEAST model directory**, the same precedent as the
classifier's own embedding (`DECISIONS.md`'s "Classifier embeddings" entry):
`cuttle_patterns.latents.load_latents`/`split_latent_spaces` only care about the
on-disk layout and one `config.yaml` key, not that the directory actually came from
`beast train`. So `cuttle reduce --model-name {model_name}`/`cuttle cluster --model-name
{model_name} --n-clusters K`/`cuttle serve`'s Model dropdown all work against a
`cuttle embed` output with no embedder-specific code.

```
results_dir/beast_models/{model_name}/
├── config.yaml                                          # model.model_class: embedder
├── readout_state.pt                                      # stateful readouts only (e.g. gram); provenance, not a cache
└── image_predictions/{predictions_name}/latents/
    ├── embeddings.npy                                    # (N, D) float32, one combined file
    └── manifest.parquet                                  # row-aligned: video_name, frame_number, day, tank, role
```

Unlike `cuttle predict --save-latents` (one `.npy` per frame), `cuttle embed` writes a
single combined `(N, D)` array plus a row-alignment manifest per run, regardless of `N`
— writing millions of tiny files is extremely slow on the external drive `results_dir`
lives on.

`config.yaml` stays minimal, absorbing what would otherwise be a separate sidecar JSON
into `model.model_params`:

```yaml
model:
  model_class: embedder
  model_params:
    backbone: dinov3_vitb16_224
    hf_model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
    readout: cls
    resolution: 224
    embed_dim: 768
    backbone_hidden_size: 768
    code_git_commit: <sha>
    extraction_timestamp: <iso8601>
```

`embed_dim` is the readout's actual output dimensionality (`embedder.dim`);
`backbone_hidden_size` is the backbone's own channel dimension — the two differ for
`gram` (e.g. `embed_dim: 2080`, `backbone_hidden_size: 768`), so they're kept as
separate keys rather than one that only sometimes means the readout's output.

`manifest.parquet`'s row order is exactly `embeddings.npy`'s row order, sorted by
`(video_name, frame_number)`.

## DINOv3 backbone

Loaded via Hugging Face `transformers` (`AutoModel`, resolving to `DINOv3ViTModel`).
`AutoImageProcessor` is not used — preprocessing is our own, so resizing stays under our
control.

| Arch | Hugging Face id | Embed dim `C` (`hidden_size`) |
|---|---|---|
| ViT-S/16 | `facebook/dinov3-vits16-pretrain-lvd1689m` | 384 |
| ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 768 |
| ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` | 1024 |

Weights are gated under Meta's DINOv3 license; the environment needs an HF token
(`hf auth login --token <token>`, persisted to `~/.cache/huggingface/token`) to pull
them.

**Preprocessing.** Input crops are uint8 RGB. Resize the full (non-square — `cuttle
inscribe`'s canonical crop is 200×100 by default) crop to `res × res` with bicubic
interpolation, no center crop (that would remove exactly the border region the leakage
diagnostics care about). `res` must be a multiple of 16 (the patch size). Normalize with
ImageNet mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)` — the LVD-1689M
web-image constants (the satellite-imagery DINOv3 variants use different ones and aren't
used here). This is the same non-aspect-preserving resize-to-square treatment BEAST's
own dataset pipeline already applies to these crops for the existing AE/MSPS-VAE
backbones.

**Token layout.** DINOv3 ViTs emit one CLS token, `num_register_tokens` register tokens
(4 for all three released ViTs above, read from the model config rather than hardcoded),
then patch tokens in row-major grid order:
`embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)`.
`last_hidden_state` from a normal `AutoModel` forward pass is already post-final-norm
(`DINOv3ViTModel` applies its own final `norm` before returning). Register tokens are
never used by any readout.

## Readouts

- **`cls`** — the post-norm CLS token, unchanged, shape `(B, C)`.
- **`meanpatch_uniform`** — the unweighted mean of post-norm patch tokens over all `N`
  grid positions, shape `(B, C)`.
- **`meanpatch_taper`** — the same mean, but weighted by a raised-cosine radial taper
  that down-weights border/corner patches (see "Spatial weights" below), shape `(B, C)`.
- **`gram`** — a spatially weighted, channel-projected covariance texture descriptor;
  see below.

Mean-patch readouts are expected to suit texture better than CLS, since CLS is trained
toward object-level semantics and every frame here is the same "object" — both are kept
so that expectation can be checked, not assumed.

### Spatial weights

The taper used by `meanpatch_taper` and `gram` (`cuttle_patterns/embedders/spatial_weights.py`)
replicates the same raised-cosine radial taper used for `use_spatial_loss_weight` in the
masked MSPS-VAE (`build_raised_cosine_weight_map` in
`beast/models/msps_vae/msps_vae_model.py`, `msps-vae` branch), evaluated analytically at
patch-center coordinates instead of a pixel grid: flat at weight 1 out to normalized
radius `r0` (default 0.5, matching `spatial_loss_weight_r0`'s default), cosine-tapers to
exactly 0 at `r = 1`, and is exactly 0 beyond — corners fall past the taper's zero point
with no separate corner logic needed. Cross-checked pointwise against the real `beast`
function in `tests/embedders/test_spatial_weights.py`.

## Gram readout

For each frame, treat the `N` patch tokens (each a `C`-dimensional vector) as samples,
and compute their spatially weighted covariance across channels: which feature
directions co-vary across positions within the frame. Summing over positions discards
*where* features occur and keeps *which* features co-occur — a texture descriptor
that's position-invariant by construction (the Gatys-style idea, applied to ViT tokens
instead of CNN feature maps).

**Per-frame pipeline:**
1. Spatial weights `w` over the patch grid (`uniform` or `taper`), summing to 1.
2. Project tokens from `C` channels to `k` channels with a fixed, fitted projection `P`.
3. Weighted, mean-centered covariance `G` (`k × k`) across the projected tokens.
4. Matrix square root of `G`.
5. Vectorize the upper triangle, with off-diagonal entries scaled by `√2`.

Output dimensionality is `k * (k + 1) / 2` (e.g. 2,080 for `k=64`). All of this runs in
float64; the backbone's output is cast up before the readout, and the final vector is
cast back to float32.

**Fitting `P`.** The projection is fit on the *within-frame* variation: for each
fit-set frame, center its (weighted) patch tokens by that frame's own weighted mean,
then average the resulting `C × C` weighted covariances across the whole fit set. `P`'s
rows are the top-`k` eigenvectors of that averaged covariance. Fitting on within-frame
variation (rather than a PCA on tokens pooled across frames) avoids the projection being
dominated by *between*-frame shifts in mean activation — lighting, individual, session —
the known confounds. `variance_retained` (the fraction of non-negative eigenvalue mass
the top `k` directions capture) is recorded in `config.yaml` as a fit diagnostic.

**Design caveats:**
- **Covariance, not correlation.** Computed on raw (projected) token values, never
  per-channel-standardized ones — covariance-based distances are invariant to rotating
  the channel basis; correlation-based ones aren't.
- **Centered.** An uncentered Gram is dominated by the outer product of the mean token,
  re-encoding first-order statistics that belong to a different kind of readout
  (`meanpatch_*`) — the mean is discarded here, not folded in.
- **Rotation loses nothing; truncation can.** Projecting onto principal components is a
  change of basis plus truncation. The change of basis alone is harmless (covariance →
  square root → Frobenius distance gives identical distances in any orthonormal basis);
  what can lose information is dropping low-variance directions, since variance isn't
  the same as relevance to pattern type. `k` is a real hyperparameter to sanity-check
  against the full untruncated Gram if results look off.
- **`√2` on off-diagonals.** Vectorizing only the upper triangle counts each
  off-diagonal entry once, but the Frobenius norm counts it twice; scaling by `√2` makes
  Euclidean distance on the vector equal Frobenius distance on the matrix.
- **Matrix square root.** Raw covariance matrices have poorly behaved geometry (a few
  large eigenvalues dominate Euclidean distance); following the bilinear-pooling
  literature (Lin, RoyChowdhury & Maji 2015; Lin & Maji 2017), the readout takes the
  matrix square root of `G` before vectorizing.
- **Effective sample size.** A `k × k` covariance estimated from a 14×14 or 28×28 grid
  is noisy — report `n_eff = 1 / Σw²` if this needs auditing later; prefer 448px
  resolution (784 tokens) over 224px (196) for less noisy Gram estimates.
- **Final-layer ViT tokens aren't local texture features.** Unlike CNN feature maps,
  final-layer ViT patch tokens have passed through many layers of global attention, so
  each token already mixes in whole-image context — a Gram over these tokens is a
  second-order summary of *contextualized* features, not a pure Gatys-style texture
  statistic. This is the main motivation for the VGG-19 backbone below.
- **Layout blindness.** Any Gram descriptor discards spatial arrangement entirely. Some
  cuttlefish pattern distinctions are partly layout (disruptive components in
  characteristic body locations) while others are closer to stationary texture (mottle)
  — expect Gram to separate textural distinctions well and possibly blur
  layout-defined ones.

**Not implemented:** the fuller design considered during planning also included an
optional first-order mean block (concatenating the taper-weighted mean patch token onto
the covariance vector) and covariance shrinkage (toward a scaled identity). Neither is
built — the mean block would reintroduce exactly the low-level, cheap-shortcut signal
(average activation level) this readout is designed to exclude, and is largely redundant
with `meanpatch_taper` anyway; shrinkage was never turned on even in the original
design's own suggested defaults. Both would be small, self-contained additions if ever
wanted.

**Reference math** (`cuttle_patterns/embedders/gram_math.py`, `spatial_weights.py`):

```python
def patch_center_coords(grid_h: int, grid_w: int) -> torch.Tensor:
    """(N, 2) patch-center coordinates in [-1, 1], row-major, matching token order."""
    ys = (torch.arange(grid_h, dtype=torch.float64) + 0.5) / grid_h * 2 - 1
    xs = (torch.arange(grid_w, dtype=torch.float64) + 0.5) / grid_w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1)


def patch_weights(grid_h: int, grid_w: int, kind: str, **taper_kwargs) -> torch.Tensor:
    """(N,) non-negative weights summing to 1."""
    if kind == "uniform":
        w = torch.ones(grid_h * grid_w, dtype=torch.float64)
    elif kind == "taper":
        w = radial_taper(patch_center_coords(grid_h, grid_w), **taper_kwargs)
    else:
        raise ValueError(kind)
    return w / w.sum()


def weighted_moments(X: torch.Tensor, w: torch.Tensor):
    """X: (B, N, C), w: (N,) summing to 1. Returns mu (B, C), G (B, C, C)."""
    mu = torch.einsum("bnc,n->bc", X, w)
    Xc = X - mu[:, None, :]
    G = torch.einsum("bnc,bnd,n->bcd", Xc, Xc, w)
    return mu, G


def psd_sqrt(G: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(G)
    s = evals.clamp_min(0).sqrt()
    return (evecs * s[..., None, :]) @ evecs.transpose(-1, -2)


def sym_to_vec(S: torch.Tensor) -> torch.Tensor:
    """Upper triangle incl. diagonal, off-diagonals scaled by sqrt(2)."""
    k = S.shape[-1]
    iu = torch.triu_indices(k, k, device=S.device)
    v = S[..., iu[0], iu[1]].clone()
    v[..., iu[0] != iu[1]] *= math.sqrt(2.0)
    return v   # (..., k(k+1)/2)
```

Per-frame forward:

```python
def gram_readout_forward(patches, grid_hw, P, weights):
    X = patches.double() @ P.T                    # (B, N, k)
    w = patch_weights(*grid_hw, kind=weights)      # (N,)
    _, G = weighted_moments(X, w)
    return sym_to_vec(psd_sqrt(G)).float()          # (B, k(k+1)/2)
```

## VGG-19 backbone

The main motivation for a second backbone: final-layer ViT patch tokens have passed
through many layers of global attention, so a Gram over them is a second-order summary
of *contextualized* features, not a pure Gatys-style texture statistic the way VGG conv
activations are (see "Final-layer ViT tokens aren't local texture features" above). If
DINOv3-Gram and VGG-Gram disagree, that difference is the first thing to investigate.

`cuttle_patterns/embedders/vgg.py` wraps a pretrained `torchvision.models.vgg19`
(`IMAGENET1K_V1` weights — no gating/token needed, unlike DINOv3), truncated at one
named layer. The Gram readout math above carries over completely unchanged: VGG's conv
feature map `(B, C, H, W)` is flattened to a `TokenOutput` with `patches` shape
`(B, H*W, C)` and `grid_hw = (H, W)`, the same shape a ViT backbone produces. `cls` is
`None` — VGG has no CLS-token equivalent, so `build_embedder` rejects
`--backbone vgg19 --readout cls` up front, before loading any weights.

`--vgg-layer` is restricted to the canonical 5 Gatys et al. texture/style layers (*A
Neural Algorithm of Artistic Style*, CVPR 2016: one ReLU per block, equally weighted) —
this is the most-cited convention for Gram-matrix texture representations, and it keeps
receptive field growing monotonically with depth:

| Layer | Key code | `vgg19().features` index | Channels `C` | Downsampling |
|---|---|---|---|---|
| `relu1_1` | `1` | 1 | 64 | 1× (no pooling yet) |
| `relu2_1` | `2` | 6 | 128 | 2× |
| `relu3_1` | `3` | 11 | 256 | 4× |
| `relu4_1` | `4` | 20 | 512 | 8× |
| `relu5_1` | `5` | 29 | 512 | 16× |

The "Key code" column is what actually appears in `backbone.key`/the default
`model_name` (`LAYER_TO_CODE` in `vgg.py`) — kept short so
`vgg19_3_448_gram_k64_taper` stays readable; `metadata()`'s `vgg_layer` field still
records the full layer name (`relu3_1`) for provenance. Multi-layer fusion concatenates
these codes in block order (e.g. `vgg19_345_...` for `relu3_1`+`relu4_1`+`relu5_1`) --
see "VGG-19 Gram fusion" below.

`--resolution` must be a multiple of the chosen layer's downsampling factor, so its
output grid divides evenly (analogous to DINOv3's "multiple of the patch size"
constraint). Preprocessing reuses DINOv3's resize/ImageNet-normalization pipeline
unchanged — torchvision's pretrained weights expect the same ImageNet statistics. VGG is
fully convolutional and cheap per-pixel relative to a ViT, so `cuttle embed` defaults
`--resolution` to 448 for `vgg19` (versus 224 for DINOv3) — a less noisy Gram covariance
estimate per this doc's own resolution caveat above, still affordable at VGG's cost per
pixel.

## VGG-19 Gram fusion

`--vgg-layer` accepts a comma-separated list (e.g. `relu3_1,relu4_1,relu5_1`), only with
`--readout gram` (fusion for `cls`/`meanpatch_*` was never requested and isn't
supported). `cuttle_patterns.embedders.vgg.MultiLayerVGGBackbone` runs VGG-19's shared
trunk exactly once per batch — not once per layer — by slicing `features` into
contiguous segments between consecutive requested layers and chaining them, returning
one `TokenOutput` per layer (canonical block order) instead of a single one.
`cuttle_patterns.embedders.readouts.FusedGramReadout` pairs with it: one independently-fit
`GramReadout` per layer (same `k`/`weights`, shared across all of them, not per-layer
knobs — first-pass simplicity, not a design ceiling), concatenated in canonical order.
`--vgg-layer`'s values are canonicalized (deduplicated, sorted to block order) regardless
of input order, so `relu4_1,relu3_1` and `relu3_1,relu4_1` name and behave identically.

**Model naming.** The backbone key concatenates each layer's short digit code (e.g.
`vgg19_345_448_gram_k64_taper` for `relu3_1`+`relu4_1`+`relu5_1`) — the readout's own
name (`gram_k64_taper`) doesn't change between single-layer and fused, since layer
identity lives entirely in the backbone key.

**Memory.** Each block halves the spatial grid area while only doubling channels, so a
layer's `N × C` product (what drives the Gram readout's float64 tensor size) roughly
halves at each deeper layer — summing all 5 canonical layers comes to only ~1.9x the
memory of `relu1_1` alone, not 5x, and `relu1_1` alone is already what today's
`VGG_BATCH_SIZE` override (`cuttle_patterns/cli/cmd_embed.py`) is sized against. Getting
close to that ~1.9x figure in practice (not worse) requires processing one layer's Gram
computation at a time and dropping it before moving to the next, which is exactly what
`FusedGramReadout.partial_fit`/`__call__` do (`list.pop`, not just a loop over an
already-materialized list — see their docstrings). The backbone's own raw (float32,
pre-Gram) activations are comparatively cheap (~3 GiB total across all 5 layers at
`VGG_BATCH_SIZE`) and are allowed to coexist momentarily as the shared trunk runs
forward; only each layer's own *float64 Gram math* — the expensive part — is kept to
one layer at a time.

**Not implemented: per-layer `k`/`weights`, and a second reduction stage over the
concatenated vector.** Fusing all 5 canonical layers at the default `k=64` concatenates
to roughly 10k dimensions (`5 × 64·65/2`) — large enough that a second, fused-level PCA
(fit once on the concatenated vectors, distinct from each layer's own within-frame
channel projection) is worth considering once fusion has actual results to evaluate
against; deliberately not built ahead of that, mirroring why `final_pca_dim` was cut
from the single-layer Gram readout above. Lowering the shared `--gram-k` (e.g. to `28`,
landing the fused total near a single layer's own ~2k) is the simpler alternative,
tried first.
