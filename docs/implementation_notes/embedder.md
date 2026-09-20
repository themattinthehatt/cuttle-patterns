# Embeddings implementation

Reference for the frozen-pretrained-backbone embedders that feed the
[evaluation harness](eval_plan.md) as Tier B. Covers the `Backbone`/`Readout`/`Embedder`
protocol every embedder implements, a DINOv3 backbone with CLS-token, mean-patch, and
Gram-matrix (second-order texture) readouts, and the `cuttle embed` CLI that runs any of
them over exported frames. VGG-19 Gram features are a planned follow-up, sketched at the
end so the design leaves room for them, but not yet implemented.

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

```bash
cuttle embed --backbone dinov3_vitb16 --resolution 224 --readout cls
# writes to results_dir/beast_models/dinov3_vitb16_224_cls/ by default;
# pass --model-name to override
```

- `--backbone`/`--resolution`/`--readout` select the embedder (`dinov3_vits16`/
  `dinov3_vitb16`/`dinov3_vitl16` × `224`/`448`/any multiple of 16 ×
  `cls`/`meanpatch_uniform`/`meanpatch_taper`/`gram`). Together with the readout's own
  hyperparameters, these determine `embed_dim`.
- `--gram-k`/`--gram-weights {uniform,taper}` configure the `gram` readout (default
  `64`/`taper`; ignored for other readouts) — see "Gram readout" below.
- `--model-name` defaults to `{backbone}_{resolution}_{readout name}` (e.g.
  `dinov3_vitb16_224_cls`, `dinov3_vitb16_224_gram_k64_taper`), overridable.
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
  statistic. This is the main motivation for the VGG-19 follow-up below.
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

## Future: VGG-19 Gram (not yet implemented)

The readout math above carries over unchanged; what differs is the backbone. VGG-19
returns multiple conv feature maps (Gatys layers `conv1_1` through `conv5_1`: 64, 128,
256, 512, 512 channels) rather than tokens. The `TokenOutput` abstraction would
generalize to a list of per-layer `(B, C, H, W)` maps. Layers with ≤128 channels need no
projection; 256- and 512-channel layers get their own within-frame projection. Spatial
weights are evaluated analytically at each layer's cell centers. Each layer is a
separate block with its own normalization scalar so higher-channel layers don't
dominate. VGG is fully convolutional, so higher-resolution crops can be fed without
resizing to 224, provided the resize policy stays fixed so pattern scale remains
normalized to body size.

The main motivation: final-layer ViT tokens have passed through many layers of global
attention, so a Gram over them is a second-order summary of *contextualized* features,
not a pure Gatys-style texture statistic the way VGG conv activations are. If
DINOv3-Gram and VGG-Gram disagree, that difference is the first thing to investigate.
