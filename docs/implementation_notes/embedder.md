Embeddings implementation

Implementation spec for the embedders that feed the evaluation harness described in [eval_plan.md](eval_plan.md). It covers three things: the embedder protocol every backbone implements, a DINOv3 embedder with CLS-token and mean-pooled patch-token readouts, and a Gram-matrix (second-order texture) readout computed on DINOv3's final-layer patch tokens. VGG-19 Gram features are a planned follow-up and are sketched at the end so the design leaves room for them, but they are out of scope for the first implementation.

This is Tier B of the harness — a new, frozen-pretrained-backbone tier. It's implemented as a general CLI pipeline stage, `cuttle embed` (see "CLI: `cuttle embed`" in section 1), not as eval-harness-internal code — none of `cuttle_patterns/embed.py`, `cuttle_patterns/embedders/`, or `cuttle_patterns/cli/cmd_embed.py` exist yet; today's `eval/` package only implements Tier A (loading already-trained BEAST checkpoint latents, see `eval/load_embeddings.py`). This doc's `Backbone`/`Readout`/`Embedder` split supersedes the placeholder `class Embedder: id/preprocess/embed` sketch in eval_plan.md's "Embedders" section — that sketch predates this design and should be updated to point here rather than kept as a second, conflicting protocol.

Sections were originally marked (verify) for details stated from memory or assumed about the repo. Most have now been checked directly against the repo/environment and are updated in place below with what was actually found; "Ambiguities to resolve against the codebase" (section 5) is updated to show what's resolved and what's still genuinely open.

**Implementation order (2026-09-20):**

1. **Done (2026-09-20):** DINOv3 backbone, CLS readout only, `cuttle embed` wired into the CLI, writing the single-combined-array duck-typed output described in "CLI: `cuttle embed`" below. Implemented as `cuttle_patterns/embedders/{base,dinov3,readouts}.py` + `cuttle_patterns/embed.py` + `cuttle_patterns/cli/cmd_embed.py`, tested (`tests/embedders/`, `tests/test_embed.py`, `tests/cli/test_cmd_embed.py`) and smoke-tested end to end against the real `facebook/dinov3-vits16-pretrain-lvd1689m` weights on synthetic frames (correct `(N, 384)` output, valid `config.yaml`/`manifest.parquet`). The `transformers` import is deferred to inside `DINOv3Backbone.__init__` (not at module top), since eagerly importing it would otherwise add ~2s to every `cuttle` invocation via `cli/main.py`'s eager `cmd_*.py` auto-discovery, not just `cuttle embed`'s. Pausing here to commit before starting step 2.
2. **Done (2026-09-20):** `cuttle_patterns.latents.load_latents` now dispatches on whether `embeddings.npy` exists in `latents_dir` — if so, reads it plus its row-aligned `manifest.parquet` (`_load_combined_latents`); otherwise falls back to the original per-frame-`.npy` glob, unchanged for real `beast train`/`cuttle predict` output. This was, as expected, the only change needed: `cuttle reduce`/`cuttle cluster` themselves needed zero edits, since both already go through `load_latents`/`split_latent_spaces` and never touch the filesystem layout directly. Verified against the real full-dataset `dinov3_vitb16_224_cls` run (94,361 frames × 768-d, 32 videos) — `cuttle reduce` and `cuttle cluster --n-clusters 16` both completed successfully against it.
3. **Done (2026-09-20):** `meanpatch_uniform` readout added (`MeanPatchUniformReadout` in
   `cuttle_patterns/embedders/readouts.py`) — unweighted mean of post-norm patch tokens
   over all N grid positions, stateless like `cls` (`fit_passes = 0`, no `partial_fit`
   needed). Reachable via `cuttle embed --readout meanpatch_uniform` with no other CLI or
   `embed.py` changes, since `build_embedder`/`write_embedder_output` were already
   readout-agnostic (dispatch through `READOUTS_BY_NAME`). Tested in
   `tests/embedders/test_readouts.py`.
4. **Done (2026-09-20):** taper spatial weights (`patch_center_coords`/`radial_taper`/
   `patch_weights` in the new `cuttle_patterns/embedders/spatial_weights.py`) and the
   `meanpatch_taper` readout (`MeanPatchTaperReadout` in `readouts.py`) built.
   `radial_taper` replicates `build_raised_cosine_weight_map` from
   `beast/models/msps_vae/msps_vae_model.py` (msps-vae branch) exactly, evaluated at
   patch centers instead of pixels, and is cross-checked pointwise against that real
   function on a pixel grid in `tests/embedders/test_spatial_weights.py` (imported
   directly — available in the `cuttle` conda env). `patch_weights` renormalizes to
   sum 1 rather than mean 1, per section 3's "Spatial weights" note, since sum-1 is
   what the weighted-moments math needs. Reachable via
   `cuttle embed --readout meanpatch_taper` (default `r0 = 0.5`, matching
   `spatial_loss_weight_r0`'s default); `MeanPatchTaperReadout.metadata()` records the
   `taper_r0` used. Tested in `tests/embedders/test_spatial_weights.py` and
   `tests/embedders/test_readouts.py`.
5. **Next:** the `gram_*` readouts (section 3), which reuse `patch_weights` for their
   own spatial weighting but additionally need the fitted channel-projection machinery
   (`fit_passes = 2`, `partial_fit`/`finalize_pass`) that `cls`/`meanpatch_*` don't
   require.

1. Embedder protocol
Design

An embedder maps a batch of egocentric crops to a batch of fixed-length vectors. The harness only ever calls the public protocol below. Internally, embedders are composed of a backbone (runs the network once and returns tokens or feature maps) and a readout (turns those into a vector).

**Deliberately not optimized: no cross-readout backbone sharing.** An earlier version of this design batched embedders by `backbone_key` so CLS/mean-patch/Gram readouts on the same DINOv3 model would share one forward pass. That's dropped: `cuttle embed` (see "CLI: `cuttle embed`" below) runs one embedder — one backbone + one readout — per invocation, writing its own output directory. Getting CLS, mean-pooled, and Gram readouts for the same frames means running the command three times, each redoing the DINOv3 forward pass. This is a deliberate simplicity-over-compute tradeoff — inference is cheap relative to how often this gets iterated on, and a single-embedder-per-run CLI is much easier to reason about (one `--model-name`, one output directory, one failure mode) than a batch script juggling multiple readouts' fit state and outputs at once. The `Backbone`/`Readout` split below is kept purely as an internal code-factoring device (DINOv3-specific preprocessing/forward logic stays separate from readout logic), not as a batching optimization.

Embedders own their preprocessing. The harness hands them raw crops in one canonical format and never normalizes or resizes on their behalf.

Embedders do not L2-normalize or PCA their outputs for clustering purposes; the harness applies its standardized clustering conventions (L2-normalize, PCA to 64) to every embedding uniformly. The only reductions an embedder applies are ones that are part of its definition (for the Gram readout, the channel projection and optional final PCA described in section 3).

Interface
python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch


# Canonical input: uint8 RGB crops, shape (B, H, W, 3), channel order RGB.
Frames = np.ndarray


@dataclass
class TokenOutput:
    """What a ViT-style backbone returns for one batch."""
    cls: torch.Tensor          # (B, C)
    patches: torch.Tensor      # (B, N, C), N = grid_h * grid_w, row-major
    grid_hw: tuple[int, int]   # (grid_h, grid_w)


class Backbone(ABC):
    key: str                   # e.g. "dinov3_vitb16_448"; shared by all readouts on it

    @abstractmethod
    def preprocess(self, frames: Frames) -> torch.Tensor: ...

    @abstractmethod
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> TokenOutput: ...

    def metadata(self) -> dict: ...   # model id, resolution, normalization, dtype


class Readout(ABC):
    name: str                  # e.g. "cls", "meanpatch_uniform", "gram_k64_taper"
    fit_passes: int = 0        # number of passes over the fit set; 0 = stateless

    def partial_fit(self, tokens: TokenOutput, pass_idx: int) -> None: ...
    def finalize_pass(self, pass_idx: int) -> None: ...

    @abstractmethod
    def __call__(self, tokens: TokenOutput) -> torch.Tensor: ...   # (B, D)

    @property
    @abstractmethod
    def dim(self) -> int: ...

    def state_dict(self) -> dict: return {}
    def load_state_dict(self, state: dict) -> None: ...
    def metadata(self) -> dict: ...


class Embedder:
    """Public protocol used by the harness. Composition of one backbone + one readout."""

    def __init__(self, backbone: Backbone, readout: Readout):
        self.backbone, self.readout = backbone, readout

    @property
    def id(self) -> str:
        return f"{self.backbone.key}_{self.readout.name}"

    @property
    def backbone_key(self) -> str:
        return self.backbone.key

    @property
    def dim(self) -> int:
        return self.readout.dim

    @property
    def requires_fit(self) -> bool:
        return self.readout.fit_passes > 0

    def embed(self, frames: Frames) -> np.ndarray:
        tokens = self.backbone.forward(self.backbone.preprocess(frames))
        return self.readout(tokens).float().cpu().numpy()   # (B, D) float32

`cuttle embed`'s extraction loop just calls `Embedder.embed` batch by batch for the one embedder it was invoked with — there's no multi-embedder batching to bypass it for (see above), so `Embedder.embed` is the real code path here, not just a fallback for tests.

Fitting stateful readouts

Some readouts (the Gram readout) have parameters fit from data. Fitting runs fit_passes passes over a fit set of frames. Each pass calls partial_fit on every batch, then finalize_pass. Multiple passes are needed when a later statistic depends on an earlier one (for Gram: pass 0 fits the channel projection; pass 1 computes block normalization scalars and the optional final PCA using that projection).

The fit set is a random sample drawn directly from `results_dir/beast_frames/` (any exported frame — anchor or ±1 context neighbor, not restricted to `manifests/extract.parquet`'s selected-anchor set), with a per-video cap, no held-out-video split — this project's eval harness deliberately has no held-out-video concept (see eval_plan.md's "Eval frame set": each `results_dir` already stands alone, so there's nothing to snapshot/hash a split against), and fitting the Gram projection on second-moment statistics only (never labels) makes reusing eval-adjacent frames an acceptable, mild form of shared data rather than real leakage. The fit set is still fixed and versioned like the eval manifest, and its identity (hash or path) is recorded in the readout's state, so a given embedder id always produces identical embeddings once fit.

Caching and output format — one combined array, not one file per frame

**Why this changed:** `cuttle predict --save-latents` (the existing BEAST path) writes one `.npy` per frame, under `latents/{video_name}/img{frame_number}.npy`. On the external hard drive `results_dir` currently lives on, writing millions of tiny files is extremely slow — this is a filesystem problem, not a training-time-cost problem, and it applies just as much to a frozen-backbone embedder as to a trained BEAST checkpoint. `cuttle embed` (below) never writes more than a few files total per run: one combined `(N, D)` float32 array covering every embedded frame, plus a small row-alignment manifest, regardless of how many frames `N` is.

CLI: `cuttle embed`

New `cuttle` subcommand, `cuttle_patterns/cli/cmd_embed.py` (thin argparse wrapper, following the existing `cmd_<name>.py` → real-logic-in-a-top-level-module convention every other subcommand uses), delegating to `cuttle_patterns/embed.py`. Not nested under `cuttle_patterns/eval/` — `eval/` is for the harness's own scoring machinery (manifest joins, clustering, metrics); embedding extraction is a general pipeline stage, parallel to `cuttle predict`, and needs to be reachable from the CLI regardless of whether the eval harness ever runs. `Backbone`/`Readout` implementations live at `cuttle_patterns/embedders/` (e.g. `embedders/dinov3.py`), a new top-level package sibling to `preprocessing/`/`dashboard/`.

```bash
cuttle embed --backbone dinov3_vitb16 --resolution 224 --readout cls
# writes to results_dir/beast_models/dinov3_vitb16_224_cls/ by default;
# pass --model-name to override
```

- `--backbone`/`--resolution`/`--readout` select the embedder (e.g. `dinov3_vits16`/`dinov3_vitb16`/`dinov3_vitl16` × `224`/`448` × `cls` for now — `meanpatch_*`/`gram_*` land later, see "Implementation order" below). Together they determine `embed_dim`.
- `--model-name` defaults to `{backbone}_{resolution}_{readout}` (e.g. `dinov3_vitb16_224_cls` — the same embedder-id scheme used throughout this doc's tables), overridable — unlike `cuttle train`/`cuttle predict`/`scripts/classify_skin_pattern.py`, which all require an explicit `--model-name`, a DINOv3 embedder's identity is already fully determined by its own flags, so there's a sensible default and no need to force one. The default already reads as non-BEAST at a glance (`dinov3_...` vs. `iter-1.1_resnet-18_d16`), matching the spirit of the classifier-embedding naming convention (`DECISIONS.md`'s "Classifier embeddings" entry) without needing to type it out.
- `--input-dir` defaults to `results_dir/beast_frames` — **every** frame under it (anchors and their ±1 context neighbors alike), matching `cuttle predict`'s own default, so a DINOv3 embedder covers the same frame set any BEAST model's latents do and `cuttle reduce`/`cuttle cluster` don't need to treat it specially. `--predictions-name` defaults to `input_dir.stem` (`beast_frames`), matching `cuttle predict`'s existing convention.
- `--batch-size` defaults to 128 (a starting guess, not benchmarked — see section 5's "Compute settings"; easy to override once real throughput numbers are in). `--device` defaults to `cuda` if available else `cpu`, matching `scripts/classify_skin_pattern.py`'s existing `--device` convention.

**Output — duck-typed as a BEAST model directory,** exactly the precedent set for the classifier's own embedding (`DECISIONS.md`'s "Classifier embeddings" entry): `cuttle_patterns.latents.load_latents`/`split_latent_spaces` only care about the on-disk `.npy` layout and one `config.yaml` key, not that the directory actually came from `beast train`, so matching that contract means `cuttle reduce --model-name {model_name}`/`cuttle cluster --model-name {model_name} --n-clusters K`/`cuttle serve`'s Model dropdown all pick a DINOv3 embedder up for free once `load_latents` gains the one new-format branch described below (see "Implementation order").

**Module rename (2026-09-20):** `cuttle_patterns/embeddings.py` was renamed to `cuttle_patterns/latents.py` (and its test file to `tests/test_latents.py`) as part of this work, to avoid a name clash with the new `cuttle_patterns/embed.py` (the `cuttle embed` extraction loop, see above). The old name described what it loads (BEAST-style latent vectors from disk) using the same word (`embeddings`) as the new write-side module for a different concept (running a model to compute embeddings); `latents.py` matches the terminology already used everywhere else in this codebase for the read side (`load_latents`, `LATENT_SPACE_*`, the `latents/` directory itself), so the rename is a better name, not just a disambiguation hack. Every importer was updated (`cluster.py`, `reduce.py`, `eval/load_embeddings.py`, `eval/build_manifest.py`, `cli/cmd_reduce.py`, `cli/cmd_cluster.py`, `scripts/classify_skin_pattern.py`, `scratch/plot_loss_weight_profiles.py`, both eval test conftests, `DECISIONS.md`, `eval_plan.md`); nothing in `cuttle_patterns.embeddings`'s public API changed, only its module path.

```
results_dir/beast_models/{model_name}/
├── config.yaml                                          # model.model_class: embedder
└── image_predictions/{predictions_name}/latents/
    ├── embeddings.npy                                    # (N, D) float32, one combined file
    └── manifest.parquet                                  # row-aligned: video_name, frame_number,
                                                            #   day, tank, role — same columns/sort
                                                            #   order load_latents already returns
```

`config.yaml` stays minimal, matching the classifier precedent, but absorbs what would otherwise be a separate sidecar JSON (backbone id, HF model id/revision, readout name, resolution, normalization constants, code git commit, extraction timestamp) into `model.model_params` — one file, not two, since minimizing file count is the whole point here:

```yaml
model:
  model_class: embedder
  model_params:
    backbone: dinov3_vitb16
    hf_model_id: facebook/dinov3-vitb16-pretrain-lvd1689m
    readout: cls
    resolution: 224
    embed_dim: 768
    code_git_commit: <sha>
    extraction_timestamp: <iso8601>
```

`manifest.parquet`'s row order is exactly `embeddings.npy`'s row order — row `i`'s embedding is frame `i` in the manifest, sorted by `(video_name, frame_number)`, the same sort `load_latents` already applies internally today. `split_latent_spaces` needs no change for this: its existing default branch (`if model_class != 'msps_vae': return {'all': X}`) already handles any non-`msps_vae` class, `embedder` included.

Conventions

Run backbones in eval() mode under torch.no_grad(). Mixed precision (bf16) is fine for the backbone forward pass; all Gram-readout math runs in float64 (see section 3). Outputs are always float32 numpy. Seeds are fixed for anything stochastic (fit-set sampling); forward passes are deterministic.

2. DINOv3 backbone and CLS / mean-patch readouts
Model loading — verified against the live models

Load via Hugging Face transformers (`AutoModel`, resolving to `DINOv3ViTModel` for all three ids below; `AutoImageProcessor` is not used — preprocessing is our own, below, so resizing stays under our control). All three candidate ids were confirmed to exist and be reachable with the project's HF account by downloading just their `config.json` (not the full weights):

Arch	Hugging Face id	Embed dim C (`hidden_size`)
ViT-S/16	facebook/dinov3-vits16-pretrain-lvd1689m	384
ViT-B/16	facebook/dinov3-vitb16-pretrain-lvd1689m	768
ViT-L/16	facebook/dinov3-vitl16-pretrain-lvd1689m	1024

The weights are gated under Meta's DINOv3 license: access is already granted on the project owner's HF account (confirmed by the successful config downloads above). The environment needs an HF token to actually pull weights — set one up with `hf auth login --token <token>`, which persists it to `~/.cache/huggingface/token` (readable by any process on the machine, unlike an exported shell env var, which only lives in the shell it was set in). `transformers==5.13.0` in the `cuttle` conda env has native `dinov3_vit` support (`model_type: dinov3_vit`, config class `DINOv3ViTConfig`, model class `DINOv3ViTModel`) — confirmed working end to end (config load + `AutoConfig`/`AutoModel` resolution) against all three ids above. Per this project's own dependency convention (`CLAUDE.md`: "do not pin versions"), don't hard-pin `transformers` in `pyproject.toml`; instead note the verified-working version here (`5.13.0`) as a floor to bump if an older environment's DINOv3 support turns out incomplete. `torch`, `transformers`, and `huggingface_hub` all need adding to `pyproject.toml`'s `dependencies` — none are currently listed (the repo's existing `torch` usage comes transitively through `beast-backbones`, which isn't a PyPI-installable dependency either — see the README's install instructions — so this is the first *direct* torch/transformers dependency of `cuttle_patterns` itself). The `torch.hub` loading path from `facebookresearch/dinov3` was not evaluated; the HF path above already works, so there's no reason to add it as an alternative.

Preprocessing

Input crops are uint8 RGB. Resize the full crop to res × res with bicubic interpolation and antialiasing. Do not center-crop: that would silently remove exactly the border region the leakage diagnostics care about, and make DINOv3 embeddings incomparable with the other embedders. res must be a multiple of 16 (patch size). Normalize with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225) — confirmed exactly via `facebook/dinov3-vitb16-pretrain-lvd1689m`'s `preprocessor_config.json` (`image_mean`/`image_std`, `image_processor_type: DINOv3ViTImageProcessorFast`). These are the LVD-1689M web-image constants; the satellite-imagery DINOv3 variants use different constants and are not used here, and weren't queried.

Crops from this pipeline are not square: `cuttle inscribe`'s canonical aligned-crop size defaults to `--aspect 2.0` × `--canonical-height 100`, i.e. 200×100 px (width×height), and both flags are configurable per run, so code must read each frame's actual shape rather than assume a fixed resolution. This is not a new problem to solve here, though — resize the full (non-square) crop to `res × res`, as already specified above, with no center crop. That's exactly what BEAST's own dataset pipeline already does to these same crops before training the existing AE/MSPS-VAE backbones (`image_size: 224` in `configs/beast_resnet_ae.yaml`/`configs/beast_msps_vae.yaml`, applied as a non-aspect-preserving resize-to-square). So this embedder sees geometrically the same treatment Tier A's embeddings are already built on, not a new policy choice.

Resolutions: 224 gives a 14×14 patch grid; 448 gives 28×28. Run both for CLS and mean-patch. For the Gram readout, 448 is the default (see "Effective sample size" in section 3).

Token layout — verified against transformers' `modeling_dinov3_vit.py`

DINOv3 ViTs emit one CLS token, a small number of register tokens (4 for all three released ViTs above — confirmed via `config.num_register_tokens`, not just the default assumption), then the patch tokens in row-major grid order — confirmed directly in `Dinov3ViTEmbeddings.forward`: `embeddings = torch.cat([cls_token, register_tokens, patch_embeddings], dim=1)`. The backbone must:

Read num_register_tokens from the model config rather than hard-coding it (confirmed 4 for S/16, B/16, L/16 — but still read it per-model, not assumed).
Slice cls = hidden[:, 0], skip the registers, patches = hidden[:, 1 + n_reg:] — matches the concatenation order above exactly.
Assert patches.shape[1] == (res // 16) ** 2.
last_hidden_state **is** post-final-norm, confirmed by reading the source: the bare encoder's own output is pre-norm, but `DINOv3ViTModel` (the class `AutoModel.from_pretrained(...)` actually returns) wraps it and applies `sequence_output = self.norm(output.last_hidden_state)` before returning its own `last_hidden_state`. So a normal `AutoModel` forward pass already gives post-norm tokens — no extra explicit final-norm call is needed in our backbone code.

Register tokens are never used by any readout.

Readouts

cls: returns the post-norm CLS token, shape (B, C).

meanpatch_uniform: returns the mean of post-norm patch tokens over all N positions, shape (B, C).

meanpatch_taper: returns the taper-weighted mean of patch tokens, using the same patch-grid weights as the Gram readout (section 3, "Spatial weights"). This is the first-order counterpart of the Gram readout and a direct test of whether down-weighting border patches alone reduces leakage sensitivity.

Mean-patch is expected to suit texture better than CLS, since CLS is trained toward object-level semantics and every frame here is the same "object." Both are kept because that expectation should be tested, not assumed.

Planned follow-up (not in first implementation): an intermediate-layer mean-patch readout (e.g. block 8 of 12 for ViT-B) via output_hidden_states=True. The TokenOutput dataclass can grow an optional hidden_states field when this is added.

Initial embedder set
Embedder id	Backbone	Readout	Dim
dinov3_vits16_224_cls	ViT-S/16 @ 224	CLS	384
dinov3_vits16_224_meanpatch_uniform	ViT-S/16 @ 224	mean-patch	384
dinov3_vitb16_224_cls	ViT-B/16 @ 224	CLS	768
dinov3_vitb16_224_meanpatch_uniform	ViT-B/16 @ 224	mean-patch	768
dinov3_vitb16_448_cls	ViT-B/16 @ 448	CLS	768
dinov3_vitb16_448_meanpatch_uniform	ViT-B/16 @ 448	mean-patch	768
dinov3_vitb16_448_meanpatch_taper	ViT-B/16 @ 448	taper mean-patch	768

Gram embedders on the same backbone are listed in section 3.

3. Gram readout on DINOv3 final-layer patch tokens
What it computes

For each frame, treat the N patch tokens (each a C-dimensional vector) as samples, and compute their spatially weighted covariance across channels: which feature directions co-vary across positions within the frame. Summing over positions discards where features occur and keeps which features co-occur: a texture descriptor that is position-invariant by construction. This is the Gatys-style idea, applied to ViT tokens instead of CNN feature maps for this first implementation.

The per-frame pipeline is:

Spatial weights w over the patch grid (taper or uniform), summing to 1.
Project tokens from C channels to k channels with a fixed, fitted projection P.
Weighted mean mu (k) and weighted centered covariance G (k×k).
Optional shrinkage of G.
Matrix square root of G.
Vectorize the upper triangle, with off-diagonal entries scaled by √2.
Optionally concatenate mu, with fixed block-normalization scalars.
Optionally apply a fitted final PCA.

Every design choice in this pipeline follows from a specific caveat, collected here.

Caveats that shape the design

Covariance, not correlation. The Gram is computed on raw (projected) token values, never on per-channel-standardized ones. Covariance-based distances are invariant to rotating the channel basis; correlation-based ones are not, and would make the result depend on the arbitrary choice of basis. Do not add per-channel standardization anywhere in this readout.

Centered, with the mean kept separately. An uncentered Gram is dominated by the outer product of the mean token, which re-encodes first-order statistics. The readout centers per frame and treats the mean as a separate, optional block. (VGG activations are non-negative after ReLU, which makes this especially important there; ViT tokens are signed, but the mean term can still dominate, so the same policy applies.)

Rotation loses nothing; truncation can. Projecting onto principal components is a change of basis plus truncation. The change of basis alone is harmless: with an orthogonal Q, ‖QAQᵀ − QBQᵀ‖_F = ‖A − B‖_F, and √(QGQᵀ) = Q √G Qᵀ, so the whole covariance → square root → Frobenius distance pipeline gives identical distances in any orthonormal basis. What can lose information is dropping low-variance directions, because variance is not the same as relevance to pattern type. The truncation check below measures this directly.

Fit the projection on within-frame variation. The Gram only sees within-frame (per-frame-centered) variation. A PCA on pixels pooled across frames would instead be dominated by between-frame shifts in mean activation, which are driven largely by lighting, individual and session: the known confounds. So the projection is fit on the average weighted within-frame covariance: for each fit frame, center tokens by that frame's weighted mean, then average the weighted covariances across frames. With C ≤ 1024 this can be accumulated exactly (a C×C running sum), so no pixel subsampling is needed.

√2 on off-diagonals. Vectorizing only the upper triangle counts each off-diagonal entry once, but the Frobenius norm counts it twice. Off-diagonal entries are multiplied by √2 so Euclidean distance between vectors equals Frobenius distance between matrices. Without this, the basis starts to matter slightly and distances are distorted.

Matrix square root normalization. Raw covariance matrices have poorly behaved geometry: a few large eigenvalues dominate Euclidean distance. Following the bilinear pooling literature (Lin, RoyChowdhury & Maji 2015; Lin & Maji 2017), the readout takes the matrix square root of G before vectorizing. This is part of the embedder definition, not an option.

Effective sample size. A k×k covariance estimated from a 14×14 grid (196 tokens, fewer effectively under the taper) is noisy: k = 64 means 2,080 parameters. Report the effective sample size n_eff = 1 / Σ w² in the readout metadata. Default the Gram backbone to 448 px (784 tokens), and include k = 32 alongside k = 64. Optional shrinkage toward a scaled identity is available but off by default: G ← (1 − α) G + α · (tr G / k) · I.

Spatial weights. Location is discarded by the sum, but border patches still contribute to the statistics in proportion to their area, so rectangle-edge leakage still enters the descriptor. The taper variant weights each patch by the same raised-cosine radial taper used for `use_spatial_loss_weight` in the masked MSPS-VAE, evaluated analytically at patch-center coordinates (not by resizing a pixel-resolution mask). A uniform-weight variant is kept so the harness's inset-crop test can show how much the taper helps.

**Located:** `build_raised_cosine_weight_map(side, r0)` in `beast/models/msps_vae/msps_vae_model.py`, on the `msps-vae` branch of `~/Dropbox/github/paninski-lab/beast` (a separate repo/branch, not part of this codebase). It builds a full `(side, side)` pixel-grid weight map — `r = hypot(x - center, y - center) / (side / 2)` with `center = (side - 1) / 2` — flat at 1.0 for `r <= r0`, the same cosine taper to exactly 0 at `r = 1`, and exactly 0 beyond, then renormalizes so the map's mean is 1.0. `spatial_loss_weight_r0` defaults to 0.5 (`DEFAULT_SPATIAL_LOSS_WEIGHT_R0` in the same file, matching `configs/beast_msps_vae.yaml`'s `spatial_loss_weight_r0: 0.5` here). Since `cuttle_patterns` deliberately doesn't import `beast` internals as a library at runtime (only as a CLI subprocess — see the "cuttle train/predict: subprocess wrappers" decision in `DECISIONS.md`), `radial_taper` below replicates this exact formula rather than importing it, evaluated at `patch_center_coords`' continuous `(y, x)` values instead of a pixel grid (same `r`-normalization convention, so the same `r0` means the same thing at any resolution). The mean-1 renormalization only matters for the loss-weighting use case (keeping MSE's scale against a fixed triplet weight) — for `patch_weights` here, the final `w / w.sum()` already normalizes to sum-1 for the weighted-moments math, so the mean-1 step is dropped, not replicated. A unit test (`tests/eval/embedders/test_gram.py` or similar) should import `beast.models.msps_vae.msps_vae_model.build_raised_cosine_weight_map` (available in the `cuttle` conda env, since `beast-backbones` is already an existing runtime dependency of this whole repo per the README) and assert it agrees with `radial_taper` pointwise on a real pixel grid, evaluated at pixel centers rather than patch centers, for the same `r0`.

Final-layer ViT tokens are not local texture features. Unlike CNN feature maps, final-layer ViT patch tokens have passed through many layers of global attention, so each token already mixes in whole-image context. A Gram over these tokens is therefore not a pure texture statistic in the Gatys sense; it is a second-order summary of contextualized features. This is the main reason VGG-19 remains a planned follow-up: if DINOv3-Gram and VGG-Gram disagree, this difference is the first suspect.

Layout blindness. Any Gram descriptor discards spatial arrangement entirely. Some cuttlefish pattern distinctions are partly layout (disruptive components in characteristic body locations) while others are closer to stationary texture (mottle). Expect Gram embeddings to separate textural distinctions well and possibly blur layout-defined ones; compare fixed-query neighbor grids for disruptive frames specifically.

Reference implementation of the core math

All of this runs in float64. Token tensors can be cast from bf16 after the backbone.

python
import math
import torch


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
        # Replicates beast/models/msps_vae/msps_vae_model.py's
        # build_raised_cosine_weight_map exactly, evaluated at patch centers instead of
        # pixels -- see section 3's "Spatial weights" for the located formula and the
        # cross-check test against beast's own function. Placeholder signature:
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


def shrink(G: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return G
    k = G.shape[-1]
    scale = G.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
    return (1 - alpha) * G + alpha * scale * torch.eye(k, dtype=G.dtype, device=G.device)
Fitting (two passes over the fit set)

Pass 0: channel projection. For each batch, compute the full-C weighted covariance with weighted_moments and add it to a running C×C sum (and a frame count). At the end, Σ_within = sum / count; eigendecompose; P = top-k eigenvectors as rows, shape (k, C). Record the fraction of within-frame variance retained, sum(top-k eigenvalues) / sum(all non-negative eigenvalues), in metadata.

Pass 1: block scalars and optional final PCA. Compute per-frame mu and vec(√G) in the projected space for all fit frames (a few thousand frames × 2,080 dims fits in memory). Set each block's normalization scalar so that the block's mean squared norm over the fit set is 1; this is what keeps the mean block from being swamped by, or swamping, the covariance block when both are concatenated. If final_pca_dim is set, fit a PCA on the normalized, concatenated fit-set vectors (centered, no whitening) and store its mean and components.

Suggested fit-set size: 3–5k frames sampled at random from `results_dir/beast_frames/` (any exported frame, per-video capped — see "Fitting stateful readouts" in section 1 for why no held-out split is needed). Fitted state (P, eigenvalues, block scalars, final PCA, fit-set hash, all hyperparameters) is saved via state_dict.

Per-frame forward
python
def gram_readout_forward(patches, grid_hw, state, cfg):
    X = patches.double() @ state["P"].T                      # (B, N, k)
    w = patch_weights(*grid_hw, kind=cfg.weights)            # (N,)
    mu, G = weighted_moments(X, w)
    G = shrink(G, cfg.shrink_alpha)
    cov_vec = sym_to_vec(psd_sqrt(G)) * state["cov_scale"]   # (B, k(k+1)/2)
    if cfg.include_mean:
        out = torch.cat([cov_vec, mu * state["mean_scale"]], dim=-1)
    else:
        out = cov_vec
    if cfg.final_pca_dim is not None:
        out = (out - state["pca_mean"]) @ state["pca_components"].T
    return out.float()

Note that mu here is the taper-weighted mean patch token projected to k dims. On its own it is nearly redundant with meanpatch_taper, so it is only offered as a block concatenated with the covariance, not as a separate embedder.

Output size and storage

With k = 64 the covariance block is 2,080 dims; with k = 32, 528 dims. For the ~30–50k-frame eval manifest, store full vectors (no final PCA): 50k × 2,080 × 4 bytes is about 0.4 GB. For any bulk extraction over the full 6.5M-frame dataset, set final_pca_dim = 256, since full vectors would be tens of GB.

Initial Gram embedder set

All on dinov3_vitb16_448 (shares its forward pass with the section 2 embedders).

Embedder id	k	Weights	Mean block	Dim
dinov3_vitb16_448_gram_k64_taper	64	taper	no	2,080
dinov3_vitb16_448_gram_k64_uniform	64	uniform	no	2,080
dinov3_vitb16_448_gram_k32_taper	32	taper	no	528
dinov3_vitb16_448_grammean_k64_taper	64	taper	yes	2,144

Shrinkage off and no final PCA for all of these by default.

Truncation check

Before trusting k = 64, compare against the untruncated Gram on about 2k fit-set-style frames sampled the same way as the fit set itself (see above). With C = 768 the full vec(√G) has 295,296 dims, about 2.4 GB in float32 for 2k frames, which is manageable. For k ∈ {32, 64, 128}, report: the mean relative Frobenius error ‖√G_full − Pᵀ √G_k P‖_F / ‖√G_full‖_F (approximate, since the square root doesn't commute exactly with truncation, but a useful magnitude check), and, more importantly, k-NN overlap@20 between the full and truncated embeddings. If k = 64 preserves most neighborhoods, keep it; otherwise move the default to k = 128 (8,256 dims).

If a projection-free cross-check is wanted later, compact bilinear pooling (Gao et al. 2016; Tensor Sketch or Random Maclaurin) approximates inner products between full Grams with random projections and has no preference for high-variance directions.

Unit tests

These should run on synthetic tensors, without the real backbone.

Rotation invariance: with k = C and a random orthogonal P, pairwise distances between sym_to_vec(psd_sqrt(G)) vectors match those computed with P = I to within 1e-8 (float64).
√2 scaling: Euclidean distance between sym_to_vec(A) and sym_to_vec(B) equals ‖A − B‖_F for random symmetric A, B.
Square root: psd_sqrt(G) @ psd_sqrt(G) ≈ G for random PSD G, including rank-deficient G.
Weights: patch_weights sums to 1, is non-negative, and for the taper is symmetric under flips of the grid.
Token order: on a synthetic TokenOutput whose patch values encode their grid position, patch_center_coords ordering matches the token ordering.
Degenerate frame: a frame with identical tokens at every position yields a zero covariance block without NaNs.
Fit reproducibility: fitting twice on the same fit set yields identical state.
4. Future: VGG-19 Gram (out of scope for now)

The readout math above carries over unchanged; what differs is the backbone. VGG-19 returns multiple conv feature maps (Gatys layers conv1_1 through conv5_1: 64, 128, 256, 512, 512 channels) rather than tokens. The TokenOutput abstraction would generalize to a list of per-layer (B, C, H, W) maps. Layers with ≤ 128 channels need no projection; 256- and 512-channel layers get their own within-frame projection. Spatial weights are evaluated analytically at each layer's cell centers. Each layer is a separate block with its own normalization scalar so higher-channel layers don't dominate. VGG is fully convolutional, so higher-resolution crops can be fed without resizing to 224, provided the resize policy stays fixed so pattern scale remains normalized to body size.

5. Ambiguities to resolve against the codebase

Status as of 2026-09-20: most items below are now resolved by direct inspection of the repo and the live HF/GPU environment; findings are folded into sections 1-3 above. What's left is genuinely new work (not a repo fact to look up) — listed at the end.

**Resolved:**

- **Existing harness interfaces.** No `Embedder` interface, `extract.py`, or cache/sidecar format exists yet in `cuttle_patterns/eval/` — only Tier A (`build_manifest.py`, `load_embeddings.py`, `clustering.py`, `metrics.py`, `report.py`, `run_core.py`; confirmed via `find cuttle_patterns/eval -type f`). eval_plan.md's own Tier B sketch (`class Embedder: id, preprocess, embed`) is an earlier placeholder that predates this doc's `Backbone`/`Readout`/`Embedder` split; this doc's protocol supersedes it, and eval_plan.md's "Embedders" section should eventually be edited to point here instead of keeping a second, simpler sketch (not done as part of this pass — flagged, not fixed).
- **Package location.** Superseded by the `cuttle embed` decision above (2026-09-20): code lives at `cuttle_patterns/embedders/` (backbone/readout implementations) and `cuttle_patterns/embed.py` (the extraction loop, one embedder per run — no batching across readouts), with `cuttle_patterns/cli/cmd_embed.py` as the CLI wrapper — **not** under `cuttle_patterns/eval/`, since this is now a general CLI-wired pipeline stage (like `cuttle predict`), not eval-harness-internal machinery. Output is duck-typed into `results_dir/beast_models/{model_name}/` (see "CLI: `cuttle embed`" in section 1), not a new `results_dir/eval/embeddings/` location — no new `paths.py` constants needed for this beyond what `BEAST_MODELS_RELPATH` already covers. Fitted Gram readout state still needs a location once section 3's readouts are built (step 3 of "Implementation order") — likely alongside `config.yaml` in the same `beast_models/{model_name}/` directory, not decided yet.
- **Frame loading format.** Crops are pre-extracted PNGs, not loaded from video or raw arrays: `results_dir/beast_frames/{video_name}/img{frame_idx:08d}.png`, uint8, one row per frame in `manifests/extract.parquet`'s `image_path` column (an **absolute** path, built via `str(save_dir / f'img{idx:08d}.png')` in `cmd_extract.py`) for the eval frame set specifically. Every existing frame-reading script in this repo (`scripts/plot_cluster_frames.py`, `scratch/plot_loss_weight_profiles.py`) loads via `cv2.imread` (BGR) then `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` — the embedder backbone's `preprocess` should follow the same convention rather than inventing a different loader.
- **Crop shape.** Resolved above (section 2, "Preprocessing"): not square, 200×100 px by default, resize-to-square is already the right policy and matches what BEAST's own pipeline does for Tier A.
- **Taper profile.** Resolved above (section 3, "Spatial weights") — located, formula copied, replication + cross-check-test plan spelled out.
- **DINOv3 model ids and loading path.** Resolved above (section 2) — all three ids confirmed reachable, access already granted, loaded via `transformers.AutoModel` → `DINOv3ViTModel`. Minimum transformers version not pinned in `pyproject.toml` per this project's "don't pin versions" convention (`CLAUDE.md`); `5.13.0` in the `cuttle` conda env confirmed working, noted as a floor.
- **DINOv3 token layout.** Resolved above (section 2, "Token layout") — register count, sequence position, and post-final-norm status all confirmed by reading `modeling_dinov3_vit.py` directly.
- **Normalization constants.** Resolved above (section 2, "Preprocessing") — confirmed via the live `preprocessor_config.json`.
- **Held-out split and fit set.** Resolved above (section 1, "Fitting stateful readouts") — no held-out-video split exists or is being added; the Gram fit set is a random, per-video-capped sample straight from `beast_frames/` (any frame, not just extract.parquet's anchors), per an explicit decision (this project's eval harness has no held-out-video concept by design — see eval_plan.md).

**Still open (real work, not a lookup):**

- **Config schema.** How embedder/readout hyperparameters (`k`, weights, shrinkage, final PCA dim, resolution) are expressed in `configs/eval.yaml` — no such file exists yet (`configs/` only has `beast_resnet_ae.yaml`/`beast_msps_vae.yaml`). Needs a real schema design pass when extraction code is actually written, not just a doc note.
- **Compute settings.** Device is confirmed (single CUDA GPU, RTX 5090, via `nvidia-smi`; bf16 autocast per the existing convention in section 1 is fine on this hardware) but per-backbone/per-resolution batch size and measured throughput at 224/448 are not — that requires actually running the model against real frames, which wasn't done in this pass (this was a docs-only task; benchmarking is implementation work). Do this once `cuttle_patterns/eval/embedders/` is actually being built.
- **Existing embeddings for comparison.** Resolved by the duck-typing decision above, not just flagged as a follow-up: once `load_latents` gains the new-format branch (step 2 of "Implementation order"), a `cuttle embed`-produced directory and a real `beast train` checkpoint are both just `EmbedderSpec`s pointing at a `beast_models/{model_name}` directory — `eval/load_embeddings.py`/`eval/report.py` need **no** Tier-B-specific code at all, the same way the classifier's own embedding needed none. This is a stronger resolution than originally planned (a shared interface via a small adapter) — it's the same interface, unmodified.