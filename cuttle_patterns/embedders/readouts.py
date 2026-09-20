"""Readouts: turn a backbone's `TokenOutput` into a fixed-length vector.

`cls`, `meanpatch_uniform`, `meanpatch_taper`, and `gram` are implemented. `gram`
deliberately omits the optional mean-block/shrinkage knobs described in
`docs/implementation_notes/embedder.md`'s "Gram readout" section -- see its "Not
implemented" note for why (cut deliberately, not forgotten).
"""

import torch

from cuttle_patterns.embedders.base import Readout, TokenOutput
from cuttle_patterns.embedders.gram_math import psd_sqrt, sym_to_vec, weighted_moments
from cuttle_patterns.embedders.spatial_weights import DEFAULT_TAPER_R0, patch_weights

CLS_READOUT_NAME = 'cls'
MEANPATCH_UNIFORM_READOUT_NAME = 'meanpatch_uniform'
MEANPATCH_TAPER_READOUT_NAME = 'meanpatch_taper'
GRAM_READOUT_NAME = 'gram'
GRAM_WEIGHTS_CHOICES = ('uniform', 'taper')


class ClsReadout(Readout):
    """Returns a ViT backbone's post-norm CLS token, unchanged."""

    name = CLS_READOUT_NAME
    fit_passes = 0

    def __init__(self, dim: int):
        """Store this backbone's embedding dimensionality.

        Args:
            dim: the backbone's `hidden_size`, i.e. `tokens.cls`'s last dimension.
        """
        self._dim = dim

    def __call__(self, tokens: TokenOutput) -> torch.Tensor:
        """Return the CLS token as-is.

        Args:
            tokens: one batch's backbone output.

        Returns:
            `tokens.cls`, shape (B, dim).
        """
        return tokens.cls

    @property
    def dim(self) -> int:
        """Output vector dimensionality, matching the backbone's `hidden_size`."""
        return self._dim


class MeanPatchUniformReadout(Readout):
    """Returns the unweighted mean of a ViT backbone's post-norm patch tokens."""

    name = MEANPATCH_UNIFORM_READOUT_NAME
    fit_passes = 0

    def __init__(self, dim: int):
        """Store this backbone's embedding dimensionality.

        Args:
            dim: the backbone's `hidden_size`, i.e. `tokens.patches`'s last dimension.
        """
        self._dim = dim

    def __call__(self, tokens: TokenOutput) -> torch.Tensor:
        """Average the patch tokens over all N grid positions.

        Args:
            tokens: one batch's backbone output.

        Returns:
            mean of `tokens.patches` over dim 1, shape (B, dim).
        """
        return tokens.patches.mean(dim=1)

    @property
    def dim(self) -> int:
        """Output vector dimensionality, matching the backbone's `hidden_size`."""
        return self._dim


class MeanPatchTaperReadout(Readout):
    """Returns the taper-weighted mean of a ViT backbone's post-norm patch tokens.

    Down-weights border/corner patches with the same raised-cosine radial taper used
    for `use_spatial_loss_weight` in the masked MSPS-VAE, so rectangle-inscription
    edge/corner leakage contributes less to the mean than under `meanpatch_uniform`
    -- see "Spatial weights" in `docs/implementation_notes/embedder.md`.
    """

    name = MEANPATCH_TAPER_READOUT_NAME
    fit_passes = 0

    def __init__(self, dim: int, r0: float = DEFAULT_TAPER_R0):
        """Store this backbone's embedding dimensionality and the taper's radius.

        Args:
            dim: the backbone's `hidden_size`, i.e. `tokens.patches`'s last dimension.
            r0: normalized radius at which the taper begins; `0 <= r0 <= 1`.
        """
        self._dim = dim
        self.r0 = r0
        # patch_weights only depends on grid_hw, fixed for a given backbone+resolution;
        # cache it per grid_hw instead of recomputing every batch
        self._weights_by_grid_hw: dict[tuple[int, int], torch.Tensor] = {}

    def _weights_for(self, grid_hw: tuple[int, int], patches: torch.Tensor) -> torch.Tensor:
        """Look up (or compute and cache) this grid's taper weights.

        Args:
            grid_hw: the patch grid shape (grid_h, grid_w).
            patches: this batch's patch tokens, for device/dtype matching.

        Returns:
            weights, shape (grid_h * grid_w,), matching `patches`' device/dtype.
        """
        if grid_hw not in self._weights_by_grid_hw:
            self._weights_by_grid_hw[grid_hw] = patch_weights(*grid_hw, kind='taper', r0=self.r0)
        return self._weights_by_grid_hw[grid_hw].to(device=patches.device, dtype=patches.dtype)

    def __call__(self, tokens: TokenOutput) -> torch.Tensor:
        """Compute the taper-weighted mean of the patch tokens.

        Args:
            tokens: one batch's backbone output.

        Returns:
            weighted mean of `tokens.patches` over dim 1, shape (B, dim).
        """
        w = self._weights_for(tokens.grid_hw, tokens.patches)
        return torch.einsum('bnc,n->bc', tokens.patches, w)

    @property
    def dim(self) -> int:
        """Output vector dimensionality, matching the backbone's `hidden_size`."""
        return self._dim

    def metadata(self) -> dict:
        """Readout identity for provenance: the taper's radius hyperparameter."""
        return {'taper_r0': self.r0}


class GramReadout(Readout):
    """Spatially weighted, channel-projected Gram (covariance) texture descriptor.

    For each frame, projects the backbone's patch tokens from its channel dimension `C`
    down to `k` channels with a fixed, fitted projection, then computes their spatially
    weighted, mean-centered covariance across positions: which feature directions
    co-vary across the frame. Summing over positions discards *where* features occur
    and keeps *which* co-occur -- a texture descriptor that's position-invariant by
    construction. See "Gram readout" in `docs/implementation_notes/embedder.md` for the
    full design and the caveat each step follows from.

    Deliberately does not implement a first-order mean block or covariance shrinkage --
    see that section's "Not implemented" note for why. With no mean block to calibrate a
    second block's scale against, fitting only needs the channel projection itself:
    `fit_passes = 1`.
    """

    fit_passes = 1

    def __init__(self, dim: int, k: int, weights: str):
        """Validate hyperparameters and set up this readout's fit-time accumulators.

        Args:
            dim: the backbone's `hidden_size` `C` -- the channel dimension the
                projection maps *from*, not this readout's own output dimensionality
                (see the `dim` property for that).
            k: the channel projection's target dimensionality; must be `<= dim`.
            weights: `'uniform'` or `'taper'` -- which `patch_weights` kind to spatially
                weight positions by.

        Raises:
            ValueError: if `weights` isn't a valid choice, or `k` exceeds `dim`.
        """
        if weights not in GRAM_WEIGHTS_CHOICES:
            raise ValueError(
                f'unknown gram weights: {weights!r}; choices: {GRAM_WEIGHTS_CHOICES}'
            )
        if k > dim:
            raise ValueError(f'gram k ({k}) cannot exceed the backbone channel dim ({dim})')

        self.name = f'{GRAM_READOUT_NAME}_k{k}_{weights}'
        self.dim_in = dim
        self.k = k
        self.weights = weights
        self.P: torch.Tensor | None = None
        self.variance_retained: float | None = None
        self._weights_by_grid_hw: dict[tuple[int, int], torch.Tensor] = {}
        self._cov_sum: torch.Tensor | None = None
        self._frame_count = 0

    def _patch_weights_for(self, grid_hw: tuple[int, int], patches: torch.Tensor) -> torch.Tensor:
        """Look up (or compute and cache) this grid's float64 spatial weights.

        Args:
            grid_hw: the patch grid shape (grid_h, grid_w).
            patches: this batch's (already-doubled) patch tokens, for device matching --
                `patch_weights` always builds on CPU, but the backbone may run on CUDA.

        Returns:
            weights, shape (grid_h * grid_w,), summing to 1, on `patches`' device.
        """
        if grid_hw not in self._weights_by_grid_hw:
            self._weights_by_grid_hw[grid_hw] = patch_weights(*grid_hw, kind=self.weights)
        return self._weights_by_grid_hw[grid_hw].to(patches.device)

    def partial_fit(self, tokens: TokenOutput, pass_idx: int) -> None:
        """Accumulate this batch's weighted covariance, in the full channel dimension.

        Args:
            tokens: one fit-set batch's backbone output.
            pass_idx: always 0 -- `fit_passes == 1`.
        """
        patches = tokens.patches.double()
        w = self._patch_weights_for(tokens.grid_hw, patches)
        _, G = weighted_moments(patches, w)
        if self._cov_sum is None:
            self._cov_sum = torch.zeros(
                self.dim_in, self.dim_in, dtype=torch.float64, device=G.device,
            )
        self._cov_sum += G.sum(dim=0)
        self._frame_count += patches.shape[0]

    def finalize_pass(self, pass_idx: int) -> None:
        """Eigendecompose the accumulated covariance and keep its top-k eigenvectors.

        Args:
            pass_idx: always 0 -- `fit_passes == 1`.

        Raises:
            RuntimeError: if no fit-set batches were seen (`partial_fit` never called).
        """
        if self._cov_sum is None:
            raise RuntimeError('GramReadout.finalize_pass called with no fit-set batches seen')

        cov_mean = self._cov_sum / self._frame_count
        evals, evecs = torch.linalg.eigh(cov_mean)
        order = torch.argsort(evals, descending=True)
        evals, evecs = evals[order], evecs[:, order]

        self.P = evecs[:, :self.k].T.contiguous()
        total_variance = evals.clamp_min(0).sum()
        top_variance = evals[:self.k].clamp_min(0).sum()
        self.variance_retained = (top_variance / total_variance).item()

    def __call__(self, tokens: TokenOutput) -> torch.Tensor:
        """Project this batch's patch tokens and compute their Gram descriptor.

        Args:
            tokens: one batch's backbone output.

        Returns:
            vectorized upper-triangle covariance square root, shape (B, dim).

        Raises:
            RuntimeError: if this readout hasn't been fit yet.
        """
        if self.P is None:
            raise RuntimeError(
                f'{self.name} must be fit before use -- see cuttle_patterns.embed.fit_readout'
            )

        projected = tokens.patches.double() @ self.P.T
        w = self._patch_weights_for(tokens.grid_hw, projected)
        _, G = weighted_moments(projected, w)
        return sym_to_vec(psd_sqrt(G)).float()

    @property
    def dim(self) -> int:
        """Output vector dimensionality: the vectorized k x k upper triangle."""
        return self.k * (self.k + 1) // 2

    def state_dict(self) -> dict:
        """Fitted state: the channel projection and its variance-retained diagnostic.

        `P` is moved to CPU first, so the provenance file `write_embedder_output` saves
        this into stays loadable on a machine without CUDA, regardless of what device
        fitting actually ran on.
        """
        return {
            'P': self.P.cpu() if self.P is not None else None,
            'variance_retained': self.variance_retained,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted state written by `state_dict`.

        Args:
            state: a dict as returned by `state_dict`.
        """
        self.P = state['P']
        self.variance_retained = state['variance_retained']

    def metadata(self) -> dict:
        """Readout identity for provenance: hyperparameters and a fit diagnostic."""
        return {
            'gram_k': self.k,
            'gram_weights': self.weights,
            'gram_variance_retained': self.variance_retained,
        }


READOUTS_BY_NAME = {
    CLS_READOUT_NAME: ClsReadout,
    MEANPATCH_UNIFORM_READOUT_NAME: MeanPatchUniformReadout,
    MEANPATCH_TAPER_READOUT_NAME: MeanPatchTaperReadout,
    GRAM_READOUT_NAME: GramReadout,
}
