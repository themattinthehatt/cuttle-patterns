"""Readouts: turn a backbone's `TokenOutput` into a fixed-length vector.

`cls`, `meanpatch_uniform`, and `meanpatch_taper` are implemented so far — see
"Implementation order" in `docs/implementation_notes/embedder.md` for the planned
`gram_*` follow-up, which reuses `meanpatch_taper`'s spatial weights.
"""

import torch

from cuttle_patterns.embedders.base import Readout, TokenOutput
from cuttle_patterns.embedders.spatial_weights import DEFAULT_TAPER_R0, patch_weights

CLS_READOUT_NAME = 'cls'
MEANPATCH_UNIFORM_READOUT_NAME = 'meanpatch_uniform'
MEANPATCH_TAPER_READOUT_NAME = 'meanpatch_taper'


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
    -- see "Spatial weights" in `docs/implementation_notes/embedder.md` section 3.
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


READOUTS_BY_NAME = {
    CLS_READOUT_NAME: ClsReadout,
    MEANPATCH_UNIFORM_READOUT_NAME: MeanPatchUniformReadout,
    MEANPATCH_TAPER_READOUT_NAME: MeanPatchTaperReadout,
}
