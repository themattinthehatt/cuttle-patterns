"""Readouts: turn a backbone's `TokenOutput` into a fixed-length vector.

Only `cls` is implemented so far — see "Implementation order" in
`docs/implementation_notes/embedder.md` for the planned `meanpatch_*`/`gram_*` follow-ups.
"""

import torch

from cuttle_patterns.embedders.base import Readout, TokenOutput

CLS_READOUT_NAME = 'cls'


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


READOUTS_BY_NAME = {
    CLS_READOUT_NAME: ClsReadout,
}
