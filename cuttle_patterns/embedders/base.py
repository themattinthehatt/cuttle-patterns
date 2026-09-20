"""Backbone/Readout/Embedder protocol for `cuttle embed`.

An embedder maps a batch of egocentric crops to a batch of fixed-length vectors,
composed of a backbone (runs the network once and returns tokens) and a readout (turns
those tokens into a vector). See `docs/implementation_notes/embedder.md`'s "Embedder
protocol" for the full design, including why `cuttle embed` deliberately runs one
backbone+readout pair per invocation rather than sharing a forward pass across readouts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

# canonical input: uint8 RGB crops, shape (B, H, W, 3), channel order RGB
Frames = np.ndarray


@dataclass
class TokenOutput:
    """What a ViT-style backbone returns for one batch.

    Attributes:
        cls: post-norm CLS token, shape (B, C).
        patches: post-norm patch tokens in row-major grid order, shape (B, N, C),
            N = grid_h * grid_w.
        grid_hw: the patch grid shape (grid_h, grid_w).
    """

    cls: torch.Tensor
    patches: torch.Tensor
    grid_hw: tuple[int, int]


class Backbone(ABC):
    """Runs a pretrained network once per batch and returns its raw tokens.

    Attributes:
        key: identifies this backbone (architecture + resolution), e.g.
            `dinov3_vitb16_224`; shared by every readout built on top of it.
    """

    key: str

    @abstractmethod
    def preprocess(self, frames: Frames) -> torch.Tensor:
        """Resize/normalize raw crops into this backbone's expected input tensor."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> TokenOutput:
        """Run the backbone forward pass. Callers are responsible for `torch.no_grad()`."""

    def metadata(self) -> dict:
        """Backbone identity for provenance (model id, resolution, normalization, dtype)."""
        return {}


class Readout(ABC):
    """Turns one batch's `TokenOutput` into a fixed-length vector.

    Attributes:
        name: identifies this readout, e.g. `cls`, `meanpatch_uniform`, `gram_k64_taper`.
        fit_passes: number of passes a stateful readout needs over a fit set before it
            can be called; 0 for a stateless readout (e.g. `cls`).
    """

    name: str
    fit_passes: int = 0

    def partial_fit(self, tokens: TokenOutput, pass_idx: int) -> None:
        """Accumulate statistics from one fit-set batch during fit pass `pass_idx`."""
        return None

    def finalize_pass(self, pass_idx: int) -> None:
        """Finalize whatever `partial_fit` accumulated during fit pass `pass_idx`."""
        return None

    @abstractmethod
    def __call__(self, tokens: TokenOutput) -> torch.Tensor:
        """Compute this readout's vector for one batch, shape (B, D)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output vector dimensionality D."""

    def state_dict(self) -> dict:
        """Fitted state to persist so this readout reproduces identical output."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore fitted state written by `state_dict`."""
        return None

    def metadata(self) -> dict:
        """Readout identity for provenance (hyperparameters, fit-set info)."""
        return {}


class Embedder:
    """One backbone + one readout: the unit of work for a single `cuttle embed` run."""

    def __init__(self, backbone: Backbone, readout: Readout):
        """Compose a backbone and a readout into one embedder.

        Args:
            backbone: runs the network once per batch.
            readout: turns the backbone's tokens into this embedder's output vector.
        """
        self.backbone = backbone
        self.readout = readout

    @property
    def id(self) -> str:
        """Embedder id used as the default `--model-name`, e.g. `dinov3_vitb16_224_cls`."""
        return f'{self.backbone.key}_{self.readout.name}'

    @property
    def dim(self) -> int:
        """Output vector dimensionality D."""
        return self.readout.dim

    @property
    def requires_fit(self) -> bool:
        """Whether this embedder's readout needs fitting before it can be called."""
        return self.readout.fit_passes > 0

    @torch.no_grad()
    def embed(self, frames: Frames) -> np.ndarray:
        """Run one batch of raw crops through the backbone and readout.

        Args:
            frames: uint8 RGB crops, shape (B, H, W, 3).

        Returns:
            float32 array of shape (B, D).
        """
        tokens = self.backbone.forward(self.backbone.preprocess(frames))
        return self.readout(tokens).float().cpu().numpy()
