"""DINOv3 backbone: Hugging Face `transformers` ViTs, CLS + patch tokens.

Loading, token layout, and normalization constants are all verified against the live
models (not assumed) — see "DINOv3 backbone and CLS / mean-patch readouts" in
`docs/implementation_notes/embedder.md`.
"""

import cv2
import numpy as np
import torch

from cuttle_patterns.embedders.base import Backbone, Frames, TokenOutput

PATCH_SIZE = 16

ARCH_TO_HF_ID = {
    'vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
}

# LVD-1689M web-image normalization constants, confirmed against
# facebook/dinov3-vitb16-pretrain-lvd1689m's preprocessor_config.json -- do not reuse
# for the satellite-imagery DINOv3 variants, which use different constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def split_tokens(
    hidden_state: torch.Tensor,
    num_register_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a DINOv3 ViT's token sequence into its CLS and patch tokens.

    Token order is `[cls_token, register_tokens, patch_embeddings]` -- confirmed by
    reading `transformers`' `Dinov3ViTEmbeddings.forward` directly, not assumed. Register
    tokens are skipped; no readout uses them.

    Args:
        hidden_state: a backbone's post-norm output, shape (B, 1 + num_register_tokens
            + N, C).
        num_register_tokens: this model's `config.num_register_tokens`.

    Returns:
        `(cls, patches)`: `cls` has shape (B, C); `patches` has shape (B, N, C).
    """
    cls = hidden_state[:, 0]
    patches = hidden_state[:, 1 + num_register_tokens:]
    return cls, patches


def resize_and_normalize(frames: Frames, resolution: int) -> np.ndarray:
    """Resize raw crops to a square and apply DINOv3's ImageNet normalization.

    Stretches the full (possibly non-square) crop to `resolution x resolution` with no
    center crop, since that would remove exactly the border region leakage diagnostics
    care about -- see "Preprocessing" in `docs/implementation_notes/embedder.md`.

    Args:
        frames: uint8 RGB crops, shape (B, H, W, 3).
        resolution: target square side length, in pixels; must be a multiple of 16.

    Returns:
        float32 array, shape (B, 3, resolution, resolution), channel-first.
    """
    resized = np.stack([
        cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        for frame in frames
    ])
    # IMAGENET_MEAN/STD are plain float64 tuples (kept that way for yaml-safe metadata),
    # so the arithmetic below upcasts to float64; cast back to float32 explicitly
    normalized = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(0, 3, 1, 2).astype(np.float32)


class DINOv3Backbone(Backbone):
    """Wraps a Hugging Face DINOv3 ViT, loaded via `AutoModel`."""

    def __init__(self, arch: str, resolution: int, device: torch.device):
        """Load a DINOv3 ViT and prepare it for inference.

        Args:
            arch: one of `ARCH_TO_HF_ID`'s keys (`vits16`, `vitb16`, `vitl16`).
            resolution: square input side length, in pixels; must be a multiple of 16
                (the patch size).
            device: device to load the model onto.

        Raises:
            ValueError: if `arch` is unknown or `resolution` isn't a multiple of 16.
        """
        if arch not in ARCH_TO_HF_ID:
            raise ValueError(f'unknown DINOv3 arch: {arch!r}; choices: {list(ARCH_TO_HF_ID)}')
        if resolution % PATCH_SIZE != 0:
            raise ValueError(f'resolution must be a multiple of {PATCH_SIZE}, got {resolution}')

        # imported lazily: transformers takes ~2s to import, which would otherwise tax
        # every `cuttle` invocation via cli/main.py's eager cmd_*.py auto-discovery, not
        # just `cuttle embed`
        from transformers import AutoModel

        self.arch = arch
        self.resolution = resolution
        self.device = device
        self.hf_model_id = ARCH_TO_HF_ID[arch]
        self.key = f'dinov3_{arch}_{resolution}'

        self.model = AutoModel.from_pretrained(self.hf_model_id)
        self.model.eval()
        self.model.to(device)
        self.num_register_tokens = self.model.config.num_register_tokens
        self.embed_dim = self.model.config.hidden_size
        self.grid_hw = (resolution // PATCH_SIZE, resolution // PATCH_SIZE)

    def preprocess(self, frames: Frames) -> torch.Tensor:
        """Resize/normalize raw crops and move them to this backbone's device.

        Args:
            frames: uint8 RGB crops, shape (B, H, W, 3).

        Returns:
            float32 tensor, shape (B, 3, resolution, resolution), on `self.device`.
        """
        x = resize_and_normalize(frames, self.resolution)
        return torch.from_numpy(x).to(self.device)

    def forward(self, x: torch.Tensor) -> TokenOutput:
        """Run the DINOv3 forward pass and split its output into CLS/patch tokens.

        Args:
            x: preprocessed input, shape (B, 3, resolution, resolution).

        Returns:
            this batch's `TokenOutput`.
        """
        hidden_state = self.model(pixel_values=x).last_hidden_state
        cls, patches = split_tokens(hidden_state, self.num_register_tokens)
        return TokenOutput(cls=cls, patches=patches, grid_hw=self.grid_hw)

    def metadata(self) -> dict:
        """Backbone identity for provenance, written into `cuttle embed`'s config.yaml.

        Normalization constants are converted to plain lists (not tuples) since
        `yaml.safe_dump` -- used to write config.yaml -- can't represent a tuple.
        Keyed `backbone_hidden_size`, not `embed_dim` -- `write_embedder_output` already
        writes a top-level `embed_dim` for the *readout's* output dimensionality
        (`embedder.dim`), which differs from this backbone's own channel dimension for
        a readout like `gram` that projects down; reusing `embed_dim` here would
        silently overwrite that (this is exactly the bug a prior version of this
        function had, invisible only because `cls`/`meanpatch_*` happen to leave the
        channel dimension unchanged).
        """
        return {
            'hf_model_id': self.hf_model_id,
            'resolution': self.resolution,
            'backbone_hidden_size': self.embed_dim,
            'num_register_tokens': self.num_register_tokens,
            'normalization_mean': list(IMAGENET_MEAN),
            'normalization_std': list(IMAGENET_STD),
        }
