"""VGG-19 backbones: torchvision's pretrained conv net, at one or several named layers.

Unlike the DINOv3 ViT, VGG-19 has no CLS token -- it returns one conv feature map,
which `VGGBackbone` flattens into a `TokenOutput` with `cls=None` (see
`cuttle_patterns.embedders.base.TokenOutput`); `MultiLayerVGGBackbone` returns several,
one per requested layer, for Gram fusion. Only the canonical 5 Gatys et al.
texture/style layers are supported -- see "VGG-19 Gram" and "VGG-19 Gram fusion" in
`docs/implementation_notes/embedder.md`.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19

from cuttle_patterns.embedders.base import Backbone, Frames, TokenOutput
from cuttle_patterns.embedders.dinov3 import resize_and_normalize

BACKBONE_NAME = 'vgg19'


@dataclass(frozen=True)
class VGGLayerInfo:
    """Static facts about one of the 5 canonical Gatys et al. VGG-19 layers.

    One dict keyed by layer name, rather than four parallel ones, so a layer's facts
    can't drift out of sync with each other -- every lookup goes through the same key.
    """

    index: int
    """`vgg19().features` index of this layer's block-first ReLU -- confirmed by
    listing `vgg19().features` directly, not assumed."""

    channels: int
    """This layer's output channel count."""

    stride: int
    """Downsampling factor from input resolution to this layer's spatial grid: one
    maxpool per block boundary crossed before it (relu1_1 crosses none, relu5_1 crosses
    four)."""

    code: str
    """Short digit alias for the backbone key/model name -- keeps model_name readable
    (e.g. vgg19_3_448_gram_k64_taper, not vgg19_relu3_1_448_gram_k64_taper). Multi-layer
    fusion concatenates these codes in block order (e.g. '345' for
    relu3_1+relu4_1+relu5_1). `metadata()` still reports the full layer name for
    provenance -- only the directory name is shortened."""


# Gatys et al.'s canonical 5 texture/style layers (A Neural Algorithm of Artistic
# Style, CVPR 2016), one ReLU per block
VGG_LAYERS: dict[str, VGGLayerInfo] = {
    'relu1_1': VGGLayerInfo(index=1, channels=64, stride=1, code='1'),
    'relu2_1': VGGLayerInfo(index=6, channels=128, stride=2, code='2'),
    'relu3_1': VGGLayerInfo(index=11, channels=256, stride=4, code='3'),
    'relu4_1': VGGLayerInfo(index=20, channels=512, stride=8, code='4'),
    'relu5_1': VGGLayerInfo(index=29, channels=512, stride=16, code='5'),
}


class VGGBackbone(Backbone):
    """Wraps a pretrained torchvision VGG-19, truncated at one named conv block."""

    def __init__(self, layer: str, resolution: int, device: torch.device):
        """Load VGG-19 and truncate it at `layer`.

        Args:
            layer: one of `VGG_LAYERS`' keys.
            resolution: square input side length, in pixels; must be a multiple of
                `layer`'s downsampling stride, so its output grid divides evenly.
            device: device to load the model onto.

        Raises:
            ValueError: if `layer` is unknown, or `resolution` isn't a multiple of
                `layer`'s stride.
        """
        if layer not in VGG_LAYERS:
            raise ValueError(f'unknown VGG layer: {layer!r}; choices: {list(VGG_LAYERS)}')
        info = VGG_LAYERS[layer]
        if resolution % info.stride != 0:
            raise ValueError(
                f'resolution must be a multiple of {layer}\'s downsampling stride '
                f'({info.stride}), got {resolution}'
            )

        self.layer = layer
        self.resolution = resolution
        self.device = device
        self.key = f'{BACKBONE_NAME}_{info.code}_{resolution}'
        self.embed_dim = info.channels
        self.grid_hw = (resolution // info.stride, resolution // info.stride)

        full_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.model = full_model.features[:info.index + 1]
        self.model.eval()
        self.model.to(device)

    def preprocess(self, frames: Frames) -> torch.Tensor:
        """Resize/normalize raw crops and move them to this backbone's device.

        Reuses DINOv3's resize/ImageNet-normalization pipeline unchanged -- torchvision's
        pretrained VGG-19 weights expect the same ImageNet statistics.

        Args:
            frames: uint8 RGB crops, shape (B, H, W, 3).

        Returns:
            float32 tensor, shape (B, 3, resolution, resolution), on `self.device`.
        """
        x = resize_and_normalize(frames, self.resolution)
        return torch.from_numpy(x).to(self.device)

    def forward(self, x: torch.Tensor) -> TokenOutput:
        """Run the truncated VGG-19 forward pass and flatten its feature map.

        Args:
            x: preprocessed input, shape (B, 3, resolution, resolution).

        Returns:
            this batch's `TokenOutput`; `cls` is `None` -- VGG has no CLS-token
            equivalent.
        """
        feature_map = self.model(x)  # (B, C, H, W)
        patches = feature_map.flatten(2).transpose(1, 2)  # (B, H * W, C), row-major
        return TokenOutput(patches=patches, grid_hw=self.grid_hw)

    def metadata(self) -> dict:
        """Backbone identity for provenance, written into `cuttle embed`'s config.yaml.

        Keyed `backbone_hidden_size`, not `embed_dim` -- see
        `cuttle_patterns.embedders.dinov3.DINOv3Backbone.metadata`'s docstring for why.
        """
        return {
            'vgg_layer': self.layer,
            'resolution': self.resolution,
            'backbone_hidden_size': self.embed_dim,
        }


class MultiLayerVGGBackbone(Backbone):
    """Wraps a pretrained torchvision VGG-19, extracting several named layers at once.

    Pairs with `cuttle_patterns.embedders.readouts.FusedGramReadout` (Gram fusion) --
    see "VGG-19 Gram fusion" in `docs/implementation_notes/embedder.md`. Unlike
    `VGGBackbone`, `forward` returns a `list[TokenOutput]`, one per requested layer in
    canonical block order, not a single `TokenOutput` -- only a readout built to expect
    that (`FusedGramReadout`) can consume it.

    Runs VGG-19's shared trunk exactly once per batch, not once per layer, by slicing
    `features` into contiguous segments between consecutive requested layers and
    chaining them: each layer's raw activation is the input to the next segment, so it's
    never held in a separate list alongside the others -- only the *sub-readout's* own
    per-layer float64 Gram computation (the expensive part) needs to avoid that, which is
    `FusedGramReadout`'s responsibility, not this backbone's.
    """

    def __init__(self, layers: list[str], resolution: int, device: torch.device):
        """Load VGG-19 and slice it at each of `layers`.

        Args:
            layers: one or more of `VGG_LAYERS`' keys, in any order (canonicalized to
                block order and deduplicated below).
            resolution: square input side length, in pixels; must be a multiple of the
                *deepest* requested layer's downsampling stride -- automatically a
                multiple of every shallower requested layer's stride too, since each is
                a power of 2 dividing the next.
            device: device to load the model onto.

        Raises:
            ValueError: if `layers` is empty, contains an unknown layer name, or
                `resolution` isn't a multiple of the deepest layer's stride.
        """
        if not layers:
            raise ValueError('layers must be non-empty')
        unknown = set(layers) - set(VGG_LAYERS)
        if unknown:
            raise ValueError(
                f'unknown VGG layer(s): {sorted(unknown)}; choices: {list(VGG_LAYERS)}'
            )

        self.layers = sorted(set(layers), key=lambda layer: VGG_LAYERS[layer].index)
        deepest = VGG_LAYERS[self.layers[-1]]
        if resolution % deepest.stride != 0:
            raise ValueError(
                f'resolution must be a multiple of {self.layers[-1]}\'s downsampling '
                f'stride ({deepest.stride}), got {resolution}'
            )

        self.resolution = resolution
        self.device = device
        codes = ''.join(VGG_LAYERS[layer].code for layer in self.layers)
        self.key = f'{BACKBONE_NAME}_{codes}_{resolution}'
        self.channels_by_layer = {layer: VGG_LAYERS[layer].channels for layer in self.layers}
        self.grid_hw_by_layer = {
            layer: (resolution // VGG_LAYERS[layer].stride, resolution // VGG_LAYERS[layer].stride)
            for layer in self.layers
        }

        full_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        segments = []
        start = 0
        for layer in self.layers:
            layer_index = VGG_LAYERS[layer].index
            segment = full_model.features[start:layer_index + 1]
            segment.eval()
            segment.to(device)
            segments.append(segment)
            start = layer_index + 1
        self.segments = nn.ModuleList(segments)

    def preprocess(self, frames: Frames) -> torch.Tensor:
        """Resize/normalize raw crops and move them to this backbone's device.

        Args:
            frames: uint8 RGB crops, shape (B, H, W, 3).

        Returns:
            float32 tensor, shape (B, 3, resolution, resolution), on `self.device`.
        """
        x = resize_and_normalize(frames, self.resolution)
        return torch.from_numpy(x).to(self.device)

    def forward(self, x: torch.Tensor) -> list[TokenOutput]:
        """Run each layer's segment in sequence, chaining the running activation.

        Args:
            x: preprocessed input, shape (B, 3, resolution, resolution).

        Returns:
            one `TokenOutput` per requested layer, in `self.layers`' canonical order;
            `cls` is `None` on each, same as `VGGBackbone`.
        """
        activation = x
        outputs = []
        for layer, segment in zip(self.layers, self.segments, strict=True):
            activation = segment(activation)
            patches = activation.flatten(2).transpose(1, 2)  # (B, H * W, C), row-major
            outputs.append(TokenOutput(patches=patches, grid_hw=self.grid_hw_by_layer[layer]))
        return outputs

    def metadata(self) -> dict:
        """Backbone identity for provenance, written into `cuttle embed`'s config.yaml."""
        return {
            'vgg_layers': self.layers,
            'resolution': self.resolution,
            'backbone_hidden_size_by_layer': self.channels_by_layer,
        }
