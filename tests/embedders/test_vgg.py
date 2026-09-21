"""Tests for cuttle_patterns.embedders.vgg."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cuttle_patterns.embedders.vgg import (
    VGG_LAYERS,
    MultiLayerVGGBackbone,
    VGGBackbone,
)


def _build_fake_vgg19_features() -> nn.Sequential:
    """Tiny stand-in for `vgg19().features`: same conv/relu/maxpool layout, few channels.

    Matches real VGG-19's per-block conv count (2, 2, 4, 4, 4 -- confirmed by listing
    `torchvision.models.vgg19().features` directly), so `VGG_LAYERS`' indices slice this
    fake at the same positions they slice the real model.
    """
    block_conv_counts = [2, 2, 4, 4, 4]
    channel_counts = [3, 2, 2, 2, 2, 2]
    layers = []
    in_channels = channel_counts[0]
    for block_idx, n_convs in enumerate(block_conv_counts):
        out_channels = channel_counts[block_idx + 1]
        for _ in range(n_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


class TestVGGBackbone:
    """Test the class VGGBackbone."""

    def test_vgg_backbone_unknown_layer_raises_before_loading(self):
        # Act & Assert
        with pytest.raises(ValueError, match='unknown VGG layer'):
            VGGBackbone('not-a-real-layer', resolution=224, device=torch.device('cpu'))

    def test_vgg_backbone_bad_resolution_raises_before_loading(self):
        # Act & Assert -- relu5_1's stride is 16, so 30 does not divide evenly
        with pytest.raises(ValueError, match='downsampling stride'):
            VGGBackbone('relu5_1', resolution=30, device=torch.device('cpu'))

    def test_vgg_backbone_forward_and_metadata(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange
        fake_features = _build_fake_vgg19_features()
        monkeypatch.setattr(
            'cuttle_patterns.embedders.vgg.vgg19',
            lambda weights: SimpleNamespace(features=fake_features),
        )
        backbone = VGGBackbone('relu3_1', resolution=32, device=torch.device('cpu'))

        # Act
        x = backbone.preprocess(np.zeros((2, 10, 20, 3), dtype=np.uint8))
        tokens = backbone.forward(x)
        metadata = backbone.metadata()

        # Assert -- relu3_1's stride is 4, so a 32px input yields an 8x8 grid; channel
        # count here is the fake model's real 2, not VGG_LAYERS['relu3_1'].channels'
        # 256 (that describes the real vgg19, which this fake does not reproduce
        # numerically)
        assert backbone.key == 'vgg19_3_32'
        assert x.shape == (2, 3, 32, 32)
        assert tokens.cls is None
        assert tokens.patches.shape == (2, 64, 2)
        assert tokens.grid_hw == (8, 8)
        assert metadata == {
            'vgg_layer': 'relu3_1',
            'resolution': 32,
            'backbone_hidden_size': 256,
        }

    def test_vgg_backbone_truncates_features_at_layer_index(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        fake_features = _build_fake_vgg19_features()
        monkeypatch.setattr(
            'cuttle_patterns.embedders.vgg.vgg19',
            lambda weights: SimpleNamespace(features=fake_features),
        )

        # Act
        backbone = VGGBackbone('relu1_1', resolution=16, device=torch.device('cpu'))

        # Assert -- relu1_1 is features[1]: exactly Conv2d then ReLU, nothing more
        assert len(backbone.model) == VGG_LAYERS['relu1_1'].index + 1


class TestMultiLayerVGGBackbone:
    """Test the class MultiLayerVGGBackbone."""

    def test_multi_layer_vgg_backbone_empty_layers_raises(self):
        # Act & Assert
        with pytest.raises(ValueError, match='non-empty'):
            MultiLayerVGGBackbone([], resolution=224, device=torch.device('cpu'))

    def test_multi_layer_vgg_backbone_unknown_layer_raises_before_loading(self):
        # Act & Assert
        with pytest.raises(ValueError, match='unknown VGG layer'):
            MultiLayerVGGBackbone(
                ['relu3_1', 'not-a-real-layer'], resolution=224, device=torch.device('cpu'),
            )

    def test_multi_layer_vgg_backbone_bad_resolution_checks_deepest_layer(self):
        # Act & Assert -- relu5_1's stride (16) governs, even though relu3_1 comes first
        with pytest.raises(ValueError, match='downsampling stride'):
            MultiLayerVGGBackbone(
                ['relu3_1', 'relu5_1'], resolution=30, device=torch.device('cpu'),
            )

    def test_multi_layer_vgg_backbone_canonicalizes_order_and_dedupes(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        fake_features = _build_fake_vgg19_features()
        monkeypatch.setattr(
            'cuttle_patterns.embedders.vgg.vgg19',
            lambda weights: SimpleNamespace(features=fake_features),
        )

        # Act -- deliberately out of order, with a duplicate
        backbone = MultiLayerVGGBackbone(
            ['relu5_1', 'relu3_1', 'relu3_1'], resolution=32, device=torch.device('cpu'),
        )

        # Assert
        assert backbone.layers == ['relu3_1', 'relu5_1']
        assert backbone.key == 'vgg19_35_32'

    def test_multi_layer_vgg_backbone_forward_and_metadata(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        fake_features = _build_fake_vgg19_features()
        monkeypatch.setattr(
            'cuttle_patterns.embedders.vgg.vgg19',
            lambda weights: SimpleNamespace(features=fake_features),
        )
        backbone = MultiLayerVGGBackbone(
            ['relu2_1', 'relu4_1'], resolution=32, device=torch.device('cpu'),
        )

        # Act
        x = backbone.preprocess(np.zeros((2, 10, 20, 3), dtype=np.uint8))
        tokens = backbone.forward(x)
        metadata = backbone.metadata()

        # Assert -- relu2_1's stride is 2 (16x16 grid at 32px), relu4_1's is 8 (4x4);
        # channel counts are the fake model's real 2, not VGG_LAYERS[...].channels'
        assert len(tokens) == 2
        assert tokens[0].cls is None and tokens[1].cls is None
        assert tokens[0].patches.shape == (2, 256, 2)
        assert tokens[0].grid_hw == (16, 16)
        assert tokens[1].patches.shape == (2, 16, 2)
        assert tokens[1].grid_hw == (4, 4)
        assert metadata == {
            'vgg_layers': ['relu2_1', 'relu4_1'],
            'resolution': 32,
            'backbone_hidden_size_by_layer': {'relu2_1': 128, 'relu4_1': 512},
        }

    def test_multi_layer_vgg_backbone_segments_partition_trunk_without_overlap(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        fake_features = _build_fake_vgg19_features()
        monkeypatch.setattr(
            'cuttle_patterns.embedders.vgg.vgg19',
            lambda weights: SimpleNamespace(features=fake_features),
        )

        # Act
        backbone = MultiLayerVGGBackbone(
            ['relu1_1', 'relu3_1', 'relu5_1'], resolution=32, device=torch.device('cpu'),
        )

        # Assert -- segment lengths sum to exactly the deepest layer's index + 1 (every
        # trunk position covered by exactly one segment), confirming the shared trunk is
        # sliced into contiguous, non-overlapping pieces -- not each layer independently
        # re-running from position 0, which would instead sum to a much larger total
        total_ops = sum(len(segment) for segment in backbone.segments)
        assert total_ops == VGG_LAYERS['relu5_1'].index + 1


class TestLayerTables:
    """Test the module-level VGG_LAYERS lookup table."""

    def test_layer_tables_has_the_five_canonical_keys(self):
        # Assert
        assert set(VGG_LAYERS) == {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'}

    def test_layer_tables_code_is_the_layer_s_block_number(self):
        # Assert -- single digits, so MultiLayerVGGBackbone can concatenate them (e.g.
        # '345' for relu3_1+relu4_1+relu5_1) into one readable model_name segment
        assert {layer: info.code for layer, info in VGG_LAYERS.items()} == {
            'relu1_1': '1',
            'relu2_1': '2',
            'relu3_1': '3',
            'relu4_1': '4',
            'relu5_1': '5',
        }

    def test_layer_tables_match_gatys_et_al_channel_counts(self):
        # Assert -- confirmed against a live torchvision.models.vgg19().features listing
        assert {layer: info.channels for layer, info in VGG_LAYERS.items()} == {
            'relu1_1': 64,
            'relu2_1': 128,
            'relu3_1': 256,
            'relu4_1': 512,
            'relu5_1': 512,
        }

    def test_layer_tables_stride_doubles_after_each_block(self):
        # Assert
        assert {layer: info.stride for layer, info in VGG_LAYERS.items()} == {
            'relu1_1': 1,
            'relu2_1': 2,
            'relu3_1': 4,
            'relu4_1': 8,
            'relu5_1': 16,
        }
