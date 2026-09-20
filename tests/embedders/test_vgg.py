"""Tests for cuttle_patterns.embedders.vgg."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cuttle_patterns.embedders.vgg import (
    LAYER_TO_CHANNELS,
    LAYER_TO_CODE,
    LAYER_TO_INDEX,
    LAYER_TO_STRIDE,
    VGGBackbone,
)


def _build_fake_vgg19_features() -> nn.Sequential:
    """Tiny stand-in for `vgg19().features`: same conv/relu/maxpool layout, few channels.

    Matches real VGG-19's per-block conv count (2, 2, 4, 4, 4 -- confirmed by listing
    `torchvision.models.vgg19().features` directly), so `LAYER_TO_INDEX` slices this
    fake at the same positions it slices the real model.
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
        # count here is the fake model's real 2, not LAYER_TO_CHANNELS' 256 (that dict
        # describes the real vgg19, which this fake does not reproduce numerically)
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
        assert len(backbone.model) == LAYER_TO_INDEX['relu1_1'] + 1


class TestLayerTables:
    """Test the module-level layer lookup tables."""

    def test_layer_tables_share_the_same_five_canonical_keys(self):
        # Assert
        assert (
            set(LAYER_TO_INDEX) == set(LAYER_TO_CHANNELS) == set(LAYER_TO_STRIDE)
            == set(LAYER_TO_CODE)
        )
        assert set(LAYER_TO_INDEX) == {'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'}

    def test_layer_tables_code_is_the_layer_s_block_number(self):
        # Assert -- single digits now, so future multi-layer fusion can concatenate them
        # (e.g. '345' for relu3_1+relu4_1+relu5_1) into one readable model_name segment
        assert LAYER_TO_CODE == {
            'relu1_1': '1',
            'relu2_1': '2',
            'relu3_1': '3',
            'relu4_1': '4',
            'relu5_1': '5',
        }

    def test_layer_tables_match_gatys_et_al_channel_counts(self):
        # Assert -- confirmed against a live torchvision.models.vgg19().features listing
        assert LAYER_TO_CHANNELS == {
            'relu1_1': 64,
            'relu2_1': 128,
            'relu3_1': 256,
            'relu4_1': 512,
            'relu5_1': 512,
        }

    def test_layer_tables_stride_doubles_after_each_block(self):
        # Assert
        assert LAYER_TO_STRIDE == {
            'relu1_1': 1,
            'relu2_1': 2,
            'relu3_1': 4,
            'relu4_1': 8,
            'relu5_1': 16,
        }
