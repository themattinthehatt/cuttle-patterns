"""Tests for cuttle_patterns.embedders.dinov3."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from cuttle_patterns.embedders.dinov3 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    DINOv3Backbone,
    resize_and_normalize,
    split_tokens,
)


class _FakeDinoModel(nn.Module):
    """Stands in for a real DINOv3 ViT: same config/output shape, no real weights."""

    def __init__(self, num_register_tokens: int, hidden_size: int, grid_hw: tuple[int, int]):
        super().__init__()
        self.config = SimpleNamespace(
            num_register_tokens=num_register_tokens, hidden_size=hidden_size,
        )
        self._grid_hw = grid_hw

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        batch_size = pixel_values.shape[0]
        num_patches = self._grid_hw[0] * self._grid_hw[1]
        num_tokens = 1 + self.config.num_register_tokens + num_patches
        last_hidden_state = torch.arange(
            batch_size * num_tokens * self.config.hidden_size, dtype=torch.float32,
        ).reshape(batch_size, num_tokens, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=last_hidden_state)


class TestSplitTokens:
    """Test the function split_tokens."""

    def test_split_tokens_layout(self):
        # Arrange
        batch_size, num_register_tokens, num_patches, channels = 2, 4, 6, 3
        num_tokens = 1 + num_register_tokens + num_patches
        hidden_state = torch.arange(
            batch_size * num_tokens * channels, dtype=torch.float32,
        ).reshape(batch_size, num_tokens, channels)

        # Act
        cls, patches = split_tokens(hidden_state, num_register_tokens)

        # Assert
        assert cls.shape == (batch_size, channels)
        assert patches.shape == (batch_size, num_patches, channels)
        torch.testing.assert_close(cls, hidden_state[:, 0])
        torch.testing.assert_close(patches, hidden_state[:, 1 + num_register_tokens:])

    def test_split_tokens_zero_registers(self):
        # Arrange
        hidden_state = torch.zeros(1, 5, 2)

        # Act
        cls, patches = split_tokens(hidden_state, num_register_tokens=0)

        # Assert
        assert cls.shape == (1, 2)
        assert patches.shape == (1, 4, 2)


class TestResizeAndNormalize:
    """Test the function resize_and_normalize."""

    def test_resize_and_normalize_shape_and_dtype(self):
        # Arrange
        frames = np.zeros((2, 10, 20, 3), dtype=np.uint8)

        # Act
        result = resize_and_normalize(frames, resolution=32)

        # Assert
        assert result.shape == (2, 3, 32, 32)
        assert result.dtype == np.float32

    def test_resize_and_normalize_constant_frame_matches_expected_value(self):
        # Arrange
        frames = np.zeros((1, 16, 16, 3), dtype=np.uint8)

        # Act
        result = resize_and_normalize(frames, resolution=16)

        # Assert
        expected = -np.array(IMAGENET_MEAN) / np.array(IMAGENET_STD)
        for channel in range(3):
            np.testing.assert_allclose(result[0, channel], expected[channel], atol=1e-5)


class TestDINOv3Backbone:
    """Test the class DINOv3Backbone."""

    def test_dinov3_backbone_unknown_arch_raises_before_loading(self):
        # Act & Assert
        with pytest.raises(ValueError, match='unknown DINOv3 arch'):
            DINOv3Backbone('not-a-real-arch', resolution=224, device=torch.device('cpu'))

    def test_dinov3_backbone_bad_resolution_raises_before_loading(self):
        # Act & Assert
        with pytest.raises(ValueError, match='multiple of 16'):
            DINOv3Backbone('vitb16', resolution=225, device=torch.device('cpu'))

    def test_dinov3_backbone_forward_and_metadata(self, monkeypatch: pytest.MonkeyPatch):
        # Arrange
        fake_model = _FakeDinoModel(num_register_tokens=4, hidden_size=8, grid_hw=(2, 2))
        monkeypatch.setattr(
            'transformers.AutoModel.from_pretrained',
            lambda model_id: fake_model,
        )
        backbone = DINOv3Backbone('vitb16', resolution=32, device=torch.device('cpu'))

        # Act
        x = backbone.preprocess(np.zeros((2, 10, 20, 3), dtype=np.uint8))
        tokens = backbone.forward(x)
        metadata = backbone.metadata()

        # Assert
        assert backbone.key == 'dinov3_vitb16_32'
        assert tokens.cls.shape == (2, 8)
        assert tokens.patches.shape == (2, 4, 8)
        assert tokens.grid_hw == (2, 2)
        assert metadata == {
            'hf_model_id': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'resolution': 32,
            'embed_dim': 8,
            'num_register_tokens': 4,
            'normalization_mean': list(IMAGENET_MEAN),
            'normalization_std': list(IMAGENET_STD),
        }
