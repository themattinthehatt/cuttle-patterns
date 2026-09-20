"""Tests for cuttle_patterns.embedders.readouts."""

import torch

from cuttle_patterns.embedders.base import TokenOutput
from cuttle_patterns.embedders.readouts import (
    READOUTS_BY_NAME,
    ClsReadout,
    MeanPatchTaperReadout,
    MeanPatchUniformReadout,
)
from cuttle_patterns.embedders.spatial_weights import patch_weights


class TestClsReadout:
    """Test the class ClsReadout."""

    def test_cls_readout_returns_cls_token_unchanged(self):
        # Arrange
        cls = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        patches = torch.zeros(2, 4, 3)
        tokens = TokenOutput(cls=cls, patches=patches, grid_hw=(2, 2))
        readout = ClsReadout(dim=3)

        # Act
        result = readout(tokens)

        # Assert
        torch.testing.assert_close(result, cls)
        assert readout.dim == 3

    def test_cls_readout_registered_under_cls(self):
        # Assert
        assert READOUTS_BY_NAME['cls'] is ClsReadout


class TestMeanPatchUniformReadout:
    """Test the class MeanPatchUniformReadout."""

    def test_mean_patch_uniform_readout_averages_over_patches(self):
        # Arrange
        cls = torch.zeros(2, 3)
        patches = torch.tensor([
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
            [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]],
        ])
        tokens = TokenOutput(cls=cls, patches=patches, grid_hw=(1, 2))
        readout = MeanPatchUniformReadout(dim=3)

        # Act
        result = readout(tokens)

        # Assert
        expected = torch.tensor([[2.0, 3.0, 4.0], [5.0, 5.0, 5.0]])
        torch.testing.assert_close(result, expected)
        assert readout.dim == 3

    def test_mean_patch_uniform_readout_registered_under_meanpatch_uniform(self):
        # Assert
        assert READOUTS_BY_NAME['meanpatch_uniform'] is MeanPatchUniformReadout


class TestMeanPatchTaperReadout:
    """Test the class MeanPatchTaperReadout."""

    def test_mean_patch_taper_readout_matches_manual_weighted_mean(self):
        # Arrange
        grid_hw = (2, 2)
        cls = torch.zeros(1, 3)
        patches = torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        tokens = TokenOutput(cls=cls, patches=patches, grid_hw=grid_hw)
        readout = MeanPatchTaperReadout(dim=3, r0=0.5)

        # Act
        result = readout(tokens)

        # Assert
        w = patch_weights(*grid_hw, kind='taper', r0=0.5).float()
        expected = torch.einsum('nc,n->c', patches[0], w).unsqueeze(0)
        torch.testing.assert_close(result, expected)
        assert readout.dim == 3

    def test_mean_patch_taper_readout_uniform_input_reduces_to_that_value(self):
        # Arrange -- taper weights sum to 1 regardless of shape, so a constant input
        # should pass through unchanged
        patches = torch.full((2, 9, 4), 3.5)
        tokens = TokenOutput(cls=torch.zeros(2, 4), patches=patches, grid_hw=(3, 3))
        readout = MeanPatchTaperReadout(dim=4)

        # Act
        result = readout(tokens)

        # Assert
        torch.testing.assert_close(result, torch.full((2, 4), 3.5))

    def test_mean_patch_taper_readout_caches_weights_per_grid_hw(self):
        # Arrange
        readout = MeanPatchTaperReadout(dim=4)
        patches = torch.zeros(1, 9, 4)
        tokens = TokenOutput(cls=torch.zeros(1, 4), patches=patches, grid_hw=(3, 3))

        # Act
        readout(tokens)
        readout(tokens)

        # Assert
        assert len(readout._weights_by_grid_hw) == 1

    def test_mean_patch_taper_readout_metadata_reports_r0(self):
        # Arrange
        readout = MeanPatchTaperReadout(dim=4, r0=0.7)

        # Act & Assert
        assert readout.metadata() == {'taper_r0': 0.7}

    def test_mean_patch_taper_readout_registered_under_meanpatch_taper(self):
        # Assert
        assert READOUTS_BY_NAME['meanpatch_taper'] is MeanPatchTaperReadout
