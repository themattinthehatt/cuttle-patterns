"""Tests for cuttle_patterns.embedders.spatial_weights."""

import pytest
import torch

from cuttle_patterns.embedders.spatial_weights import (
    patch_center_coords,
    patch_weights,
    radial_taper,
)


class TestPatchCenterCoords:
    """Test the function patch_center_coords."""

    def test_patch_center_coords_shape_and_row_major_order(self):
        # Act
        coords = patch_center_coords(grid_h=2, grid_w=3)

        # Assert
        assert coords.shape == (6, 2)
        assert coords.dtype == torch.float64
        # row-major: y increases every grid_w entries, x cycles within a row
        torch.testing.assert_close(coords[0, 0], coords[1, 0])
        torch.testing.assert_close(coords[0, 0], coords[2, 0])
        assert coords[3, 0] > coords[0, 0]

    def test_patch_center_coords_within_open_unit_interval(self):
        # Act
        coords = patch_center_coords(grid_h=4, grid_w=4)

        # Assert
        assert coords.min() > -1
        assert coords.max() < 1


class TestRadialTaper:
    """Test the function radial_taper."""

    def test_radial_taper_flat_at_center(self):
        # Arrange
        coords = torch.tensor([[0.0, 0.0]], dtype=torch.float64)

        # Act
        w = radial_taper(coords, r0=0.5)

        # Assert
        torch.testing.assert_close(w, torch.tensor([1.0], dtype=torch.float64))

    def test_radial_taper_zero_at_and_beyond_r_equals_one(self):
        # Arrange
        coords = torch.tensor([[1.0, 0.0], [0.0, 1.2]], dtype=torch.float64)

        # Act
        w = radial_taper(coords, r0=0.5)

        # Assert
        torch.testing.assert_close(w, torch.zeros(2, dtype=torch.float64))

    def test_radial_taper_matches_beast_pixel_grid_reference(self):
        # Arrange -- cross-check against the exact formula this replicates, evaluated
        # at pixel centers on a real pixel grid instead of patch centers
        beast_msps_vae_model = pytest.importorskip(
            'beast.models.msps_vae.msps_vae_model',
            reason='beast (msps-vae branch) not installed in this environment',
        )
        side, r0 = 33, 0.5
        # mean-1 normalized: build_raised_cosine_weight_map's own final step
        expected_mean1 = beast_msps_vae_model.build_raised_cosine_weight_map(side, r0)
        center = (side - 1) / 2
        ys, xs = torch.meshgrid(
            torch.arange(side, dtype=torch.float64), torch.arange(side, dtype=torch.float64),
            indexing='ij',
        )
        coords = torch.stack([(ys - center) / (side / 2), (xs - center) / (side / 2)], dim=-1)

        # Act
        actual = radial_taper(coords, r0=r0)

        # Assert -- radial_taper doesn't mean-1 normalize (see its docstring), so
        # mean-1 normalize it here before comparing to build_raised_cosine_weight_map
        torch.testing.assert_close(
            (actual / actual.mean()).float(), expected_mean1, atol=1e-5, rtol=1e-5,
        )


class TestPatchWeights:
    """Test the function patch_weights."""

    def test_patch_weights_uniform_sums_to_one_and_is_flat(self):
        # Act
        w = patch_weights(2, 2, kind='uniform')

        # Assert
        torch.testing.assert_close(w.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(w, torch.full_like(w, 0.25))

    def test_patch_weights_taper_sums_to_one_and_downweights_border(self):
        # Act
        w = patch_weights(4, 4, kind='taper')

        # Assert
        torch.testing.assert_close(w.sum(), torch.tensor(1.0, dtype=torch.float64))
        center_weight = w.reshape(4, 4)[1:3, 1:3].mean()
        border_weight = w.reshape(4, 4)[0, 0]
        assert center_weight > border_weight

    def test_patch_weights_unknown_kind_raises(self):
        # Act & Assert
        with pytest.raises(ValueError, match='unknown patch_weights kind'):
            patch_weights(2, 2, kind='not-a-real-kind')
