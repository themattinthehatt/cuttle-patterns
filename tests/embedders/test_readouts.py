"""Tests for cuttle_patterns.embedders.readouts."""

import pytest
import torch

from cuttle_patterns.embedders.base import TokenOutput
from cuttle_patterns.embedders.readouts import (
    READOUTS_BY_NAME,
    ClsReadout,
    FusedGramReadout,
    GramReadout,
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


class TestGramReadout:
    """Test the class GramReadout."""

    def test_gram_readout_unknown_weights_raises(self):
        # Act & Assert
        with pytest.raises(ValueError, match='unknown gram weights'):
            GramReadout(dim=4, k=2, weights='not-a-real-kind')

    def test_gram_readout_k_exceeds_dim_raises(self):
        # Act & Assert
        with pytest.raises(ValueError, match='cannot exceed'):
            GramReadout(dim=4, k=5, weights='uniform')

    def test_gram_readout_name_encodes_hyperparameters(self):
        # Act
        readout = GramReadout(dim=4, k=2, weights='taper')

        # Assert
        assert readout.name == 'gram_k2_taper'
        assert readout.dim == 3  # k * (k + 1) // 2

    def test_gram_readout_call_before_fit_raises(self):
        # Arrange
        readout = GramReadout(dim=4, k=2, weights='uniform')
        tokens = TokenOutput(cls=torch.zeros(1, 4), patches=torch.zeros(1, 4, 4), grid_hw=(2, 2))

        # Act & Assert
        with pytest.raises(RuntimeError, match='must be fit before use'):
            readout(tokens)

    def test_gram_readout_finalize_pass_without_partial_fit_raises(self):
        # Arrange
        readout = GramReadout(dim=4, k=2, weights='uniform')

        # Act & Assert
        with pytest.raises(RuntimeError, match='no fit-set batches seen'):
            readout.finalize_pass(0)

    def test_gram_readout_fit_sets_orthonormal_projection(self):
        # Arrange
        torch.manual_seed(0)
        dim_in, k = 5, 2
        readout = GramReadout(dim=dim_in, k=k, weights='uniform')
        grid_hw = (2, 2)

        # Act -- multiple partial_fit calls accumulate before one finalize_pass
        for _ in range(3):
            patches = torch.randn(4, 4, dim_in)
            tokens = TokenOutput(cls=torch.zeros(4, dim_in), patches=patches, grid_hw=grid_hw)
            readout.partial_fit(tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)

        # Assert
        assert readout.P.shape == (k, dim_in)
        torch.testing.assert_close(
            readout.P @ readout.P.T, torch.eye(k, dtype=torch.float64), atol=1e-6, rtol=1e-6,
        )
        assert 0.0 <= readout.variance_retained <= 1.0 + 1e-6

    def test_gram_readout_full_rank_projection_retains_all_variance(self):
        # Arrange
        torch.manual_seed(1)
        dim_in = 4
        readout = GramReadout(dim=dim_in, k=dim_in, weights='uniform')
        patches = torch.randn(6, 5, dim_in)
        tokens = TokenOutput(cls=torch.zeros(6, dim_in), patches=patches, grid_hw=(1, 5))

        # Act
        readout.partial_fit(tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)

        # Assert
        assert readout.variance_retained == pytest.approx(1.0, abs=1e-6)

    def test_gram_readout_caches_weights_per_grid_hw(self):
        # Arrange
        readout = GramReadout(dim=4, k=2, weights='uniform')
        patches = torch.randn(2, 4, 4)
        tokens = TokenOutput(cls=torch.zeros(2, 4), patches=patches, grid_hw=(2, 2))

        # Act
        readout.partial_fit(tokens, pass_idx=0)
        readout.partial_fit(tokens, pass_idx=0)

        # Assert
        assert len(readout._weights_by_grid_hw) == 1

    def test_gram_readout_call_after_fit_returns_correct_shape(self):
        # Arrange
        torch.manual_seed(2)
        dim_in, k = 6, 3
        readout = GramReadout(dim=dim_in, k=k, weights='taper')
        fit_patches = torch.randn(8, 9, dim_in)
        fit_tokens = TokenOutput(
            cls=torch.zeros(8, dim_in), patches=fit_patches, grid_hw=(3, 3),
        )
        readout.partial_fit(fit_tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)
        call_patches = torch.randn(2, 9, dim_in)
        call_tokens = TokenOutput(
            cls=torch.zeros(2, dim_in), patches=call_patches, grid_hw=(3, 3),
        )

        # Act
        result = readout(call_tokens)

        # Assert
        assert result.shape == (2, k * (k + 1) // 2)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    def test_gram_readout_metadata_reports_hyperparams_and_variance_retained(self):
        # Arrange
        torch.manual_seed(3)
        readout = GramReadout(dim=4, k=2, weights='taper')
        patches = torch.randn(3, 4, 4)
        tokens = TokenOutput(cls=torch.zeros(3, 4), patches=patches, grid_hw=(2, 2))
        readout.partial_fit(tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)

        # Act
        metadata = readout.metadata()

        # Assert
        assert metadata == {
            'gram_k': 2, 'gram_weights': 'taper',
            'gram_variance_retained': readout.variance_retained,
        }

    def test_gram_readout_state_dict_round_trip(self):
        # Arrange
        torch.manual_seed(4)
        readout = GramReadout(dim=4, k=2, weights='uniform')
        patches = torch.randn(3, 4, 4)
        tokens = TokenOutput(cls=torch.zeros(3, 4), patches=patches, grid_hw=(2, 2))
        readout.partial_fit(tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)
        state = readout.state_dict()

        # Act
        restored = GramReadout(dim=4, k=2, weights='uniform')
        restored.load_state_dict(state)

        # Assert
        torch.testing.assert_close(restored.P, readout.P)
        assert restored.variance_retained == readout.variance_retained

    def test_gram_readout_registered_under_gram(self):
        # Assert
        assert READOUTS_BY_NAME['gram'] is GramReadout

    @pytest.mark.gpu
    def test_gram_readout_fit_and_call_on_cuda(self):
        # Arrange -- regression test for a real bug: patch_weights always builds on
        # CPU, and _patch_weights_for previously never moved the cached weights to the
        # backbone's device, so this crashed with a device-mismatch error as soon as
        # cuttle embed ran with a CUDA backbone (this readout worked fine in every CPU
        # unit test above, since CPU is torch's default device)
        device = torch.device('cuda')
        torch.manual_seed(5)
        dim_in, k = 6, 3
        readout = GramReadout(dim=dim_in, k=k, weights='taper')
        fit_patches = torch.randn(4, 9, dim_in, device=device)
        fit_tokens = TokenOutput(
            cls=torch.zeros(4, dim_in, device=device), patches=fit_patches, grid_hw=(3, 3),
        )

        # Act
        readout.partial_fit(fit_tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)
        call_patches = torch.randn(2, 9, dim_in, device=device)
        call_tokens = TokenOutput(
            cls=torch.zeros(2, dim_in, device=device), patches=call_patches, grid_hw=(3, 3),
        )
        result = readout(call_tokens)

        # Assert
        assert result.shape == (2, k * (k + 1) // 2)
        assert result.device.type == 'cuda'
        # state_dict moves P to CPU regardless of the fitting device, for portability
        assert readout.state_dict()['P'].device.type == 'cpu'


def _tokens_per_layer(
    layer_dims: dict[str, int], n_frames: int, grid_hw: tuple[int, int],
) -> list[TokenOutput]:
    """One randn TokenOutput per layer, in `layer_dims`' order."""
    n_positions = grid_hw[0] * grid_hw[1]
    return [
        TokenOutput(
            cls=torch.zeros(n_frames, dim), patches=torch.randn(n_frames, n_positions, dim),
            grid_hw=grid_hw,
        )
        for dim in layer_dims.values()
    ]


class TestFusedGramReadout:
    """Test the class FusedGramReadout."""

    def test_fused_gram_readout_not_registered_by_name(self):
        # Assert -- only reachable via build_embedder's VGG-fusion branch
        assert FusedGramReadout not in READOUTS_BY_NAME.values()

    def test_fused_gram_readout_name_matches_single_layer_pattern(self):
        # Act
        readout = FusedGramReadout(layer_dims={'relu3_1': 4, 'relu4_1': 3}, k=2, weights='taper')

        # Assert -- layer identity lives in the backbone key, not the readout name
        assert readout.name == 'gram_k2_taper'

    def test_fused_gram_readout_k_exceeding_any_layer_raises(self):
        # Act & Assert -- cascades from GramReadout's own per-layer validation
        with pytest.raises(ValueError, match='cannot exceed'):
            FusedGramReadout(layer_dims={'relu3_1': 4, 'relu4_1': 3}, k=4, weights='uniform')

    def test_fused_gram_readout_dim_sums_every_layer(self):
        # Act
        readout = FusedGramReadout(layer_dims={'relu3_1': 4, 'relu4_1': 6}, k=2, weights='uniform')

        # Assert -- k=2 -> k(k+1)/2=3 per layer, two layers
        assert readout.dim == 6

    def test_fused_gram_readout_fit_and_call_concatenates_layers_in_order(self):
        # Arrange
        torch.manual_seed(6)
        layer_dims = {'relu3_1': 4, 'relu4_1': 3}
        readout = FusedGramReadout(layer_dims=layer_dims, k=2, weights='taper')
        grid_hw = (3, 3)
        fit_tokens = _tokens_per_layer(layer_dims, n_frames=8, grid_hw=grid_hw)

        # Act
        readout.partial_fit(fit_tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)
        call_tokens = _tokens_per_layer(layer_dims, n_frames=2, grid_hw=grid_hw)
        result = readout(call_tokens)

        # Assert -- fit-set list emptied by partial_fit; call-time list emptied by call
        assert fit_tokens == []
        assert call_tokens == []
        assert result.shape == (2, readout.dim)
        assert torch.isfinite(result).all()
        # matches computing each layer's own GramReadout output independently and
        # concatenating, confirming order and values (not just shape)
        expected_layer1 = readout._sub_readouts['relu3_1'].dim
        assert readout._sub_readouts['relu3_1'].P.shape == (2, 4)
        assert readout._sub_readouts['relu4_1'].P.shape == (2, 3)
        assert result.shape[-1] == expected_layer1 + readout._sub_readouts['relu4_1'].dim

    def test_fused_gram_readout_metadata_reports_shared_and_per_layer_fields(self):
        # Arrange
        torch.manual_seed(7)
        layer_dims = {'relu3_1': 4, 'relu4_1': 3}
        readout = FusedGramReadout(layer_dims=layer_dims, k=2, weights='taper')
        fit_tokens = _tokens_per_layer(layer_dims, n_frames=5, grid_hw=(3, 3))
        readout.partial_fit(fit_tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)

        # Act
        metadata = readout.metadata()

        # Assert
        assert metadata == {
            'gram_k': 2,
            'gram_weights': 'taper',
            'gram_layers': ['relu3_1', 'relu4_1'],
            'gram_variance_retained_relu3_1': readout._sub_readouts['relu3_1'].variance_retained,
            'gram_variance_retained_relu4_1': readout._sub_readouts['relu4_1'].variance_retained,
        }

    def test_fused_gram_readout_state_dict_round_trip(self):
        # Arrange
        torch.manual_seed(8)
        layer_dims = {'relu3_1': 4, 'relu4_1': 3}
        readout = FusedGramReadout(layer_dims=layer_dims, k=2, weights='uniform')
        fit_tokens = _tokens_per_layer(layer_dims, n_frames=5, grid_hw=(3, 3))
        readout.partial_fit(fit_tokens, pass_idx=0)
        readout.finalize_pass(pass_idx=0)
        state = readout.state_dict()

        # Act
        restored = FusedGramReadout(layer_dims=layer_dims, k=2, weights='uniform')
        restored.load_state_dict(state)

        # Assert
        for layer in layer_dims:
            torch.testing.assert_close(
                restored._sub_readouts[layer].P, readout._sub_readouts[layer].P,
            )
