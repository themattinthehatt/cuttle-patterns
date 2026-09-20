"""Tests for cuttle_patterns.embedders.gram_math."""

import torch

from cuttle_patterns.embedders.gram_math import psd_sqrt, sym_to_vec, weighted_moments


class TestWeightedMoments:
    """Test the function weighted_moments."""

    def test_weighted_moments_matches_manual_computation(self):
        # Arrange -- one frame, 2 tokens, 2 channels, non-uniform weights
        X = torch.tensor([[[1.0, 2.0], [3.0, 0.0]]], dtype=torch.float64)
        w = torch.tensor([0.25, 0.75], dtype=torch.float64)

        # Act
        mu, G = weighted_moments(X, w)

        # Assert
        expected_mu = torch.tensor(
            [[0.25 * 1 + 0.75 * 3, 0.25 * 2 + 0.75 * 0]], dtype=torch.float64,
        )
        torch.testing.assert_close(mu, expected_mu)
        Xc = X - mu[:, None, :]
        expected_G = (
            w[0] * torch.outer(Xc[0, 0], Xc[0, 0]) + w[1] * torch.outer(Xc[0, 1], Xc[0, 1])
        ).unsqueeze(0)
        torch.testing.assert_close(G, expected_G)

    def test_weighted_moments_uniform_weights_matches_plain_covariance(self):
        # Arrange
        torch.manual_seed(0)
        X = torch.randn(3, 10, 4, dtype=torch.float64)
        w = torch.full((10,), 1.0 / 10, dtype=torch.float64)

        # Act
        mu, G = weighted_moments(X, w)

        # Assert
        expected_mu = X.mean(dim=1)
        torch.testing.assert_close(mu, expected_mu)
        Xc = X - expected_mu[:, None, :]
        expected_G = torch.einsum('bnc,bnd->bcd', Xc, Xc) / 10
        torch.testing.assert_close(G, expected_G)


class TestPsdSqrt:
    """Test the function psd_sqrt."""

    def test_psd_sqrt_reconstructs_original_matrix(self):
        # Arrange -- build a real PSD matrix as A @ A.T
        torch.manual_seed(1)
        A = torch.randn(2, 5, 5, dtype=torch.float64)
        G = A @ A.transpose(-1, -2)

        # Act
        S = psd_sqrt(G)

        # Assert
        torch.testing.assert_close(S @ S, G, atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(S, S.transpose(-1, -2))

    def test_psd_sqrt_zero_matrix_is_zero(self):
        # Arrange
        G = torch.zeros(3, 3, dtype=torch.float64)

        # Act
        S = psd_sqrt(G)

        # Assert
        torch.testing.assert_close(S, torch.zeros(3, 3, dtype=torch.float64))


class TestSymToVec:
    """Test the function sym_to_vec."""

    def test_sym_to_vec_shape(self):
        # Arrange
        S = torch.eye(4, dtype=torch.float64)

        # Act
        v = sym_to_vec(S)

        # Assert
        assert v.shape == (10,)

    def test_sym_to_vec_preserves_frobenius_norm(self):
        # Arrange -- a random symmetric matrix
        torch.manual_seed(2)
        A = torch.randn(6, 6, dtype=torch.float64)
        S = A + A.T

        # Act
        v = sym_to_vec(S)

        # Assert -- Euclidean norm of the vectorized form equals Frobenius norm of S
        torch.testing.assert_close(v.norm(), S.norm())

    def test_sym_to_vec_diagonal_unscaled_off_diagonal_scaled(self):
        # Arrange
        S = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float64)

        # Act
        v = sym_to_vec(S)

        # Assert -- upper triangle order: (0,0), (0,1), (1,1)
        torch.testing.assert_close(v, torch.tensor([1.0, 2.0 * 2**0.5, 3.0], dtype=torch.float64))
