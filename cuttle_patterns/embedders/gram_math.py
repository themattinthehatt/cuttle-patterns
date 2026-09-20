"""Core math for the Gram (weighted, channel-projected covariance) readout.

Pure functions on tensors, matching the reference implementation in "Gram readout on
DINOv3 final-layer patch tokens" (`docs/implementation_notes/embedder.md` section 3)
verbatim, so each design caveat documented there (covariance not correlation, centered
with the mean kept separately, matrix square root normalization, sqrt(2) on
off-diagonals) traces back to one line here. Everything here runs in float64 -- callers
are responsible for casting token tensors up from the backbone's native dtype.
"""

import math

import torch


def weighted_moments(X: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame weighted mean and weighted, centered covariance across tokens.

    Args:
        X: float64 tensor, shape (B, N, C) -- e.g. a batch's (projected) patch tokens.
        w: float64 non-negative weights summing to 1, shape (N,).

    Returns:
        `(mu, G)`: `mu` is the weighted mean, shape (B, C); `G` is the weighted,
        mean-centered covariance across the C channels, shape (B, C, C).
    """
    mu = torch.einsum('bnc,n->bc', X, w)
    Xc = X - mu[:, None, :]
    G = torch.einsum('bnc,bnd,n->bcd', Xc, Xc, w)
    return mu, G


def psd_sqrt(G: torch.Tensor) -> torch.Tensor:
    """Matrix square root of a batch of symmetric positive-semidefinite matrices.

    Eigenvalues are clamped to be non-negative before the square root, to absorb
    floating-point noise on a matrix that's PSD by construction (a weighted covariance).

    Args:
        G: float64 tensor, shape (..., k, k), symmetric.

    Returns:
        `S` such that `S @ S` reconstructs `G` (up to the eigenvalue clamp above), same
        shape as `G`.
    """
    evals, evecs = torch.linalg.eigh(G)
    s = evals.clamp_min(0).sqrt()
    return (evecs * s[..., None, :]) @ evecs.transpose(-1, -2)


def sym_to_vec(S: torch.Tensor) -> torch.Tensor:
    """Vectorize a batch of symmetric matrices' upper triangle.

    Off-diagonal entries are scaled by `sqrt(2)` so that Euclidean distance between two
    vectorized outputs equals Frobenius distance between the original matrices --
    vectorizing only the upper triangle otherwise counts each off-diagonal entry once
    while the Frobenius norm counts it twice.

    Args:
        S: float64 tensor, shape (..., k, k), symmetric.

    Returns:
        vectorized upper triangle including the diagonal, shape (..., k * (k + 1) // 2).
    """
    k = S.shape[-1]
    iu = torch.triu_indices(k, k, device=S.device)
    v = S[..., iu[0], iu[1]].clone()
    v[..., iu[0] != iu[1]] *= math.sqrt(2.0)
    return v
