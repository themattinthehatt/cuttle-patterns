"""Patch-grid spatial weights shared by the `meanpatch_taper` and (planned) `gram_*`
readouts -- see "Spatial weights" in `docs/implementation_notes/embedder.md` section 3.
"""

import torch

# matches configs/beast_msps_vae.yaml's spatial_loss_weight_r0 and
# beast/models/msps_vae/msps_vae_model.py's DEFAULT_SPATIAL_LOSS_WEIGHT_R0 (msps-vae
# branch) -- same r0 means the same taper shape at any resolution, pixel grid or patch
# grid, by construction (see radial_taper's docstring)
DEFAULT_TAPER_R0 = 0.5


def patch_center_coords(grid_h: int, grid_w: int) -> torch.Tensor:
    """Patch-center coordinates in a `[-1, 1]` frame, row-major, matching token order.

    Args:
        grid_h: number of patch rows.
        grid_w: number of patch columns.

    Returns:
        float64 tensor, shape (grid_h * grid_w, 2), each row `(y, x)`.
    """
    ys = (torch.arange(grid_h, dtype=torch.float64) + 0.5) / grid_h * 2 - 1
    xs = (torch.arange(grid_w, dtype=torch.float64) + 0.5) / grid_w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([yy.flatten(), xx.flatten()], dim=-1)


def radial_taper(coords: torch.Tensor, r0: float = DEFAULT_TAPER_R0) -> torch.Tensor:
    """Raised-cosine radial taper, evaluated at continuous coordinates.

    Replicates `beast.models.msps_vae.msps_vae_model.build_raised_cosine_weight_map`
    exactly (cross-checked in `tests/embedders/test_spatial_weights.py`), evaluated at
    arbitrary `(y, x)` coordinates instead of a pixel grid: flat at 1.0 for `r <= r0`
    (`r = hypot(y, x)`, so `r = 1` sits at each edge's midpoint and corners sit at
    `r = sqrt(2)`), cosine-tapers to exactly 0 (value and slope) at `r = 1`, and is
    exactly 0 beyond -- corners fall past the taper's zero point with no separate
    corner logic needed. Unlike `build_raised_cosine_weight_map`, this does not
    renormalize to mean 1.0 -- that step only matters for loss-weighting, where it
    keeps MSE's scale fixed against a separate loss term; here, `patch_weights`
    renormalizes to sum 1 instead, for the weighted-moments math.

    Args:
        coords: coordinates in the same `[-1, 1]` frame as `patch_center_coords`,
            shape (..., 2).
        r0: normalized radius at which the taper begins; `0 <= r0 <= 1`.

    Returns:
        non-negative weights, shape `coords.shape[:-1]`.
    """
    r = coords.norm(dim=-1)
    w = torch.ones_like(r)
    tapered = (r > r0) & (r <= 1)
    w[tapered] = 0.5 * (1 + torch.cos(torch.pi * (r[tapered] - r0) / (1 - r0)))
    w[r > 1] = 0.0
    return w


def patch_weights(grid_h: int, grid_w: int, kind: str, **taper_kwargs: float) -> torch.Tensor:
    """Non-negative per-patch weights, summing to 1, in token order.

    Args:
        grid_h: number of patch rows.
        grid_w: number of patch columns.
        kind: `'uniform'` or `'taper'`.
        **taper_kwargs: forwarded to `radial_taper` when `kind == 'taper'` (e.g. `r0`).

    Returns:
        float64 tensor, shape (grid_h * grid_w,), summing to 1.

    Raises:
        ValueError: if `kind` isn't `'uniform'` or `'taper'`.
    """
    if kind == 'uniform':
        w = torch.ones(grid_h * grid_w, dtype=torch.float64)
    elif kind == 'taper':
        w = radial_taper(patch_center_coords(grid_h, grid_w), **taper_kwargs)
    else:
        raise ValueError(f'unknown patch_weights kind: {kind!r}; choices: uniform, taper')
    return w / w.sum()
