"""Shared fixtures for cuttle_patterns.preprocessing tests."""

from collections.abc import Callable

import numpy as np
import pytest


@pytest.fixture
def make_mask() -> Callable[..., np.ndarray]:
    """Return a factory that builds a binary mask with a single filled rectangle.

    Returns:
        a callable taking (shape, rect) -> mask, where shape is (height, width) and
        rect is (y0, y1, x0, x1). Pixels inside rect are 255, everywhere else 0.
    """

    def _make_mask(shape: tuple[int, int], rect: tuple[int, int, int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        y0, y1, x0, x1 = rect
        mask[y0:y1, x0:x1] = 255
        return mask

    return _make_mask
