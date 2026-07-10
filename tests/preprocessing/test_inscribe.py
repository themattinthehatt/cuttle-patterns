"""Tests for cuttle_patterns.preprocessing.inscribe."""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from cuttle_patterns.preprocessing.inscribe import (
    body_orientation,
    body_orientation_signed,
    cut_mask_at_neck,
    grow_rectangle,
    inscribe_rectangle,
    rect_fully_inside,
    rotate_mask_upright,
    seed_from_distance_transform,
    threshold_body_mask,
)


class TestThresholdBodyMask:
    """Test the function threshold_body_mask."""

    def test_threshold_body_mask_recovers_enclosed_dark_patch(self):
        # Arrange: a foreground square with a fully-enclosed zero-valued patch
        frame = np.zeros((50, 50), dtype=np.uint8)
        frame[10:40, 10:40] = 100
        frame[20:25, 20:25] = 0

        # Act
        mask = threshold_body_mask(frame)

        # Assert: the patch isn't connected to the real background, so it's recovered
        assert mask[22, 22] == 255
        assert mask[10:40, 10:40].all()

    def test_threshold_body_mask_keeps_largest_component(self):
        # Arrange: two disconnected foreground blobs of different sizes
        frame = np.zeros((50, 50), dtype=np.uint8)
        frame[5:10, 5:10] = 100
        frame[20:40, 20:40] = 100

        # Act
        mask = threshold_body_mask(frame)

        # Assert
        assert mask[25, 25] == 255
        assert mask[7, 7] == 0

    def test_threshold_body_mask_empty_frame(self):
        # Arrange
        frame = np.zeros((20, 20), dtype=np.uint8)

        # Act
        mask = threshold_body_mask(frame)

        # Assert
        assert not mask.any()

    def test_threshold_body_mask_all_foreground(self):
        # Arrange: no background pixels at all
        frame = np.full((20, 20), 100, dtype=np.uint8)

        # Act
        mask = threshold_body_mask(frame)

        # Assert
        assert mask.all()


class TestBodyOrientation:
    """Test the function body_orientation."""

    def test_body_orientation_horizontal_rectangle(self, make_mask: Callable):
        # Arrange
        mask = make_mask((50, 100), (15, 35, 10, 90))

        # Act
        cx, cy, angle = body_orientation(mask)

        # Assert
        assert cx == pytest.approx(49.5, abs=1.0)
        assert cy == pytest.approx(24.5, abs=1.0)
        assert angle % 180 == pytest.approx(0.0, abs=1.0)

    def test_body_orientation_vertical_rectangle(self, make_mask: Callable):
        # Arrange
        mask = make_mask((100, 50), (10, 90, 15, 35))

        # Act
        _, _, angle = body_orientation(mask)

        # Assert
        assert abs(angle) % 180 == pytest.approx(90.0, abs=1.0)


class TestBodyOrientationSigned:
    """Test the function body_orientation_signed."""

    def test_body_orientation_signed_midpoint_center(self):
        # Act
        cx, cy, _ = body_orientation_signed(tail=(0.0, 0.0), neck=(10.0, 20.0))

        # Assert
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(10.0)

    def test_body_orientation_signed_neck_to_the_right_is_zero(self):
        # Act
        _, _, angle = body_orientation_signed(tail=(0.0, 0.0), neck=(10.0, 0.0))

        # Assert
        assert angle == pytest.approx(0.0)

    def test_body_orientation_signed_neck_to_the_left_is_180(self):
        # Act
        _, _, angle = body_orientation_signed(tail=(10.0, 0.0), neck=(0.0, 0.0))

        # Assert
        assert abs(angle) == pytest.approx(180.0)


class TestCutMaskAtNeck:
    """Test the function cut_mask_at_neck."""

    def test_cut_mask_at_neck_zeroes_head_side(self):
        # Arrange
        mask = np.full((50, 100), 255, dtype=np.uint8)

        # Act
        cut = cut_mask_at_neck(mask, neck_x=60)

        # Assert
        assert cut[:, :60].all()
        assert not cut[:, 60:].any()


class TestRotateMaskUpright:
    """Test the function rotate_mask_upright."""

    def test_rotate_mask_upright_preserves_area_at_zero_angle(self, make_mask: Callable):
        # Arrange
        mask = make_mask((50, 100), (15, 35, 10, 90))
        diag = int(np.ceil(np.hypot(*mask.shape)))

        # Act
        rotated, m_inv = rotate_mask_upright(mask, center=(49.5, 24.5), angle=0.0)

        # Assert
        assert rotated.shape == (diag, diag)
        assert int((rotated == 255).sum()) == int((mask == 255).sum())
        assert m_inv.shape == (2, 3)


class TestSeedFromDistanceTransform:
    """Test the function seed_from_distance_transform."""

    def test_seed_from_distance_transform_centered_square(self, make_mask: Callable):
        # Arrange
        mask = make_mask((60, 60), (10, 50, 10, 50))

        # Act
        x, y = seed_from_distance_transform(mask)

        # Assert
        assert x == pytest.approx(29.5, abs=2)
        assert y == pytest.approx(29.5, abs=2)


class TestRectFullyInside:
    """Test the function rect_fully_inside."""

    def test_rect_fully_inside_true_for_all_foreground(self):
        # Arrange
        mask = np.full((50, 50), 255, dtype=np.uint8)
        integral = cv2.integral(mask)

        # Act & Assert
        assert rect_fully_inside(integral, 5, 5, 20, 20)

    def test_rect_fully_inside_false_when_box_contains_background(self):
        # Arrange
        mask = np.full((50, 50), 255, dtype=np.uint8)
        mask[10, 10] = 0
        integral = cv2.integral(mask)

        # Act & Assert
        assert not rect_fully_inside(integral, 5, 5, 20, 20)


class TestGrowRectangle:
    """Test the function grow_rectangle."""

    def test_grow_rectangle_fills_full_mask(self):
        # Arrange: 100x200 all-foreground mask, height-limited at aspect 2:1
        mask = np.full((100, 200), 255, dtype=np.uint8)

        # Act
        _, _, w, h = grow_rectangle(mask, seed=(100, 50), aspect=2.0)

        # Assert
        assert h == pytest.approx(100.0, abs=1.0)
        assert w == pytest.approx(200.0, abs=1.0)

    def test_grow_rectangle_respects_a_hole(self):
        # Arrange: a hole right at the seed forces the rectangle to shrink or relocate
        mask = np.full((100, 200), 255, dtype=np.uint8)
        mask[45:55, 95:105] = 0

        # Act
        _, _, _, h = grow_rectangle(mask, seed=(100, 50), aspect=2.0)

        # Assert
        assert h < 100.0


class TestInscribeRectangle:
    """Test the function inscribe_rectangle."""

    def test_inscribe_rectangle_finds_box_in_ellipse(self):
        # Arrange
        frame = np.zeros((400, 600), dtype=np.uint8)
        cv2.ellipse(frame, (300, 200), (150, 60), 30, 0, 360, 100, -1)

        # Act
        result = inscribe_rectangle(frame)

        # Assert
        assert result is not None
        assert result.corners.shape == (4, 2)
        assert result.width == pytest.approx(2 * result.height, rel=0.01)
        assert result.width > 0

    def test_inscribe_rectangle_respects_custom_aspect(self):
        # Arrange
        frame = np.zeros((400, 600), dtype=np.uint8)
        cv2.ellipse(frame, (300, 200), (150, 60), 30, 0, 360, 100, -1)

        # Act
        result = inscribe_rectangle(frame, aspect=1.0)

        # Assert
        assert result is not None
        assert result.width == pytest.approx(result.height, rel=0.01)

    def test_inscribe_rectangle_no_body(self):
        # Arrange
        frame = np.zeros((50, 50), dtype=np.uint8)

        # Act
        result = inscribe_rectangle(frame)

        # Assert
        assert result is None

    def test_inscribe_rectangle_with_pose_avoids_arm_bias(self):
        # Arrange: a mantle ellipse with a large "arm crown" blob attached at its right
        # edge (x=280) — PCA over the combined mask is known to straddle the arm side,
        # which pose-informed cutting at the neck should avoid entirely
        frame = np.zeros((400, 500), dtype=np.uint8)
        cv2.ellipse(frame, (200, 200), (80, 40), 0, 0, 360, 100, -1)
        cv2.rectangle(frame, (280, 150), (450, 250), 100, -1)

        # Act
        pca_result = inscribe_rectangle(frame)
        pose_result = inscribe_rectangle(frame, tail=(120.0, 200.0), neck=(280.0, 200.0))

        # Assert: PCA lands in the arm region, pose-informed stays on the mantle side
        assert pca_result is not None
        assert pca_result.corners[:, 0].max() > 280
        assert pose_result is not None
        assert pose_result.corners[:, 0].max() <= 280

    def test_inscribe_rectangle_with_pose_no_mantle_left_returns_none(self):
        # Arrange: tail/neck sit well clear of the mask entirely, on the side that
        # cut_mask_at_neck keeps — so every actual body pixel falls on the discarded
        # (head/arm) side and nothing is left to size a rectangle from
        frame = np.zeros((400, 500), dtype=np.uint8)
        cv2.ellipse(frame, (200, 200), (80, 40), 0, 0, 360, 100, -1)

        # Act
        result = inscribe_rectangle(frame, tail=(0.0, 200.0), neck=(0.0, 200.0))

        # Assert
        assert result is None
