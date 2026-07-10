"""Inscribe a fixed-aspect-ratio rectangle inside a cuttlefish body mask.

The rectangle is aligned to the body's principal axis rather than the image axes, and its
aspect ratio is fixed at 2:1 so that box shape doesn't add its own noise on top of
body-position/orientation noise. By default (Phase 2a) the axis comes from PCA on the
mask, which leaves head/tail direction unresolved and is biased by extended arms (PCA is
free to find its largest rectangle straddling the arm crown rather than the mantle). When
per-frame tail/neck keypoints are available, `inscribe_rectangle` switches to the
pose-informed Phase 2b path instead, which resolves both issues. See `docs/PHASES.md`
(Phase 2a/2b) for the surrounding pipeline.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# background pixels are confirmed pure black (value 0, no compression noise observed in
# spot checks); see docs/PHASES.md Phase 2a for how this was verified
DEFAULT_THRESHOLD = 0
DEFAULT_ASPECT_RATIO = 2.0
DEFAULT_N_REFINE_STEPS = 20


@dataclass
class InscribedRect:
    """A rectangle inscribed in a body mask, in original-frame coordinates.

    Attributes:
        corners: (4, 2) array of (x, y) corners, clockwise from top-left.
        center: (cx, cy) of the rectangle in the original frame.
        angle: rotation angle in degrees of the body's principal axis.
        width: rectangle width in pixels (long side).
        height: rectangle height in pixels (short side); width == aspect * height.
    """

    corners: np.ndarray
    center: tuple[float, float]
    angle: float
    width: float
    height: float


def threshold_body_mask(frame: np.ndarray, thresh: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Recover the body mask as the complement of the frame's largest background blob.

    The background is pure black (pixel value 0), but some of the body's own dark skin
    patterning also renders at or near 0 — naively thresholding on intensity alone can't
    tell those apart, since the pixel values are identical. Instead, this labels
    connected components of near-black pixels and treats everything outside the single
    largest one (the true background) as body: small dark patches on the animal don't
    connect to the real background, so they end up on the body side of the split
    regardless of whether they're fully enclosed by lighter pixels. Assumes a single
    dominant background region per frame (holds as long as the body doesn't split the
    frame into disconnected background pockets).

    Args:
        frame: decoded video frame, grayscale or BGR.
        thresh: pixel intensities <= this value are candidate background pixels.

    Returns:
        binary mask (0/255, uint8) of the largest connected foreground component after
        the background split. All zeros if no foreground pixels are found.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bg_candidates = np.where(gray <= thresh, np.uint8(255), np.uint8(0))
    n_bg_labels, bg_labels, bg_stats, _ = cv2.connectedComponentsWithStats(
        bg_candidates, connectivity=8,
    )
    if n_bg_labels <= 1:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
    else:
        idx_largest_bg = 1 + np.argmax(bg_stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(bg_labels == idx_largest_bg, np.uint8(0), np.uint8(255))

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask)

    idx_largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == idx_largest, np.uint8(255), np.uint8(0))


def body_orientation(mask: np.ndarray) -> tuple[float, float, float]:
    """Estimate the body centroid and principal axis angle via PCA on foreground pixels.

    Args:
        mask: binary body mask (0/255).

    Returns:
        (cx, cy, angle) where angle is the major-axis angle in degrees. Direction along
        the axis (head vs. tail) is not resolved — see Phase 2b in `docs/PHASES.md`.
    """
    ys, xs = np.nonzero(mask)
    points = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = points.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(np.cov((points - centroid).T))
    major_axis = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))

    return float(centroid[0]), float(centroid[1]), angle


def body_orientation_signed(
    tail: tuple[float, float],
    neck: tuple[float, float],
) -> tuple[float, float, float]:
    """Estimate body center and a signed axis angle from tail/neck keypoints.

    Signed counterpart to `body_orientation`: since `neck - tail` has a known direction
    (unlike a PCA major axis), the resulting angle resolves head/tail directly, with no
    separate direction-disambiguation step. See Phase 2b in `docs/PHASES.md`.

    Args:
        tail: (x, y) tip of the mantle, opposite the head.
        neck: (x, y) head/body transition point.

    Returns:
        (cx, cy, angle): the tail/neck midpoint and the signed angle in degrees such
        that, after `rotate_mask_upright`, the neck lands on the +x side of center and
        the tail on the -x side.
    """
    cx = (tail[0] + neck[0]) / 2
    cy = (tail[1] + neck[1]) / 2
    angle = float(np.degrees(np.arctan2(neck[1] - tail[1], neck[0] - tail[0])))

    return cx, cy, angle


def cut_mask_at_neck(mask: np.ndarray, neck_x: float) -> np.ndarray:
    """Zero out the head/arm side of a rotated mask, keeping only the tail/mantle side.

    Assumes the mask was rotated upright via `rotate_mask_upright` fed a signed
    tail->neck angle (see `body_orientation_signed`), so the head/arms are the +x side of
    `neck_x`. This is the step that fixes the arm-posture bias described in Phase 2a: the
    rectangle-growing step downstream only ever sees the mantle.

    Args:
        mask: binary body mask (0/255), already rotated upright.
        neck_x: x-coordinate of the neck keypoint in the rotated mask's frame.

    Returns:
        a copy of mask with every column at or past `round(neck_x)` set to 0.
    """
    cut = mask.copy()
    cut[:, round(neck_x):] = 0
    return cut


def rotate_mask_upright(
    mask: np.ndarray,
    center: tuple[float, float],
    angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a mask so the given center is at the canvas center and angle is horizontal.

    The output canvas is padded to the input's diagonal length so the rotated body can't
    clip off the edge.

    Args:
        mask: binary body mask (0/255).
        center: (cx, cy) point to rotate about (typically the body centroid).
        angle: angle in degrees to rotate to horizontal (typically the PCA axis angle).

    Returns:
        rotated mask on a padded canvas, and the affine matrix mapping rotated-frame
        coordinates back to the original frame.
    """
    diag = int(np.ceil(np.hypot(*mask.shape)))
    pad_center = (diag / 2, diag / 2)

    m_fwd = cv2.getRotationMatrix2D(center, angle, 1.0)
    m_fwd[0, 2] += pad_center[0] - center[0]
    m_fwd[1, 2] += pad_center[1] - center[1]

    rotated = cv2.warpAffine(mask, m_fwd, (diag, diag), flags=cv2.INTER_NEAREST)
    m_inv = cv2.invertAffineTransform(m_fwd)

    return rotated, m_inv


def seed_from_distance_transform(mask: np.ndarray) -> tuple[int, int]:
    """Find the center of the largest inscribed circle, as a rectangle-search seed.

    Args:
        mask: binary body mask (0/255).

    Returns:
        (x, y) location of the distance-transform peak.
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return int(x), int(y)


def rect_fully_inside(integral: np.ndarray, x0: int, y0: int, w: int, h: int) -> bool:
    """Check whether an axis-aligned box is fully foreground, via a summed-area table.

    Args:
        integral: integral image of a binary (0/255) mask, from `cv2.integral`.
        x0: left edge of the box.
        y0: top edge of the box.
        w: box width.
        h: box height.

    Returns:
        True if every pixel covered by the box is foreground.
    """
    x1, y1 = x0 + w, y0 + h
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return total >= 255 * w * h


def grow_rectangle(
    mask: np.ndarray,
    seed: tuple[int, int],
    aspect: float = DEFAULT_ASPECT_RATIO,
    n_refine_steps: int = DEFAULT_N_REFINE_STEPS,
) -> tuple[float, float, float, float]:
    """Grow the largest fixed-aspect-ratio rectangle inscribed in a mask, from a seed.

    Alternates binary-searching the largest scale at a fixed center with a coordinate-
    ascent step on the center, since the distance-transform seed (the largest inscribed
    circle's center) is a good but not necessarily optimal center for a non-square
    rectangle.

    Args:
        mask: binary body mask (0/255), already rotated upright.
        seed: (x, y) starting center, typically from `seed_from_distance_transform`.
        aspect: fixed width-to-height ratio of the rectangle.
        n_refine_steps: number of coordinate-ascent refinement iterations.

    Returns:
        (cx, cy, width, height) of the largest rectangle found, in mask pixel
        coordinates.
    """
    integral = cv2.integral(mask)

    def max_height_at(cx: float, cy: float) -> float:
        # binary search the largest height h (width = aspect * h) centered at (cx, cy)
        lo, hi = 0.0, float(min(mask.shape))
        for _ in range(30):
            mid = (lo + hi) / 2
            w, h = aspect * mid, mid
            x0, y0 = int(cx - w / 2), int(cy - h / 2)
            x1, y1 = int(cx + w / 2), int(cy + h / 2)
            in_bounds = 0 <= x0 and x1 <= mask.shape[1] and 0 <= y0 and y1 <= mask.shape[0]
            if in_bounds and rect_fully_inside(integral, x0, y0, x1 - x0, y1 - y0):
                lo = mid
            else:
                hi = mid
        return lo

    cx, cy = float(seed[0]), float(seed[1])
    best_h = max_height_at(cx, cy)
    step = min(mask.shape) / 8.0

    for _ in range(n_refine_steps):
        improved = False
        for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
            h = max_height_at(cx + dx, cy + dy)
            if h > best_h:
                best_h, cx, cy = h, cx + dx, cy + dy
                improved = True
        if not improved:
            step *= 0.5

    return cx, cy, aspect * best_h, best_h


def inscribe_rectangle(
    frame: np.ndarray,
    thresh: int = DEFAULT_THRESHOLD,
    aspect: float = DEFAULT_ASPECT_RATIO,
    tail: tuple[float, float] | None = None,
    neck: tuple[float, float] | None = None,
) -> InscribedRect | None:
    """Inscribe the largest fixed-aspect-ratio rectangle inside a frame's body mask.

    Recovers the body mask (see `threshold_body_mask`), rotates it upright, seeds a
    candidate rectangle from the distance-transform peak, and grows/refines it before
    mapping the result back to original-frame coordinates.

    When `tail`/`neck` are both given (Phase 2b), orientation comes from
    `body_orientation_signed` instead of PCA, and the rotated mask is cut at the neck
    (`cut_mask_at_neck`) before sizing, so the rectangle search never sees the head/arm
    side. See Phase 2b in `docs/PHASES.md`.

    Args:
        frame: decoded video frame, grayscale or BGR.
        thresh: pixel intensities <= this value are candidate background pixels; see
            `threshold_body_mask`.
        aspect: fixed width-to-height ratio of the rectangle.
        tail: (x, y) tip of the mantle; if given along with `neck`, switches to the
            pose-informed Phase 2b path.
        neck: (x, y) head/body transition point; if given along with `tail`, switches to
            the pose-informed Phase 2b path.

    Returns:
        the inscribed rectangle in original-frame coordinates, or None if no foreground
        body (or, in the pose-informed path, no mantle) is found.
    """
    mask = threshold_body_mask(frame, thresh)
    if not mask.any():
        return None

    use_pose = tail is not None and neck is not None
    if use_pose:
        cx, cy, angle = body_orientation_signed(tail, neck)
    else:
        cx, cy, angle = body_orientation(mask)
    rotated, m_inv = rotate_mask_upright(mask, (cx, cy), angle)

    if use_pose:
        m_fwd = cv2.invertAffineTransform(m_inv)
        neck_rotated = cv2.transform(np.array([[neck]], dtype=np.float64), m_fwd)[0, 0]
        rotated = cut_mask_at_neck(rotated, neck_rotated[0])
        if not rotated.any():
            return None

    seed = seed_from_distance_transform(rotated)
    rcx, rcy, w, h = grow_rectangle(rotated, seed, aspect=aspect)

    local_corners = np.array(
        [
            [rcx - w / 2, rcy - h / 2],
            [rcx + w / 2, rcy - h / 2],
            [rcx + w / 2, rcy + h / 2],
            [rcx - w / 2, rcy + h / 2],
        ]
    )
    corners = cv2.transform(local_corners[None, :, :], m_inv)[0]

    return InscribedRect(
        corners=corners,
        center=(float(corners[:, 0].mean()), float(corners[:, 1].mean())),
        angle=angle,
        width=w,
        height=h,
    )
