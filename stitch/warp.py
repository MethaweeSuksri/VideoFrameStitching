"""
Warp every frame into one floating canvas, then blend.

Canvas sizing: transform the four image corners with each **G_k** (frame k → reference),
take the axis-aligned bounding box, add margin, then shift by a translation **T** so the
top-left of the bbox is a few pixels inside the canvas. That avoids excessive cropping while
keeping coordinates well-conditioned for ``warpPerspective``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _image_corners_xy(width: int, height: int) -> np.ndarray:
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)


def apply_homography_xy(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """Apply 3×3 **H** to N×2 points; returns N×2 finite Cartesian coordinates."""
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    hom = np.c_[pts, np.ones(len(pts))]
    w = (H.astype(np.float64) @ hom.T).T
    wc = w[:, 2:3]
    wc = np.where(np.abs(wc) < 1e-12, 1e-12, wc)
    return w[:, :2] / wc


@dataclass(frozen=True)
class Canvas:
    width: int
    height: int
    """Full transform from frame pixels to canvas pixels: **M = T_offset @ G_k**."""
    T_offset: np.ndarray


def build_canvas(
    sizes_hw: list[tuple[int, int]],
    global_maps: list[np.ndarray],
    margin_px: int,
) -> Canvas:
    """
    **sizes_hw[i]** = (height, width) of frame **i**; **global_maps[i]** = frame **i → ref**.
    """
    all_pts: list[np.ndarray] = []
    for (h, w), G in zip(sizes_hw, global_maps, strict=True):
        corners = _image_corners_xy(w, h)
        all_pts.append(apply_homography_xy(G, corners))

    pts = np.vstack(all_pts)
    xmin, ymin = np.floor(pts.min(axis=0) - margin_px).astype(int)
    xmax, ymax = np.ceil(pts.max(axis=0) + margin_px).astype(int)

    width = max(2, int(xmax - xmin))   # even width helps video encoders
    height = max(2, int(ymax - ymin))
    if width % 2:
        width += 1
    if height % 2:
        height += 1

    T = np.array(
        [[1.0, 0.0, float(-xmin)], [0.0, 1.0, float(-ymin)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return Canvas(width=width, height=height, T_offset=T)


def warp_to_canvas(
    image_bgr: np.ndarray,
    G_frame_to_ref: np.ndarray,
    canvas: Canvas,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warp **image_bgr** with **T_offset @ G**. Returns (warped BGR, float alpha mask in [0,1]).
    """
    M = canvas.T_offset @ G_frame_to_ref
    warped = cv2.warpPerspective(
        image_bgr,
        M,
        (canvas.width, canvas.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    alpha = (gray > 0).astype(np.float32)
    return warped, alpha


def feather_alpha(alpha: np.ndarray, radius_px: int) -> np.ndarray:
    """Soft edges for blending: distance falloff inside the valid (alpha>0) region."""
    if radius_px <= 0:
        return alpha
    mask = (alpha > 0.5).astype(np.uint8)
    if not np.any(mask):
        return alpha
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    w = np.clip(dist / float(radius_px), 0.0, 1.0).astype(np.float32)
    return w * alpha


def blend_weighted_average(
    warped_images: list[np.ndarray],
    alphas: list[np.ndarray],
    feather_radius: int,
) -> np.ndarray:
    """Σ (image * weight) / Σ weight — reduces seams when **feather_radius** > 0."""
    if not warped_images:
        raise ValueError("no images")

    acc = np.zeros_like(warped_images[0], dtype=np.float64)
    wsum = np.zeros(warped_images[0].shape[:2], dtype=np.float64)

    for im, a in zip(warped_images, alphas, strict=True):
        w = feather_alpha(a, feather_radius) if feather_radius > 0 else a
        acc += im.astype(np.float64) * w[..., np.newaxis]
        wsum += w

    wsum = np.maximum(wsum, 1e-9)
    out = (acc / wsum[..., np.newaxis]).clip(0, 255).astype(np.uint8)
    return out
