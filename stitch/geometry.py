"""
Homography estimation and multi-frame composition.

Coordinate convention (important for reading the code):

- We estimate **H** with ``cv2.findHomography(src, dst)`` so that, in homogeneous coords,
  **p_dst ≈ H @ p_src** (src frame pixel → dst frame pixel).
- For consecutive video frames **i** (earlier) and **i+1** (later), we match features on
  both and set **src = frame i+1**, **dst = frame i**. Then **p_i ≈ H @ p_{i+1}**.

Chaining:

- Let **G_k** map frame **k** into frame **0**’s plane: **p_0 = G_k @ p_k**.
- With **H_k** mapping **k+1 → k**: **p_k = H_k @ p_{k+1}**, so
  **G_{k+1} = G_k @ H_k** with **G_0 = I**.

Middle reference (often *looks* better):

- After computing **G_k** into frame 0, pick **r = n // 2** and use
  **G'_k = inv(G_r) @ G_k** so frame **r** becomes the identity (upright center).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HomographyEstimate:
    H: np.ndarray | None
    mask: np.ndarray | None
    inliers: int
    matches_used: int
    status: Literal["ok", "few_matches", "rejected", "degenerate"]


def homography_is_plausible(
    H: np.ndarray,
    *,
    max_singular_ratio: float,
    max_scale: float,
    min_scale: float,
) -> bool:
    """
    Drop extreme projective maps (wild zoom, inversion, insane shear).

    We inspect the upper-left 2×2 of **H** (linear part of the affine-ish core).
    """
    if not np.all(np.isfinite(H)):
        return False
    A = H[:2, :2].astype(np.float64)
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    smax, smin = float(s[0]), float(s[-1])
    if smin < 1e-8:
        return False
    if smax / smin > max_singular_ratio:
        return False
    if smax > max_scale or smin < min_scale:
        return False
    return True


def estimate_pair_homography(
    kp_src: list[cv2.KeyPoint],
    kp_dst: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    *,
    ransac_thresh: float,
    max_iter: int = 5000,
    confidence: float = 0.995,
    max_singular_ratio: float = 6.0,
    max_scale: float = 8.0,
    min_scale: float = 0.05,
) -> HomographyEstimate:
    """
    **src** keypoints matched to **dst**; returns **H** with **p_dst ≈ H @ p_src**.
    """
    n = len(matches)
    if n < 4:
        return HomographyEstimate(None, None, 0, n, "few_matches")

    src = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=max_iter,
        confidence=confidence,
    )
    if H is None:
        return HomographyEstimate(None, None, 0, n, "rejected")

    inl = int(mask.ravel().sum()) if mask is not None else 0
    if inl < 4:
        return HomographyEstimate(None, mask, inl, n, "degenerate")

    if not homography_is_plausible(
        H,
        max_singular_ratio=max_singular_ratio,
        max_scale=max_scale,
        min_scale=min_scale,
    ):
        logger.debug("Homography failed plausibility check; treating as rejected")
        return HomographyEstimate(None, mask, inl, n, "rejected")

    return HomographyEstimate(H.astype(np.float64), mask, inl, n, "ok")


def chain_to_first_frame(pairwise_hi_to_him1: list[np.ndarray]) -> list[np.ndarray]:
    """
    **pairwise[i]**: maps frame **i+1 → i**. Returns **G[k]**: frame **k → 0**.
    """
    if not pairwise_hi_to_him1:
        return [np.eye(3, dtype=np.float64)]

    out: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    acc = np.eye(3, dtype=np.float64)
    for H in pairwise_hi_to_him1:
        acc = acc @ H
        out.append(acc.copy())
    return out


def recenter_to_middle(G_into_0: list[np.ndarray]) -> list[np.ndarray]:
    """Apply **inv(G_mid) @ G_k** so the middle frame becomes the reference."""
    n = len(G_into_0)
    if n == 0:
        return []
    mid = n // 2
    Gm = G_into_0[mid]
    try:
        Gm_inv = np.linalg.inv(Gm)
    except np.linalg.LinAlgError:
        logger.warning("Could not invert G_mid; keeping first-frame reference")
        return G_into_0
    return [Gm_inv @ Gk for Gk in G_into_0]


def save_homographies_npz(path: Path | str, global_matrices: list[np.ndarray], pairwise: list[np.ndarray]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pw = np.stack(pairwise, axis=0) if pairwise else np.zeros((0, 3, 3), dtype=np.float64)
    np.savez_compressed(p, global_H=np.stack(global_matrices, axis=0), pairwise_H=pw)
