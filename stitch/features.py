"""
Local feature backends (OpenCV).

Distance semantics (must match the matcher):

- **SIFT**: 128-D **float** vectors → L2 / Euclidean distance.
- **ORB**: binary strings (BRIEF-style) → **Hamming** distance (bit mismatches).

``DescriptorKind`` drives BF norm and FLANN index type in ``stitch.matching``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import cv2
import numpy as np


class DescriptorKind(Enum):
    FLOAT = "float"
    BINARY = "binary"


@dataclass(frozen=True)
class DescriptorSpec:
    kind: DescriptorKind
    name: str


class FeatureBackend(Protocol):
    """Detect keypoints and compute descriptors on a single-channel uint8 image."""

    spec: DescriptorSpec

    def detect_and_compute(self, gray_u8: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        ...


@dataclass
class _CvBackend:
    _algo: cv2.Feature2D
    spec: DescriptorSpec

    def detect_and_compute(self, gray_u8: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        return self._algo.detectAndCompute(gray_u8, None)


def make_sift(
    n_features: int = 800,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
) -> FeatureBackend:
    n = n_features if n_features > 0 else 800
    sift = cv2.SIFT_create(
        nfeatures=n,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
    )
    return _CvBackend(sift, DescriptorSpec(DescriptorKind.FLOAT, "sift"))


def make_orb(n_features: int = 2500, scale_factor: float = 1.2, n_levels: int = 8) -> FeatureBackend:
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)
    return _CvBackend(orb, DescriptorSpec(DescriptorKind.BINARY, "orb"))


def make_backend(name: str) -> FeatureBackend:
    n = name.lower()
    if n == "sift":
        return make_sift()
    if n == "orb":
        return make_orb()
    raise ValueError(f"Unknown detector: {name}")
