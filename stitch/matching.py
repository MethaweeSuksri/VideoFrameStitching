"""
Descriptor matching: Brute-Force vs FLANN, plus Lowe’s ratio test.

Combinations (see module docstring in original spec):

+------------------+-----------+------------------------------------------+
| Detector         | Matcher   | Notes                                    |
+==================+===========+==========================================+
| SIFT (float)     | BF        | NORM_L2                                  |
| SIFT (float)     | FLANN     | KD-tree; descriptors as float32        |
| ORB (binary)     | BF        | NORM_HAMMING                           |
| ORB (binary)     | FLANN     | **LSH** index (approximate); not KD-tree |
+------------------+-----------+------------------------------------------+

Using KD-tree FLANN on ORB is wrong: those trees assume Euclidean geometry in R^d.
LSH hashes binary-ish vectors; quality depends on ``table_number``, ``key_size``, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from stitch.features import DescriptorKind, DescriptorSpec


class MatchEngine(Protocol):
    def knn(self, query: np.ndarray, train: np.ndarray, k: int) -> list[list[cv2.DMatch]]:
        ...


@dataclass
class BruteForce:
    norm: int

    def knn(self, query: np.ndarray, train: np.ndarray, k: int) -> list[list[cv2.DMatch]]:
        bf = cv2.BFMatcher(self.norm, crossCheck=False)
        return bf.knnMatch(query, train, k=k)


@dataclass
class FlannKnn:
    index_params: dict
    search_params: dict

    def knn(self, query: np.ndarray, train: np.ndarray, k: int) -> list[list[cv2.DMatch]]:
        q = np.asarray(query)
        t = np.asarray(train)
        # LSH (algorithm=6) expects uint8 rows for ORB; KD-tree needs float32 for SIFT.
        if int(self.index_params.get("algorithm", 1)) != 6:
            q = np.ascontiguousarray(q, dtype=np.float32)
            t = np.ascontiguousarray(t, dtype=np.float32)
        flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)
        return flann.knnMatch(q, t, k=k)


def build_matcher(name: str, spec: DescriptorSpec) -> MatchEngine:
    key = name.lower()
    if key in ("bf", "brute", "bfmatcher"):
        norm = cv2.NORM_L2 if spec.kind == DescriptorKind.FLOAT else cv2.NORM_HAMMING
        return BruteForce(norm)

    if key == "flann":
        if spec.kind == DescriptorKind.FLOAT:
            return FlannKnn(
                index_params={"algorithm": 1, "trees": 5},
                search_params={"checks": 64},
            )
        # ORB: FLANN_INDEX_LSH — approximate Hamming-neighborhood search
        return FlannKnn(
            index_params={
                "algorithm": 6,
                "table_number": 8,
                "key_size": 12,
                "multi_probe_level": 1,
            },
            search_params={"checks": 48},
        )

    raise ValueError(f"Unknown matcher: {name}")


def lowe_ratio_filter(knn_pairs: list[list[cv2.DMatch]], ratio: float) -> list[cv2.DMatch]:
    """Keep match m if d1 < ratio * d2 for the two nearest neighbors."""
    good: list[cv2.DMatch] = []
    for pair in knn_pairs:
        if len(pair) < 2:
            continue
        a, b = pair[0], pair[1]
        if a.distance < ratio * b.distance:
            good.append(a)
    return good


def match_with_ratio(
    desc_query: np.ndarray,
    desc_train: np.ndarray,
    engine: MatchEngine,
    *,
    ratio: float,
) -> tuple[list[cv2.DMatch], list[list[cv2.DMatch]]]:
    """
    Match **query** descriptors against **train** descriptors.

    OpenCV convention: ``DMatch.queryIdx`` indexes the **query** set,
    ``trainIdx`` indexes the **train** set.
    """
    if desc_query is None or desc_train is None:
        return [], []
    if len(desc_query) < 2 or len(desc_train) < 2:
        return [], []

    raw = engine.knn(desc_query, desc_train, k=2)
    good = lowe_ratio_filter(raw, ratio)
    return good, raw
