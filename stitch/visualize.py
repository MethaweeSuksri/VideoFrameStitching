"""Debug visuals: keypoints, correspondences, inlier coloring."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def draw_keypoints(bgr: np.ndarray, keypoints: list[cv2.KeyPoint], color=(0, 255, 0)) -> np.ndarray:
    return cv2.drawKeypoints(bgr, keypoints, None, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def draw_matches_side_by_side(
    img_q: np.ndarray,
    kp_q: list[cv2.KeyPoint],
    img_t: np.ndarray,
    kp_t: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    *,
    max_count: int = 60,
) -> np.ndarray:
    return cv2.drawMatches(
        img_q,
        kp_q,
        img_t,
        kp_t,
        matches[:max_count],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_matches_inlier_outlier(
    img_q: np.ndarray,
    kp_q: list[cv2.KeyPoint],
    img_t: np.ndarray,
    kp_t: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    ransac_mask: np.ndarray | None,
    *,
    max_inliers: int = 80,
    max_outliers: int = 30,
) -> np.ndarray:
    if ransac_mask is None or len(matches) != len(ransac_mask.ravel()):
        return draw_matches_side_by_side(img_q, kp_q, img_t, kp_t, matches, max_count=max_inliers)

    flags = np.asarray(ransac_mask).ravel().astype(bool)
    inl = [m for m, ok in zip(matches, flags, strict=True) if ok][:max_inliers]
    out = [m for m, ok in zip(matches, flags, strict=True) if not ok][:max_outliers]

    canvas = cv2.drawMatches(
        img_q,
        kp_q,
        img_t,
        kp_t,
        inl,
        None,
        matchColor=(0, 220, 0),
        singlePointColor=(0, 220, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if out:
        canvas = cv2.drawMatches(
            img_q,
            kp_q,
            img_t,
            kp_t,
            out,
            canvas,
            matchColor=(40, 40, 255),
            singlePointColor=(40, 40, 255),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    return canvas


def imwrite(path: Path | str, bgr: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), bgr)


def compose_alignment_overview(
    warped_bgr: list[np.ndarray],
    alphas: list[np.ndarray],
    current_idx: int,
    *,
    non_current_weight: float = 0.01,
    border_bgr: tuple[int, int, int] = (255, 255, 255),
    border_thickness: int = 4,
    show_labels: bool = True,
) -> np.ndarray:
    """
    One video frame: non-current warps are added at **non_current_weight** onto black (true
    “~opacity”, not a normalized average). The **current_idx** warp is alpha-composited on
    top at full strength. White border and optional ``#i`` labels at centroids.
    """
    if not warped_bgr or len(warped_bgr) != len(alphas):
        raise ValueError("warped_bgr and alphas must be same length")
    n = len(warped_bgr)
    if not (0 <= current_idx < n):
        raise IndexError("current_idx out of range")

    h, w = warped_bgr[0].shape[:2]
    wc = float(non_current_weight)
    base = np.zeros((h, w, 3), dtype=np.float64)

    for i, (im, a) in enumerate(zip(warped_bgr, alphas, strict=True)):
        if i == current_idx:
            continue
        a64 = a.astype(np.float64)[..., np.newaxis]
        base += im.astype(np.float64) * a64 * wc

    base = np.clip(base, 0.0, 255.0)

    im_c = warped_bgr[current_idx].astype(np.float64)
    a_c = alphas[current_idx].astype(np.float64)[..., np.newaxis]
    out = base * (1.0 - a_c) + im_c * a_c
    out = out.clip(0, 255).astype(np.uint8)

    m = (alphas[current_idx] > 0.5).astype(np.uint8) * 255
    if int(cv2.countNonZero(m)) > 0:
        x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(m))
        cv2.rectangle(
            out,
            (x, y),
            (x + bw - 1, y + bh - 1),
            border_bgr,
            border_thickness,
            lineType=cv2.LINE_AA,
        )

    if show_labels:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = min(w, h) / 900.0
        fs = float(np.clip(fs, 0.45, 1.2))
        for i, a in enumerate(alphas):
            mask = a > 0.5
            if not np.any(mask):
                continue
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            label = f"#{i}"
            (tw, th), _ = cv2.getTextSize(label, font, fs, 2)
            ox, oy = cx - tw // 2, cy + th // 2
            for du, dv in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                cv2.putText(
                    out,
                    label,
                    (ox + du, oy + dv),
                    font,
                    fs,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
            color = (255, 255, 255) if i != current_idx else (0, 255, 255)
            cv2.putText(out, label, (ox, oy), font, fs, color, 2, cv2.LINE_AA)

    return out


def preview_sequence(images: list[np.ndarray], window: str = "frame") -> None:
    """Press any key to advance; **q** exits."""
    for im in images:
        cv2.imshow(window, im)
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
    cv2.destroyAllWindows()
