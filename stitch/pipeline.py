"""
End-to-end stitch: features → matches → homographies → canvas → blend → artifacts.

This module is intentionally linear top-to-bottom in ``run_stitch`` so you can read it
like a script, with helpers kept small in other ``stitch.*`` files.
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stitch.config import IOConfig, ReferenceFrame, StitchConfig
from stitch import geometry as geom
from stitch import matching
from stitch import visualize as vis
from stitch import warp
from stitch.features import make_backend
from stitch.util import ensure_dir

logger = logging.getLogger(__name__)


def _load_bgr(paths: list[Path]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        out.append(im)
    return out


def run_stitch(
    frame_paths: list[Path],
    stitch_cfg: StitchConfig,
    io_cfg: IOConfig,
    *,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """
    Run the full pipeline. **frame_paths** must be in time order (sorted names recommended).

    Returns a JSON-serializable summary dict (also written to ``experiment.json``).
    """
    out_root = ensure_dir(io_cfg.output_dir)
    seq_dir = ensure_dir(out_root / "sequence")
    overview_dir = ensure_dir(out_root / "sequence_overview")
    dbg_dir = ensure_dir(out_root / "debug")

    paths = list(frame_paths)
    if max_frames is not None:
        paths = paths[:max_frames]

    if len(paths) < 2:
        raise ValueError("Need at least two frames")

    # --- Load pixels ---
    images_bgr = _load_bgr(paths)
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
    n = len(grays)
    sizes = [(im.shape[0], im.shape[1]) for im in images_bgr]

    # --- Features (all frames up front — simple and fast enough for class-scale clips) ---
    backend = make_backend(stitch_cfg.detector)
    matcher = matching.build_matcher(stitch_cfg.matcher, backend.spec)

    keypoints: list[list[cv2.KeyPoint]] = []
    descriptors: list[np.ndarray | None] = []
    t_detect = 0.0
    import time as _time

    for gi in range(n):
        t0 = _time.perf_counter()
        kp, des = backend.detect_and_compute(grays[gi])
        t_detect += _time.perf_counter() - t0
        keypoints.append(kp)
        descriptors.append(des)

    # --- Adjacent pairs: match (later → earlier), estimate H: later → earlier ---
    pairwise: list[np.ndarray] = []
    pair_rows: list[dict[str, Any]] = []
    t_match = t_h = 0.0

    for i in range(n - 1):
        # Frame i+1 is "src", frame i is "dst" so we warp newer frames into older coords first,
        # then chain into frame 0.
        t0 = _time.perf_counter()
        good_matches, _raw = matching.match_with_ratio(
            descriptors[i + 1],
            descriptors[i],
            matcher,
            ratio=stitch_cfg.ratio_thresh,
        )
        t_match += _time.perf_counter() - t0

        t0 = _time.perf_counter()
        est = geom.estimate_pair_homography(
            keypoints[i + 1],
            keypoints[i],
            good_matches,
            ransac_thresh=stitch_cfg.ransac_reproj_threshold,
            max_singular_ratio=stitch_cfg.homography_max_singular_ratio,
            max_scale=stitch_cfg.homography_max_scale,
            min_scale=stitch_cfg.homography_min_scale,
        )
        t_h += _time.perf_counter() - t0

        if est.H is None:
            logger.warning(
                "Pair %d→%d: homography unusable (%s, %d inliers / %d matches); using identity",
                i + 1,
                i,
                est.status,
                est.inliers,
                est.matches_used,
            )
            H_use = np.eye(3, dtype=np.float64)
            mask = est.mask
        else:
            H_use = est.H
            mask = est.mask

        pairwise.append(H_use)
        pair_rows.append(
            {
                "pair": i,
                "matches": len(good_matches),
                "inliers": est.inliers,
                "status": est.status,
            }
        )

        if io_cfg.save_debug_matches and i == 0:
            panel = vis.draw_matches_inlier_outlier(
                images_bgr[i + 1],
                keypoints[i + 1],
                images_bgr[i],
                keypoints[i],
                good_matches,
                mask,
            )
            vis.imwrite(dbg_dir / "pair00_matches.png", panel)
            vis.imwrite(dbg_dir / "frame00_keypoints.png", vis.draw_keypoints(images_bgr[0], keypoints[0]))

    # --- Compose: each frame → frame 0, optionally re-center to middle frame ---
    G_into_0 = geom.chain_to_first_frame(pairwise)
    if stitch_cfg.reference_frame == ReferenceFrame.MIDDLE:
        G_ref = geom.recenter_to_middle(G_into_0)
    else:
        G_ref = G_into_0

    geom.save_homographies_npz(out_root / "homographies.npz", G_ref, pairwise)

    canvas = warp.build_canvas(sizes, G_ref, io_cfg.canvas_margin_px)

    # --- Warp + save sequence + blend ---
    t_warp = 0.0
    warped_list: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for fi in range(n):
        t0 = _time.perf_counter()
        wim, alpha = warp.warp_to_canvas(images_bgr[fi], G_ref[fi], canvas)
        t_warp += _time.perf_counter() - t0
        warped_list.append(wim)
        alphas.append(alpha)
        vis.imwrite(seq_dir / f"frame_{fi:04d}_global.png", wim)

    for fi in range(n):
        panel = vis.compose_alignment_overview(
            warped_list,
            alphas,
            fi,
            non_current_weight=io_cfg.overview_non_current_weight,
        )
        vis.imwrite(overview_dir / f"frame_{fi:04d}.png", panel)

    t0 = _time.perf_counter()
    panorama = warp.blend_weighted_average(warped_list, alphas, stitch_cfg.blend_feather_radius)
    t_blend = _time.perf_counter() - t0
    cv2.imwrite(str(out_root / "panorama.png"), panorama)

    # --- MP4: overview (all frames + highlight current); even dims for libx264+yuv420p ---
    if io_cfg.save_video:
        vid = out_root / "reconstructed_sequence.mp4"
        cmd = [
            io_cfg.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(float(io_cfg.overview_video_fps)),
            "-i",
            str(overview_dir / "frame_%04d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(vid),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("ffmpeg encode failed: %s", proc.stderr)
        else:
            logger.info("Wrote %s", vid)

    if io_cfg.interactive_preview:
        vis.preview_sequence(warped_list, window="global alignment")

    avg_m = float(np.mean([r["matches"] for r in pair_rows])) if pair_rows else 0.0
    avg_i = float(np.mean([r["inliers"] for r in pair_rows])) if pair_rows else 0.0

    summary: dict[str, Any] = {
        "num_frames": n,
        "canvas": {"width": canvas.width, "height": canvas.height},
        "reference": stitch_cfg.reference_frame.value,
        "stitch": {
            "detector": stitch_cfg.detector,
            "matcher": stitch_cfg.matcher,
            "ratio_thresh": stitch_cfg.ratio_thresh,
            "ransac_reproj_threshold": stitch_cfg.ransac_reproj_threshold,
            "homography_max_singular_ratio": stitch_cfg.homography_max_singular_ratio,
            "blend_feather_radius": stitch_cfg.blend_feather_radius,
            "reference_frame": stitch_cfg.reference_frame.value,
        },
        "timing_s": {
            "detect": t_detect,
            "match": t_match,
            "homography": t_h,
            "warp": t_warp,
            "blend": t_blend,
            "total": t_detect + t_match + t_h + t_warp + t_blend,
        },
        "pairs": pair_rows,
        "avg_matches": avg_m,
        "avg_inliers": avg_i,
    }

    (out_root / "experiment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "matches", "inliers", "status"])
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    return summary
