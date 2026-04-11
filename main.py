#!/usr/bin/env python3
"""
CLI: extract frames once, then run every detector × matcher × reference combo
into its own subdirectory under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal, cast

from stitch.config import IOConfig, ReferenceFrame, StitchConfig
from stitch.extract import extract_frames, ffmpeg_on_path
from stitch.pipeline import run_stitch
from stitch.util import ensure_dir, setup_logging


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all stitch variants (SIFT/ORB × BF/FLANN × first/middle ref)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Input video (ffmpeg in PATH)")
    src.add_argument("--frames-dir", type=Path, help="Folder of ordered images")

    p.add_argument("--output-dir", type=Path, default=Path("output/video_test/run_new"))
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="With --video: extraction rate (default 2). Ignored if --frame-interval is set.",
    )
    p.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        help="With --video: keep every Nth input frame instead of --fps",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--ffmpeg-bin", default="ffmpeg")
    return p.parse_args(argv)


def _subdir_name(det: str, matcher: str, ref: ReferenceFrame) -> str:
    return f"{det}_{matcher}_ref-{ref.value}"


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    out = ensure_dir(args.output_dir)
    setup_logging(out / "run.log")

    log = logging.getLogger(__name__)

    if args.video:
        if not ffmpeg_on_path(args.ffmpeg_bin):
            log.error("ffmpeg not found; use --frames-dir or install ffmpeg")
            return 1
        if args.frame_interval is not None:
            ext_dir = out / "extracted_frames"
            paths = extract_frames(
                args.video,
                ext_dir,
                frame_interval=args.frame_interval,
                ffmpeg_bin=args.ffmpeg_bin,
            )
        else:
            ext_dir = out / "extracted_frames"
            paths = extract_frames(
                args.video,
                ext_dir,
                fps=args.fps,
                ffmpeg_bin=args.ffmpeg_bin,
            )
    else:
        paths = sorted(
            p
            for p in args.frames_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not paths:
            log.error("No images in %s", args.frames_dir)
            return 1

    combos: list[tuple[str, str, ReferenceFrame]] = [
        (det, mat, ref)
        for det in ("sift", "orb")
        for mat in ("bf", "flann")
        for ref in (ReferenceFrame.FIRST, ReferenceFrame.MIDDLE)
    ]

    for det, mat, ref in combos:
        name = _subdir_name(det, mat, ref)
        sub = ensure_dir(out / name)
        log.info("=== %s ===", name)
        stitch_cfg = StitchConfig(
            detector=cast(Literal["sift", "orb"], det),
            matcher=cast(Literal["bf", "flann"], mat),
            reference_frame=ref,
        )
        io_cfg = IOConfig(
            output_dir=sub,
            ffmpeg_bin=args.ffmpeg_bin,
            save_debug_matches=True,
            save_video=True,
            interactive_preview=False,
        )
        run_stitch(paths, stitch_cfg, io_cfg, max_frames=args.max_frames)

    log.info("Done. Outputs under %s (%d runs)", out, len(combos))
    return 0


if __name__ == "__main__":
    sys.exit(main())
