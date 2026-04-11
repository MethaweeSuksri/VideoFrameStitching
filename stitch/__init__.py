"""
Video frame stitching: extract frames, match features, estimate homographies, warp & blend.

Public entry: ``stitch.pipeline.run_stitch`` or the ``main.py`` CLI.
"""

from stitch.pipeline import run_stitch

__all__ = ["run_stitch"]
