"""
Run configuration — single place for experiment knobs (CLI maps into these dataclasses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class ReferenceFrame(str, Enum):
    """
    Which frame defines the “world” coordinate axes.

    - FIRST: frame 0 is fixed; later frames accumulate motion. Simple, but drift
      from a long chain can bend the mosaic.
    - MIDDLE: same chain, then we apply inv(H_mid) so the center frame is upright
      and centered — panoramas often *look* more natural for symmetric pans.
    """

    FIRST = "first"
    MIDDLE = "middle"


@dataclass
class StitchConfig:
    """Algorithm parameters (detector, matcher, RANSAC, blending)."""

    detector: Literal["sift", "orb"] = "sift"
    matcher: Literal["bf", "flann"] = "bf"
    ratio_thresh: float = 0.75
    ransac_reproj_threshold: float = 4.0
    # Reject homographies that rescale or shear too extremely (reduces crazy warps).
    homography_max_singular_ratio: float = 6.0
    homography_max_scale: float = 8.0
    homography_min_scale: float = 0.05
    blend_feather_radius: int = 12
    reference_frame: ReferenceFrame = ReferenceFrame.MIDDLE
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IOConfig:
    """Paths and ffmpeg-related I/O."""

    output_dir: Path = Path("output/stitch_run")
    ffmpeg_bin: str = "ffmpeg"
    save_debug_matches: bool = False
    save_video: bool = True
    interactive_preview: bool = False
    canvas_margin_px: int = 48
    overview_video_fps: float = 6.0
    overview_non_current_weight: float = 0.01
