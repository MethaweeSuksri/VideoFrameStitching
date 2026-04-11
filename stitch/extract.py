"""
Frame extraction with ffmpeg only through ``subprocess`` (no ffmpeg Python bindings).

We output sequentially named PNGs so sorting by filename equals time order.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ffmpeg_on_path(ffmpeg_bin: str = "ffmpeg") -> bool:
    return shutil.which(ffmpeg_bin) is not None


def extract_frames(
    video_path: Path | str,
    out_dir: Path | str,
    *,
    fps: float | None = None,
    frame_interval: int | None = None,
    filename_pattern: str = "frame_%06d.png",
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
) -> list[Path]:
    """
    Extract frames.

    Provide **exactly one** of:

    - ``fps``: resample to this constant output frame rate.
    - ``frame_interval``: keep every N-th input frame (1 = all frames).

    ``filename_pattern`` must contain one ``%06d``-style printf placeholder for ffmpeg.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))

    if (fps is None) == (frame_interval is None):
        raise ValueError("Set exactly one of: fps, frame_interval")

    out_template = str(out_dir / filename_pattern)

    cmd: list[str] = [ffmpeg_bin, "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    cmd.extend(["-i", str(video_path)])

    if fps is not None:
        cmd.extend(["-vf", f"fps={float(fps)}"])
    else:
        assert frame_interval is not None
        # n = frame index in demuxer; select every Nth frame
        # vfr keeps only selected frames (works on older ffmpeg; newer also accepts -fps_mode vfr)
        cmd.extend(["-vf", f"select='not(mod(n\\,{int(frame_interval)}))'", "-vsync", "vfr"])

    cmd.extend(["-q:v", "2", out_template])

    logger.info("ffmpeg extract: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout or f"ffmpeg exit {proc.returncode}")

    # Collect outputs: same stem prefix as pattern (e.g. frame_)
    prefix = filename_pattern.split("%")[0]
    paths = sorted(
        p
        for p in out_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"} and p.name.startswith(prefix)
    )
    logger.info("Extracted %d frame file(s) → %s", len(paths), out_dir)
    return paths
