"""Filesystem helpers, logging, and simple timing utilities."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(log_file: Path | None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


@contextmanager
def timed(name: str, log: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Record wall time in seconds under key ``elapsed_s`` on the yielded dict."""
    stats: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        yield stats
    finally:
        stats["elapsed_s"] = time.perf_counter() - t0
        if log:
            log.info("%s finished in %.3fs", name, stats["elapsed_s"])
