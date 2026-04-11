#!/usr/bin/env python3
"""Compare several ``experiment.json`` files in a simple text table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_paths", nargs="+", type=Path)
    ns = ap.parse_args()

    rows: list[dict] = []
    for path in ns.json_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        st = data.get("stitch", {})
        tm = data.get("timing_s", {})
        rows.append(
            {
                "run": path.parent.name,
                "detector": st.get("detector"),
                "matcher": st.get("matcher"),
                "ref": data.get("reference"),
                "frames": data.get("num_frames"),
                "avg_m": round(float(data.get("avg_matches", 0)), 1),
                "avg_i": round(float(data.get("avg_inliers", 0)), 1),
                "total_s": round(float(tm.get("total", 0)), 4),
            }
        )

    cols = ["run", "detector", "matcher", "ref", "frames", "avg_m", "avg_i", "total_s"]
    w = [max(len(c), max(len(str(r[c])) for r in rows)) for c in cols]
    print(" | ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    print("-+-".join("-" * w[i] for i in range(len(cols))))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(w[i]) for i, c in enumerate(cols)))


if __name__ == "__main__":
    main()
