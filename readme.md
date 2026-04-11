# Video frame stitching (CS381)

Modular pipeline: **ffmpeg** frame extraction → **SIFT/ORB** → **BF/FLANN** + Lowe ratio → **RANSAC homography** → **warp** to a shared canvas → **blend**. Code lives under `stitch/` with section comments so the flow is easy to follow.

## Setup

```bash
pip install -r requirements.txt
# ffmpeg on PATH for --video and --save-video
```

## Layout

| Path | Role |
|------|------|
| `main.py` | CLI |
| `stitch/config.py` | `StitchConfig`, `IOConfig`, reference mode |
| `stitch/extract.py` | `subprocess` ffmpeg |
| `stitch/features.py` | SIFT / ORB |
| `stitch/matching.py` | BF / FLANN + ratio test (see file for metric table) |
| `stitch/geometry.py` | Homography, chain, **middle-frame re-center** |
| `stitch/warp.py` | Canvas, feather blend |
| `stitch/visualize.py` | Debug drawings |
| `stitch/pipeline.py` | One readable `run_stitch()` |

## CLI

```bash
python main.py --video sample.mp4 --fps 1 --detector sift --matcher bf --output-dir ./out/a
python main.py --frames-dir ./frames --detector orb --matcher flann --ref middle --save-video
```

- **`--ref middle`** (default): after chaining to frame 0, apply `inv(H_mid)` so the **center** frame is the reference — panoramas often look more balanced than `--ref first`.
- **`--blend-feather`**: distance-based weights in overlaps (default 12; use `0` to disable).
- **`--debug-matches`**: writes `debug/pair00_matches.png`.

Outputs: `panorama.png`, `sequence/frame_XXXX_global.png`, `homographies.npz`, `experiment.json`, `metrics.csv`, `run.log`.

## Compare runs

```bash
python compare_experiments.py ./out/a/experiment.json ./out/b/experiment.json
```
