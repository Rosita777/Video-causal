#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

input_video=$(find \
  outputs/waterdrop_scene_probe30_wan_seed8300_8329/part_a/videos \
  -maxdepth 1 \
  -name '*seed8310.mp4' \
  -print \
  -quit)

if [[ -z "$input_video" ]]; then
  echo "wd010 factual video with seed 8310 was not found" >&2
  exit 1
fi

models/.wan-runtime/bin/python scripts/build_static_counterfactual_pair.py \
  --input "$input_video" \
  --output-dir outputs/waterdrop_wd010_aligned_pair \
  --reference-start 0 \
  --reference-end 16
