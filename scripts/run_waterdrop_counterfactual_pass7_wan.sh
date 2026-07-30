#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

output_dir="outputs/waterdrop_counterfactual_pass7_wan_matched_seeds"
log_dir="logs/waterdrop_counterfactual_pass7_wan_matched_seeds"
mkdir -p "$log_dir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}" \
  models/.wan-runtime/bin/python scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts prompts/waterdrop_counterfactual_pass7.txt \
  --output-dir "$output_dir" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --seeds 8300,8301,8302,8307,8308,8310,8316 \
  --steps 25 \
  --guidance-scale 5.0 \
  --num-frames 49 \
  --fps 8 \
  --height 480 \
  --width 832 \
  --dtype bf16 \
  --device cuda \
  --vae-slicing \
  --vae-tiling \
  2>&1 | tee "$log_dir/wan.log"
