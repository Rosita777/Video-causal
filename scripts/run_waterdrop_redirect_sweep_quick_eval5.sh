#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REDIRECT_WEIGHT="${REDIRECT_WEIGHT:?Set REDIRECT_WEIGHT to 0.025 or 0.10}"
case "$REDIRECT_WEIGHT" in
  0.025)
    LORA_PATH="outputs/adapters/waterdrop_dual_traj_rw0025_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_dual_traj_rw0025_lora_100_quick_eval5"
    ;;
  0.10)
    LORA_PATH="outputs/adapters/waterdrop_dual_traj_rw0100_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_dual_traj_rw0100_lora_100_quick_eval5"
    ;;
  *)
    echo "REDIRECT_WEIGHT must be 0.025 or 0.10" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts prompts/waterdrop_plain_lora_quick_eval5.txt \
  --output-dir "$OUTPUT_DIR" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --lora-path "$LORA_PATH" \
  --lora-scale 1 \
  --seeds 9600,9100,9100,9600,9100 \
  --steps 25 \
  --guidance-scale 5 \
  --num-frames 49 \
  --fps 8 \
  --height 480 \
  --width 832 \
  --dtype bf16 \
  --device cuda \
  --vae-slicing \
  --vae-tiling
