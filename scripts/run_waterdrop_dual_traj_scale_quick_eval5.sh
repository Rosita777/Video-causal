#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LORA_SCALE="${LORA_SCALE:?Set LORA_SCALE to 0.50 or 0.75}"
case "$LORA_SCALE" in
  0.50) OUTPUT_DIR="outputs/waterdrop_dual_traj_bg1_lora_100_scale050_quick_eval5" ;;
  0.75) OUTPUT_DIR="outputs/waterdrop_dual_traj_bg1_lora_100_scale075_quick_eval5" ;;
  *)
    echo "LORA_SCALE must be 0.50 or 0.75" >&2
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
  --lora-path outputs/adapters/waterdrop_dual_traj_bg1_lora_100/checkpoint-000100 \
  --lora-scale "$LORA_SCALE" \
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
