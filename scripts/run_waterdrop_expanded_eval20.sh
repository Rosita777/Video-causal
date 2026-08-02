#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

METHOD="${METHOD:?Set METHOD to plain, dual, or dual_scale075}"
case "$METHOD" in
  plain)
    LORA_PATH="outputs/adapters/waterdrop_plain_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_plain_lora_100_eval20"
    ;;
  dual)
    LORA_PATH="outputs/adapters/waterdrop_dual_traj_bg1_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_dual_traj_bg1_lora_100_eval20"
    LORA_SCALE=1
    ;;
  dual_scale075)
    LORA_PATH="outputs/adapters/waterdrop_dual_traj_bg1_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_dual_traj_bg1_lora_100_scale075_eval20"
    LORA_SCALE=0.75
    ;;
  *)
    echo "METHOD must be plain, dual, or dual_scale075" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
SEEDS="9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100,9100"

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts prompts/waterdrop_dual_traj_eval20.txt \
  --output-dir "$OUTPUT_DIR" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --lora-path "$LORA_PATH" \
  --lora-scale "${LORA_SCALE:-1}" \
  --seeds "$SEEDS" \
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
