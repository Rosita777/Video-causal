#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

METHOD="${METHOD:?Set METHOD to plain or dual}"
case "$METHOD" in
  plain)
    LORA_PATH="outputs/adapters/waterdrop_plain_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_plain_lora_100_eval20"
    ;;
  dual)
    LORA_PATH="outputs/adapters/waterdrop_dual_traj_bg1_lora_100/checkpoint-000100"
    OUTPUT_DIR="outputs/waterdrop_dual_traj_bg1_lora_100_eval20"
    ;;
  *)
    echo "METHOD must be plain or dual" >&2
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
  --lora-scale 1 \
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
