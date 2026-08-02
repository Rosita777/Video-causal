#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LORA_SCALE="${LORA_SCALE:-0.75}"
LORA_PATH="${LORA_PATH:-outputs/adapters/waterdrop_generalization_v2_dual_traj_100/checkpoint-000100}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/waterdrop_generalization_v2_dual_traj_scale075_eval16}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
SEEDS="$(tr -d '\n' < prompts/waterdrop_generalization_eval16_seeds.txt)"

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts prompts/waterdrop_generalization_eval16.txt \
  --output-dir "$OUTPUT_DIR" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --lora-path "$LORA_PATH" \
  --lora-scale "$LORA_SCALE" \
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
