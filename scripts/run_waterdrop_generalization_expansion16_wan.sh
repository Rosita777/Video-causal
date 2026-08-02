#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SHARD="${SHARD:?Set SHARD to 0 or 1}"
case "$SHARD" in
  0) SEEDS="9800,9802,9804,9806,9808,9810,9812,9814" ;;
  1) SEEDS="9801,9803,9805,9807,9809,9811,9813,9815" ;;
  *) echo "SHARD must be 0 or 1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$((SHARD + 2))}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts "prompts/waterdrop_generalization_expansion16_shard_${SHARD}.txt" \
  --output-dir "outputs/waterdrop_generalization_expansion16_wan/shard_${SHARD}" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
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
