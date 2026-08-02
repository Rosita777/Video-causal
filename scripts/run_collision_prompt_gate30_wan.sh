#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SHARD="${SHARD:?Set SHARD to 0 or 1}"
case "$SHARD" in
  0) SEEDS="9900,9902,9904,9906,9908,9910,9912,9914,9916,9918,9920,9922,9924,9926,9928" ;;
  1) SEEDS="9901,9903,9905,9907,9909,9911,9913,9915,9917,9919,9921,9923,9925,9927,9929" ;;
  *) echo "SHARD must be 0 or 1" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$((SHARD + 2))}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts "prompts/collision_prompt_gate30_shard_${SHARD}.txt" \
  --output-dir "outputs/collision_prompt_gate30_wan/shard_${SHARD}" \
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
