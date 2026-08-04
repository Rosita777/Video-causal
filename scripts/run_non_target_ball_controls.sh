#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SHARD="${SHARD:?Set SHARD to 0 or 1}"
case "$SHARD" in 0|1) ;; *) echo "SHARD must be 0 or 1" >&2; exit 2 ;; esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$((SHARD + 2))}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

if [[ "$SHARD" == 0 ]]; then
  PROMPTS=data/non_target_ball_collision10_shard0.prompts
  SEEDS=15000,15001,15002,15003,15004
else
  PROMPTS=data/non_target_ball_collision10_shard1.prompts
  SEEDS=15005,15006,15007,15008,15009
fi

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean --prompts "$PROMPTS" \
  --output-dir "outputs/non_target_ball_collision10/shard_${SHARD}" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --seeds "$SEEDS" --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
  --height 480 --width 832 --dtype bf16 --device cuda --vae-slicing --vae-tiling \
  --skip-existing
