#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SHARD="${SHARD:?Set SHARD to 0 or 1}"
case "$SHARD" in 0|1) ;; *) echo "SHARD must be 0 or 1" >&2; exit 2 ;; esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$((SHARD + 2))}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

if [[ "$SHARD" == 0 ]]; then
  PROMPTS=data/red_ball_negation8_shard0.prompts
  SEEDS=15100,15101,15102,15103
else
  PROMPTS=data/red_ball_negation8_shard1.prompts
  SEEDS=15104,15105,15106,15107
fi

exec "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean --prompts "$PROMPTS" \
  --output-dir "outputs/red_ball_negation8/shard_${SHARD}" \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --seeds "$SEEDS" --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
  --height 480 --width 832 --dtype bf16 --device cuda --vae-slicing --vae-tiling \
  --skip-existing
