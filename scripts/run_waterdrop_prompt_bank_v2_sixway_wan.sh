#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
FIXED_SEED="${FIXED_SEED:-9000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/waterdrop_prompt_bank_v2_simple_wan}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/waterdrop_prompt_bank_v2_simple_wan}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

fixed_seed_list() {
  awk -v seed="$FIXED_SEED" '
    NF && $1 !~ /^#/ {count++}
    END {
      for (i = 1; i <= count; i++) {
        printf "%s%s", seed, (i == count ? "" : ",")
      }
    }
  ' "$1"
}

run_shard() {
  local shard="$1"
  local gpu="$2"
  local prompts="prompts/waterdrop_prompt_bank_v2_simple_shard_${shard}.txt"
  local seeds
  seeds="$(fixed_seed_list "$prompts")"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_wan_clean.py \
    --baseline clean \
    --prompts "$prompts" \
    --output-dir "$OUTPUT_ROOT/shard_${shard}" \
    --model "$MODEL_PATH" \
    --seeds "$seeds" \
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
    >"$LOG_ROOT/shard_${shard}.log" 2>&1
}

pids=()
for shard in 0 1 2; do
  run_shard "$shard" "$GPU_A" &
  pids+=("$!")
done
for shard in 3 4 5; do
  run_shard "$shard" "$GPU_B" &
  pids+=("$!")
done

echo "gpu=$GPU_A shards=0,1,2"
echo "gpu=$GPU_B shards=3,4,5"
echo "pids=${pids[*]} fixed_seed=$FIXED_SEED"

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
