#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/waterdrop_train_pilot40_wan}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/waterdrop_train_pilot40_wan}"
FIXED_SEED="${FIXED_SEED:-9600}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

run_shard() {
  local shard="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" scripts/generate_wan_clean.py \
      --baseline clean \
      --prompts "prompts/waterdrop_train_pilot40_shard_${shard}.txt" \
      --output-dir "$OUTPUT_ROOT/shard_${shard}" \
      --model "$MODEL_PATH" \
      --seeds "$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED,$FIXED_SEED" \
      --steps 25 --guidance-scale 5.0 --num-frames 49 --fps 8 \
      --height 480 --width 832 --dtype bf16 --device cuda --vae-slicing --vae-tiling \
      >"$LOG_ROOT/shard_${shard}.log" 2>&1
}

run_shard 0 2 & pid0="$!"
run_shard 1 2 & pid1="$!"
run_shard 2 3 & pid2="$!"
run_shard 3 3 & pid3="$!"
status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do wait "$pid" || status=$?; done
exit "$status"
