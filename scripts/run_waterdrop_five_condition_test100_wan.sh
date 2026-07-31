#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
FIXED_SEED="${FIXED_SEED:-9100}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/waterdrop_five_condition_test100_wan}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/waterdrop_five_condition_test100_wan}"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

fixed_seed_list() {
  awk -v seed="$FIXED_SEED" '
    NF && $1 !~ /^#/ {count++}
    END { for (i = 1; i <= count; i++) printf "%s%s", seed, (i == count ? "" : ",") }
  ' "$1"
}

run_shard() {
  local shard="$1" gpu="$2"
  local prompts="prompts/waterdrop_five_condition_test100_shard_${shard}.txt"
  local seeds
  seeds="$(fixed_seed_list "$prompts")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" scripts/generate_wan_clean.py \
      --baseline clean --prompts "$prompts" --output-dir "$OUTPUT_ROOT/shard_${shard}" \
      --model "$MODEL_PATH" --seeds "$seeds" --steps 25 --guidance-scale 5.0 \
      --num-frames 49 --fps 8 --height 480 --width 832 --dtype bf16 --device cuda \
      --vae-slicing --vae-tiling >"$LOG_ROOT/shard_${shard}.log" 2>&1
}

run_shard 0 2 & pid0="$!"
run_shard 1 3 & pid1="$!"
run_shard 2 3 & pid2="$!"
run_shard 3 3 & pid3="$!"
echo "gpu=2 shard=0; gpu=3 shards=1,2,3"
echo "pids=$pid0 $pid1 $pid2 $pid3 fixed_seed=$FIXED_SEED"
status=0
for pid in "$pid0" "$pid1" "$pid2" "$pid3"; do wait "$pid" || status=$?; done
exit "$status"
