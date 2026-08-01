#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
FIXED_SEED="${FIXED_SEED:-9400}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/waterdrop_control_replacements12_wan}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/waterdrop_control_replacements12_wan}"
GPU_SHARD0="${GPU_SHARD0:?GPU_SHARD0 is required}"
GPU_SHARD1="${GPU_SHARD1:?GPU_SHARD1 is required}"
GPU_SHARD2="${GPU_SHARD2:?GPU_SHARD2 is required}"
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
  local prompts="prompts/waterdrop_control_replacements12_shard_${shard}.txt"
  local seeds
  seeds="$(fixed_seed_list "$prompts")"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" scripts/generate_wan_clean.py \
      --baseline clean --prompts "$prompts" --output-dir "$OUTPUT_ROOT/shard_${shard}" \
      --model "$MODEL_PATH" --seeds "$seeds" --steps 25 --guidance-scale 5.0 \
      --num-frames 49 --fps 8 --height 480 --width 832 --dtype bf16 --device cuda \
      --vae-slicing --vae-tiling >"$LOG_ROOT/shard_${shard}.log" 2>&1
}

run_shard 0 "$GPU_SHARD0" & pid0="$!"
run_shard 1 "$GPU_SHARD1" & pid1="$!"
run_shard 2 "$GPU_SHARD2" & pid2="$!"
echo "shard0=gpu$GPU_SHARD0 shard1=gpu$GPU_SHARD1 shard2=gpu$GPU_SHARD2"
status=0
for pid in "$pid0" "$pid1" "$pid2"; do wait "$pid" || status=$?; done
exit "$status"
