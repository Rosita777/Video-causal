#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
FIXED_SEEDS="9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000,9000"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/waterdrop_prompt_gate40_v1_wan}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/waterdrop_prompt_gate40_v1_wan}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

run_part() {
  local gpu="$1"
  local prompts="$2"
  local output_dir="$3"

  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_wan_clean.py \
    --baseline clean \
    --prompts "$prompts" \
    --output-dir "$output_dir" \
    --model "$MODEL_PATH" \
    --seeds "$FIXED_SEEDS" \
    --steps 25 \
    --guidance-scale 5.0 \
    --num-frames 49 \
    --fps 8 \
    --height 480 \
    --width 832 \
    --dtype bf16 \
    --device cuda \
    --vae-slicing \
    --vae-tiling
}

run_part "$GPU_A" prompts/waterdrop_prompt_gate40_v1_part_a.txt "$OUTPUT_ROOT/part_a" \
  >"$LOG_ROOT/part_a.log" 2>&1 &
pid_a=$!

run_part "$GPU_B" prompts/waterdrop_prompt_gate40_v1_part_b.txt "$OUTPUT_ROOT/part_b" \
  >"$LOG_ROOT/part_b.log" 2>&1 &
pid_b=$!

echo "part_a_pid=$pid_a gpu=$GPU_A prompts=20 fixed_seed=9000"
echo "part_b_pid=$pid_b gpu=$GPU_B prompts=20 fixed_seed=9000"

status=0
wait "$pid_a" || status=$?
wait "$pid_b" || status=$?
exit "$status"
