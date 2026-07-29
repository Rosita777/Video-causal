#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
PROMPT_PATH="${PROMPT_PATH:-$PROJECT_ROOT/prompts/ball_blocks_clean_candidates50.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/ball_blocks_clean_candidates50_wan21_t2v_1.3b_seed7000_step25_f49_480x832}"

cd "$PROJECT_ROOT"

PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_wan_clean.py \
  --baseline clean \
  --prompts "$PROMPT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_PATH" \
  --seed 7000 \
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
