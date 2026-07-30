#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/models/.wan-runtime/bin/python}"
WAN_MODEL="${WAN_MODEL:-$PROJECT_ROOT/models/Wan2.1-T2V-1.3B-Diffusers}"
COG_MODEL="${COG_MODEL:-$PROJECT_ROOT/models/CogVideoX-2b}"
WAN_GPU="${WAN_GPU:-2}"
COG_GPU="${COG_GPU:-3}"
BASE_SEED="${BASE_SEED:-8100}"

FACTUAL_PROMPTS="$PROJECT_ROOT/prompts/waterdrop_capability_factual5.txt"
COUNTERFACTUAL_PROMPTS="$PROJECT_ROOT/prompts/waterdrop_capability_counterfactual5.txt"
OUTPUT_ROOT="$PROJECT_ROOT/outputs/waterdrop_backbone_probe_seed${BASE_SEED}"
LOG_ROOT="$PROJECT_ROOT/logs/waterdrop_backbone_probe_seed${BASE_SEED}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

run_wan() {
  CUDA_VISIBLE_DEVICES="$WAN_GPU" PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_wan_clean.py \
    --baseline clean \
    --prompts "$FACTUAL_PROMPTS" \
    --output-dir "$OUTPUT_ROOT/wan_factual" \
    --model "$WAN_MODEL" \
    --seed "$BASE_SEED" \
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

  CUDA_VISIBLE_DEVICES="$WAN_GPU" PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_wan_clean.py \
    --baseline clean \
    --prompts "$COUNTERFACTUAL_PROMPTS" \
    --output-dir "$OUTPUT_ROOT/wan_counterfactual" \
    --model "$WAN_MODEL" \
    --seed "$BASE_SEED" \
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

run_cog() {
  CUDA_VISIBLE_DEVICES="$COG_GPU" PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_cogvideox_clean.py \
    --prompts "$FACTUAL_PROMPTS" \
    --output-dir "$OUTPUT_ROOT/cog_factual" \
    --model "$COG_MODEL" \
    --seed "$BASE_SEED" \
    --steps 50 \
    --guidance-scale 6.0 \
    --num-frames 49 \
    --fps 8 \
    --height 480 \
    --width 720 \
    --dtype fp16 \
    --device cuda \
    --vae-slicing \
    --vae-tiling

  CUDA_VISIBLE_DEVICES="$COG_GPU" PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/generate_cogvideox_clean.py \
    --prompts "$COUNTERFACTUAL_PROMPTS" \
    --output-dir "$OUTPUT_ROOT/cog_counterfactual" \
    --model "$COG_MODEL" \
    --seed "$BASE_SEED" \
    --steps 50 \
    --guidance-scale 6.0 \
    --num-frames 49 \
    --fps 8 \
    --height 480 \
    --width 720 \
    --dtype fp16 \
    --device cuda \
    --vae-slicing \
    --vae-tiling
}

run_wan >"$LOG_ROOT/wan.log" 2>&1 &
wan_pid=$!
run_cog >"$LOG_ROOT/cog.log" 2>&1 &
cog_pid=$!

echo "wan_pid=$wan_pid gpu=$WAN_GPU"
echo "cog_pid=$cog_pid gpu=$COG_GPU"

status=0
wait "$wan_pid" || status=$?
wait "$cog_pid" || status=$?
exit "$status"
