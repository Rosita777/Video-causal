#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
LORA_PATH="${LORA_PATH:-outputs/adapters/collision_general_preserve68_bg4_pw4_200/checkpoint-000200}"

CUDA_VISIBLE_DEVICES=2 "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean --prompts data/collision_validation7.prompts \
  --output-dir outputs/collision_validation7_general_preserve68 \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --seeds 11013,11019,11024,11025,11041,11045,11050 \
  --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
  --height 480 --width 832 --dtype bf16 --device cuda --vae-slicing --vae-tiling \
  --lora-path "$LORA_PATH" --lora-scale 1.0 \
  > logs/collision_validation7_general_preserve68.log 2>&1 &
validation_pid=$!

CUDA_VISIBLE_DEVICES=3 "$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean --prompts data/collision_specificity8.prompts \
  --output-dir outputs/collision_specificity8_general_preserve68 \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --seeds 12000,12001,12002,12003,12004,12005,12006,12007 \
  --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
  --height 480 --width 832 --dtype bf16 --device cuda --vae-slicing --vae-tiling \
  --lora-path "$LORA_PATH" --lora-scale 1.0 \
  > logs/collision_specificity8_general_preserve68.log 2>&1 &
specificity_pid=$!

wait "$validation_pid"
wait "$specificity_pid"

"$PYTHON" scripts/evaluate_collision_validation7.py \
  --base-dir outputs/collision_validation7_base \
  --adapter-dir outputs/collision_validation7_general_preserve68 \
  --output-csv experiments/pilot_week1/collision_validation7_general_preserve68_metrics.csv \
  --sheet-dir outputs/collision_validation7_contact_sheets_general_preserve68

"$PYTHON" scripts/evaluate_collision_validation7.py \
  --base-dir outputs/collision_specificity8_base \
  --adapter-dir outputs/collision_specificity8_general_preserve68 \
  --output-csv experiments/pilot_week1/collision_specificity8_general_preserve68_metrics.csv \
  --sheet-dir outputs/collision_specificity8_contact_sheets_general_preserve68
