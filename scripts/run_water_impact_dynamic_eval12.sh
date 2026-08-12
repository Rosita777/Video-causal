#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
PYTHON="models/.wan-runtime/bin/python"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
PROMPTS="prompts/water_impact_dynamic_v1/eval12.prompts"
SEEDS="$($PYTHON - <<'PY'
import csv
rows=list(csv.DictReader(open('data/water_impact_dynamic_v1/eval12.csv')))
print(','.join(row['seed'] for row in rows))
PY
)"

generate() {
  local name="$1"
  shift
  "$PYTHON" scripts/generate_wan_clean.py \
    --baseline clean \
    --prompts "$PROMPTS" \
    --output-dir "outputs/water_impact_dynamic_v1/eval12_${name}" \
    --model "$MODEL" \
    --seeds "$SEEDS" \
    --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
    --height 480 --width 832 --dtype bf16 --device cuda \
    --vae-slicing --vae-tiling --skip-existing "$@" \
    > "logs/water_impact_dynamic_eval12_${name}.log" 2>&1
}

generate base
generate ckpt200 \
  --lora-path outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v1/checkpoint-000200 \
  --lora-scale 1.0
generate ckpt200_scale0p5 \
  --lora-path outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v1/checkpoint-000200 \
  --lora-scale 0.5
generate ckpt300 \
  --lora-path outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v1/checkpoint-000300 \
  --lora-scale 1.0

for checkpoint in ckpt200 ckpt200_scale0p5 ckpt300; do
  "$PYTHON" scripts/build_paired_video_sheets.py \
    --prompts "$PROMPTS" \
    --base-dir outputs/water_impact_dynamic_v1/eval12_base/videos \
    --candidate-dir "outputs/water_impact_dynamic_v1/eval12_${checkpoint}/videos" \
    --candidate-label "$checkpoint" \
    --output-dir "outputs/water_impact_dynamic_v1/eval12_${checkpoint}_contact_sheets"
done
