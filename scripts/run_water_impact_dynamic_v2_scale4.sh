#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
PYTHON="models/.wan-runtime/bin/python"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
PROMPTS="prompts/water_impact_dynamic_v1/scale4.prompts"
SEEDS="$(tr -d '\n' < data/water_impact_dynamic_v1/scale4_seeds.txt)"

for step in 25 50 100 200; do
  checkpoint="$(printf '%06d' "$step")"
  name="v2_ckpt${step}"
  "$PYTHON" scripts/generate_wan_clean.py \
    --baseline clean --prompts "$PROMPTS" \
    --output-dir "outputs/water_impact_dynamic_v1/scale4_${name}" \
    --model "$MODEL" --seeds "$SEEDS" \
    --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
    --height 480 --width 832 --dtype bf16 --device cuda \
    --vae-slicing --vae-tiling --skip-existing \
    --lora-path "outputs/water_impact_dynamic_v1/adapter_dynamic_sft_preserve_v2/checkpoint-${checkpoint}" \
    --lora-scale 1.0 \
    > "logs/water_impact_dynamic_scale4_${name}.log" 2>&1

  "$PYTHON" scripts/build_paired_video_sheets.py \
    --prompts "$PROMPTS" \
    --base-dir outputs/water_impact_dynamic_v1/scale4_base/videos \
    --candidate-dir "outputs/water_impact_dynamic_v1/scale4_${name}/videos" \
    --candidate-label "$name" \
    --output-dir "outputs/water_impact_dynamic_v1/scale4_${name}_sheets"
done
