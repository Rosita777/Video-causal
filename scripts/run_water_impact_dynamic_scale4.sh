#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
PYTHON="models/.wan-runtime/bin/python"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
PROMPTS="prompts/water_impact_dynamic_v1/scale4.prompts"
SEEDS="$(tr -d '\n' < data/water_impact_dynamic_v1/scale4_seeds.txt)"
LORA="outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v1/checkpoint-000200"

"$PYTHON" scripts/generate_wan_clean.py \
  --baseline clean --prompts "$PROMPTS" \
  --output-dir outputs/water_impact_dynamic_v1/scale4_base \
  --model "$MODEL" --seeds "$SEEDS" \
  --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
  --height 480 --width 832 --dtype bf16 --device cuda \
  --vae-slicing --vae-tiling --skip-existing \
  > logs/water_impact_dynamic_scale4_base.log 2>&1

for scale in 0.25 0.5 0.75; do
  tag="${scale/./p}"
  "$PYTHON" scripts/generate_wan_clean.py \
    --baseline clean --prompts "$PROMPTS" \
    --output-dir "outputs/water_impact_dynamic_v1/scale4_ckpt200_${tag}" \
    --model "$MODEL" --seeds "$SEEDS" \
    --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
    --height 480 --width 832 --dtype bf16 --device cuda \
    --vae-slicing --vae-tiling --skip-existing \
    --lora-path "$LORA" --lora-scale "$scale" \
    > "logs/water_impact_dynamic_scale4_${tag}.log" 2>&1
  "$PYTHON" scripts/build_paired_video_sheets.py \
    --prompts "$PROMPTS" \
    --base-dir outputs/water_impact_dynamic_v1/scale4_base/videos \
    --candidate-dir "outputs/water_impact_dynamic_v1/scale4_ckpt200_${tag}/videos" \
    --candidate-label "ckpt200_scale_${scale}" \
    --output-dir "outputs/water_impact_dynamic_v1/scale4_ckpt200_${tag}_sheets"
done
