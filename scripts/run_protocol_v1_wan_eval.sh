#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
mkdir -p logs/protocol_v1_eval_ours

run_one() {
  local mech="$1" gpu="$2" start="$3" seed="$4"
  local seeds
  seeds=$(printf "${seed},%.0s" {1..20})
  seeds=${seeds%,}
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 models/.wan-runtime/bin/python scripts/generate_wan_clean.py \
    --baseline clean \
    --prompts data/protocol_v1/eval.prompts \
    --output-dir "outputs/protocol_v1/eval_ours/${mech}" \
    --model models/Wan2.1-T2V-1.3B-Diffusers \
    --lora-path "outputs/protocol_v1/adapter_${mech}/checkpoint-000150" \
    --lora-scale 0.75 \
    --seeds "$seeds" \
    --start-index "$start" \
    --limit 20 \
    --steps 25 \
    --guidance-scale 5.0 \
    --num-frames 49 \
    --fps 8 \
    --height 480 \
    --width 832 \
    --dtype bf16 \
    --device cuda \
    --vae-slicing \
    --vae-tiling \
    --skip-existing
}

run_one water_impact 0 0 12000 > logs/protocol_v1_eval_ours/water_impact.log 2>&1 & p1=$!
run_one rigid_collision 1 20 12001 > logs/protocol_v1_eval_ours/rigid_collision.log 2>&1 & p2=$!
run_one brittle_fracture 2 40 12002 > logs/protocol_v1_eval_ours/brittle_fracture.log 2>&1 & p3=$!
run_one powder_impact 3 60 12003 > logs/protocol_v1_eval_ours/powder_impact.log 2>&1 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
