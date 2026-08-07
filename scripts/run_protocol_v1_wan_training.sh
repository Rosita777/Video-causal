#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
PYTHONNOUSERSITE=1
MODEL=models/Wan2.1-T2V-1.3B-Diffusers
COMMON=(scripts/train_wan_waterdrop_lora.py --model "$MODEL" --height 480 --width 832 --num-frames 49 --max-steps 150 --learning-rate 1e-4 --rank 16 --alpha 16 --save-every 50 --seed 12000 --role all --objective dual_traj --balanced-roles --mask-weight 8.0 --background-weight 1.0 --pair-weight 1.0 --pair-margin 0.05 --redirect-weight 0.5 --preserve-weight 8.0)

run_one() {
  local mech="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 models/.wan-runtime/bin/python "${COMMON[@]}" \
    --manifest "data/protocol_v1/wan_train_${mech}.csv" \
    --cache-dir "outputs/protocol_v1/cache_${mech}" \
    --output-dir "outputs/protocol_v1/adapter_${mech}"
}

run_one water_impact 0 > logs/protocol_v1_train_water_impact.log 2>&1 & p1=$!
run_one rigid_collision 1 > logs/protocol_v1_train_rigid_collision.log 2>&1 & p2=$!
run_one brittle_fracture 3 > logs/protocol_v1_train_brittle_fracture.log 2>&1 & p3=$!
wait "$p1" "$p2" "$p3"
run_one powder_impact 0 > logs/protocol_v1_train_powder_impact.log 2>&1
