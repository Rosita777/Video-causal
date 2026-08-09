#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal
mkdir -p logs/protocol_v1_videoeraser_official

run_one() {
  local mech="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    models/.wan-runtime/bin/python scripts/run_videoeraser_official_cogvideox_batch.py \
    --manifest data/protocol_v1/eval_manifest.csv \
    --mechanism "$mech" \
    --model models/CogVideoX-2b \
    --official-root baselines/external/VideoEraser \
    --output-dir "outputs/protocol_v1/videoeraser_official/${mech}" \
    --steps 50 \
    --guidance-scale 6.0 \
    --num-frames 49 \
    --fps 8 \
    --skip-existing
}

run_one water_impact 0 > logs/protocol_v1_videoeraser_official/water_impact.log 2>&1 & p1=$!
run_one rigid_collision 1 > logs/protocol_v1_videoeraser_official/rigid_collision.log 2>&1 & p2=$!
run_one brittle_fracture 2 > logs/protocol_v1_videoeraser_official/brittle_fracture.log 2>&1 & p3=$!
run_one powder_impact 3 > logs/protocol_v1_videoeraser_official/powder_impact.log 2>&1 & p4=$!
wait "$p1" "$p2" "$p3" "$p4"
