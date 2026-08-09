#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs/protocol_v1_cogvideox_controls

run_one() {
  local gpu="$1"
  local baseline="$2"
  local mechanism="$3"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    models/.wan-runtime/bin/python scripts/run_protocol_v1_cogvideox_control_batch.py \
    --manifest data/protocol_v1/eval_manifest.csv \
    --mechanism "$mechanism" \
    --baseline "$baseline" \
    --model models/CogVideoX-2b \
    --output-dir "outputs/protocol_v1/eval_cogvideox_${baseline}/${mechanism}" \
    --skip-existing
}

(
  run_one 2 original water_impact
  run_one 2 original brittle_fracture
  run_one 2 negative_prompt water_impact
  run_one 2 negative_prompt brittle_fracture
) > logs/protocol_v1_cogvideox_controls/gpu2.log 2>&1 &
pid_gpu2=$!

(
  run_one 3 original rigid_collision
  run_one 3 original powder_impact
  run_one 3 negative_prompt rigid_collision
  run_one 3 negative_prompt powder_impact
) > logs/protocol_v1_cogvideox_controls/gpu3.log 2>&1 &
pid_gpu3=$!

echo "Started CogVideoX control queue on GPU 2 as PID $pid_gpu2"
echo "Started CogVideoX control queue on GPU 3 as PID $pid_gpu3"
wait "$pid_gpu2" "$pid_gpu3"
