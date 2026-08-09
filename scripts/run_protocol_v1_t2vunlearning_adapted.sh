#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_mechanism() {
  local gpu="$1"
  local mechanism="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false \
    models/.wan-runtime/bin/python \
    scripts/run_t2vunlearning_adapted_cogvideox_batch.py \
    --manifest data/protocol_v1/eval_manifest.csv \
    --mechanism "$mechanism" \
    --model models/CogVideoX-2b \
    --checkpoint "outputs/protocol_v1/t2vunlearning_adapted_${mechanism}/checkpoint-000500" \
    --output-dir "outputs/protocol_v1/eval_t2vunlearning_adapted/${mechanism}" \
    --skip-existing
}

mkdir -p logs/protocol_v1_t2vunlearning_adapted

(
  run_mechanism 2 water_impact
  run_mechanism 2 brittle_fracture
) > logs/protocol_v1_t2vunlearning_adapted/gpu2.log 2>&1 &
pid_gpu2=$!

(
  run_mechanism 3 rigid_collision
  run_mechanism 3 powder_impact
) > logs/protocol_v1_t2vunlearning_adapted/gpu3.log 2>&1 &
pid_gpu3=$!

echo "Started GPU 2 queue as PID $pid_gpu2"
echo "Started GPU 3 queue as PID $pid_gpu3"
wait "$pid_gpu2" "$pid_gpu3"
