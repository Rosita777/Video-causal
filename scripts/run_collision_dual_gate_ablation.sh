#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

"$PYTHON" scripts/evaluate_dual_gate_causal_cone.py \
  --features outputs/causal_subspace/collision_layer15_cone.pt \
  --output-dir experiments/pilot_week1/collision_dual_gate_ablation \
  --collision-manifest data/collision_train31.csv \
  --generic-motion-scores data/generic_preservation32_motion_scores.csv \
  --generic-train-count 16 --generic-eval-count 8 --other-ball-train-count 3 \
  --steps 500 --target-train-coverage 0.80 --device cuda \
  --fixed-gate-config experiments/pilot_week1/collision_dual_gate_ablation/summary.json \
  --export-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --export-gate-dilation 2
