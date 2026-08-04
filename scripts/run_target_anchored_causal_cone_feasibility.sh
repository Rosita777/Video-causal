#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
FEATURES="outputs/causal_subspace/collision_layer15_cone.pt"
RESULTS="experiments/pilot_week1/collision_target_anchored_cone"

"$PYTHON" scripts/extract_wan_causal_difference_features.py \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --collision-cache outputs/training_cache/collision_general_preserve68_dual_traj \
  --generic-cache outputs/training_cache/collision_general_preserve68_dual_traj \
  --waterdrop-cache outputs/training_cache/waterdrop_generalization_v2_dual_traj \
  --other-ball-cache outputs/training_cache/non_target_ball_collision_controls \
  --negation-cache outputs/training_cache/red_ball_negation_controls \
  --output "$FEATURES" --layer 15 --sigma 0.5 \
  --tokens-per-frame 16 --background-per-frame 4

"$PYTHON" scripts/evaluate_target_anchored_causal_cone.py \
  --features "$FEATURES" --output-dir "$RESULTS" \
  --generic-motion-scores data/generic_preservation32_motion_scores.csv \
  --generic-train-count 16 --generic-eval-count 8 \
  --steps 500 --radius 6 --decay 0.95
