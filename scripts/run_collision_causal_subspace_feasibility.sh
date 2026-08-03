#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
FEATURES="outputs/causal_subspace/collision_layer15_sigma050.pt"
RESULTS="experiments/pilot_week1/collision_causal_subspace_layer15"

"$PYTHON" scripts/extract_wan_causal_difference_features.py \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --collision-cache outputs/training_cache/collision_general_preserve68_dual_traj \
  --generic-cache outputs/training_cache/collision_general_preserve68_dual_traj \
  --waterdrop-cache outputs/training_cache/waterdrop_generalization_v2_dual_traj \
  --output "$FEATURES" --layer 15 --sigma 0.5 --tokens-per-sample 64

"$PYTHON" scripts/evaluate_causal_subspace_separation.py \
  --features "$FEATURES" --output-dir "$RESULTS" --rank 16 --negative-weight 1.0 \
  --collision-manifest data/collision_train31.csv \
  --heldout-receivers paper_cup,short_tin,wide_domino,wood_peg \
  --generic-train-count 24
