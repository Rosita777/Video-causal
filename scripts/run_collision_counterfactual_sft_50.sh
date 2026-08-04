#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

exec "$PYTHON" scripts/train_wan_waterdrop_lora.py \
  --manifest data/collision_general_preserve68.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/training_cache/collision_general_preserve68_dual_traj \
  --output-dir outputs/adapters/collision_counterfactual_sft_50 \
  --role all --objective counterfactual_sft --balanced-roles \
  --causal-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --activation-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --gate-floor 0 --max-steps 50 --save-every 25 --learning-rate 1e-4 \
  --rank 16 --alpha 16 --background-weight 4 \
  --pair-weight 0 --redirect-weight 0 --preserve-weight 16
