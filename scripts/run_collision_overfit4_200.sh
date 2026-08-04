#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

exec "$PYTHON" scripts/train_wan_waterdrop_lora.py \
  --manifest data/collision_overfit4.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/training_cache/collision_overfit4 \
  --output-dir outputs/adapters/collision_overfit4_200 \
  --role erase --objective counterfactual_sft \
  --causal-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --activation-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --gate-floor 0 --max-steps 200 --save-every 50 --learning-rate 1e-4 \
  --rank 16 --alpha 16 --background-weight 4 \
  --pair-weight 0 --redirect-weight 0 --preserve-weight 0
