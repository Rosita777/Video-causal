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
  --output-dir outputs/adapters/collision_target_conditioned_redirect4_50 \
  --role erase --objective target_conditioned_redirect \
  --target-phrase "red rubber ball" \
  --causal-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --activation-gate-dir outputs/causal_gates/collision_dual_gate_d2 \
  --gate-floor 0 --max-steps 50 --save-every 25 --learning-rate 1e-4 \
  --rank 16 --alpha 16 --background-weight 4 \
  --pair-weight 0 --redirect-weight 4 --preserve-weight 0
