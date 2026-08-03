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
  --output-dir outputs/adapters/collision_general_preserve68_bg4_pw4_200 \
  --role all --objective dual_traj \
  --max-steps 200 --save-every 50 --learning-rate 1e-4 \
  --rank 16 --alpha 16 --mask-weight 4 --background-weight 4 \
  --pair-weight 1 --redirect-weight 1 --preserve-weight 4
