#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="models/.wan-runtime/bin/python"
"$PYTHON_BIN" scripts/train_wan_waterdrop_lora.py \
  --manifest data/collision_train31.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/training_cache/collision_train31_dual_traj \
  --output-dir outputs/adapters/collision_train31_bg2_100 \
  --role erase \
  --objective dual_traj \
  --mask-weight 4.0 \
  --background-weight 2.0 \
  --redirect-weight 1.0 \
  --rank 16 \
  --alpha 16 \
  --learning-rate 1e-4 \
  --max-steps 100 \
  --save-every 25 \
  --seed 20260803
