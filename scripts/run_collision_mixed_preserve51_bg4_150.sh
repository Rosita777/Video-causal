#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="models/.wan-runtime/bin/python"
"$PYTHON_BIN" scripts/train_wan_waterdrop_lora.py \
  --manifest data/collision_mixed_preserve51.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/training_cache/collision_mixed_preserve51 \
  --output-dir outputs/adapters/collision_mixed_preserve51_bg4_150 \
  --role all \
  --objective dual_traj \
  --mask-weight 4.0 \
  --background-weight 4.0 \
  --redirect-weight 1.0 \
  --rank 16 \
  --alpha 16 \
  --learning-rate 1e-4 \
  --max-steps 150 \
  --save-every 50 \
  --seed 20260803
