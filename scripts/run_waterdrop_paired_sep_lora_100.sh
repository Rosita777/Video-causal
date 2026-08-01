#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"

exec "$PYTHON" scripts/train_wan_waterdrop_lora.py \
  --manifest data/waterdrop_train_pilot40_sft_v0.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/training_cache/waterdrop_mask_bg_lora_v1 \
  --output-dir outputs/adapters/waterdrop_paired_sep_bg1_lora_100 \
  --role erase \
  --objective paired_sep \
  --background-weight 1.0 \
  --pair-weight 1.0 \
  --pair-margin 0.05 \
  --rank 16 \
  --alpha 16 \
  --learning-rate 1e-4 \
  --max-steps 100 \
  --save-every 25 \
  --seed 20260801
