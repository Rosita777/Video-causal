#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MASK_WEIGHT="${MASK_WEIGHT:?Set MASK_WEIGHT to 0, 1, 2, or 4}"
case "$MASK_WEIGHT" in
  0|1|2|4) ;;
  *) echo "MASK_WEIGHT must be one of: 0, 1, 2, 4" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
TAG="mask_w${MASK_WEIGHT}_bg1_lora_100"

exec "$PYTHON" scripts/train_wan_waterdrop_lora.py \
  --manifest data/waterdrop_train_pilot40_sft_v0.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/training_cache/waterdrop_mask_bg_lora_v1 \
  --output-dir "outputs/adapters/waterdrop_${TAG}" \
  --role erase \
  --objective mask_bg \
  --mask-weight "$MASK_WEIGHT" \
  --background-weight 1.0 \
  --rank 16 \
  --alpha 16 \
  --learning-rate 1e-4 \
  --max-steps 100 \
  --save-every 25 \
  --seed 20260801
