#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal

exec models/.wan-runtime/bin/python scripts/train_wan_waterdrop_lora.py \
  --manifest data/water_impact_dynamic_v1/train_dynamic_sft_manifest.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/water_impact_dynamic_v1/cache_dynamic_sft_v1 \
  --output-dir outputs/water_impact_dynamic_v1/adapter_dynamic_sft_v1 \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --max-steps 300 \
  --learning-rate 1e-4 \
  --rank 16 \
  --alpha 16 \
  --save-every 50 \
  --seed 26000 \
  --role erase \
  --objective plain
