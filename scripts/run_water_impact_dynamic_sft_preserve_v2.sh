#!/usr/bin/env bash
set -euo pipefail

cd /data/xiaohuang_workspace/ljc/Video-causal

models/.wan-runtime/bin/python scripts/build_water_impact_dynamic_v2_manifest.py
models/.wan-runtime/bin/python scripts/prepare_water_impact_dynamic_v2_cache.py

exec models/.wan-runtime/bin/python scripts/train_wan_waterdrop_lora.py \
  --manifest data/water_impact_dynamic_v1/train_dynamic_sft_preserve_v2.csv \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --cache-dir outputs/water_impact_dynamic_v1/cache_dynamic_sft_preserve_v2 \
  --output-dir outputs/water_impact_dynamic_v1/adapter_dynamic_sft_preserve_v2 \
  --height 480 \
  --width 832 \
  --num-frames 49 \
  --max-steps 200 \
  --learning-rate 5e-5 \
  --rank 16 \
  --alpha 16 \
  --save-every 25 \
  --seed 26000 \
  --role all \
  --objective plain \
  --balanced-roles \
  --preserve-weight 4.0
