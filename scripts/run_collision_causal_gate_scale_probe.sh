#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 LORA_SCALE TAG [CHECKPOINT]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${WAN_PYTHON:-models/.wan-runtime/bin/python}"
MODEL="models/Wan2.1-T2V-1.3B-Diffusers"
LORA_SCALE="$1"
TAG="$2"
CHECKPOINT="${3:-025}"

generate_pair() {
  local split="$1"
  local prompts seeds base_dir
  if [[ "$split" == "target" ]]; then
    prompts="data/collision_validation7.prompts"
    seeds="11013,11019"
    base_dir="outputs/collision_validation7_base"
  else
    prompts="data/collision_specificity8.prompts"
    seeds="12000,12001"
    base_dir="outputs/collision_specificity8_base"
  fi
  local output="outputs/collision_causal_gate_ckpt${CHECKPOINT}_scale${TAG}_${split}2"
  "$PYTHON" scripts/generate_wan_clean.py \
    --baseline clean --prompts "$prompts" --limit 2 --seeds "$seeds" \
    --output-dir "$output" --model "$MODEL" \
    --steps 25 --guidance-scale 5 --num-frames 49 --fps 8 \
    --height 480 --width 832 --dtype bf16 --device cuda \
    --vae-slicing --vae-tiling --skip-existing \
    --lora-path "outputs/adapters/collision_causal_gate_100/checkpoint-000${CHECKPOINT}" \
    --lora-scale "$LORA_SCALE" \
    > "logs/collision_causal_gate_ckpt${CHECKPOINT}_scale${TAG}_${split}2.log" 2>&1
  "$PYTHON" scripts/evaluate_collision_validation7.py \
    --base-dir "$base_dir" --adapter-dir "$output" --limit 2 \
    --output-csv "experiments/pilot_week1/collision_causal_gate_ckpt${CHECKPOINT}_scale${TAG}_${split}2_metrics.csv" \
    --sheet-dir "outputs/collision_causal_gate_ckpt${CHECKPOINT}_scale${TAG}_${split}2_contact_sheets"
}

generate_pair target
generate_pair control
